#include <Arduino.h>
#include "model_weights.h"
#include <math.h>
#include <string.h>

#define MAX_INPUT_LEN 64
#define LED_PIN 21 // GPIO pin for LED
char input_line[MAX_INPUT_LEN];

void embed_input(const int *input, float *output)
{
  // output shape: SEQ_LEN x EMBED_DIM
  for (int i = 0; i < SEQ_LEN; i++)
  {
    int idx = input[i];
    for (int j = 0; j < EMBED_DIM; j++)
    {
      output[i * EMBED_DIM + j] = embedding_weights[idx * EMBED_DIM + j];
    }
  }
}

void sigmoid_vec(float *x, int len)
{
  for (int i = 0; i < len; i++)
  {
    x[i] = 1.0f / (1.0f + exp(-x[i]));
  }
}

void tanh_vec(float *x, int len)
{
  for (int i = 0; i < len; i++)
  {
    x[i] = tanhf(x[i]);
  }
}

void matvec_mul(const float *mat, const float *vec, float *out, int rows, int cols)
{
  for (int i = 0; i < rows; i++)
  {
    out[i] = 0;
    for (int j = 0; j < cols; j++)
    {
      out[i] += mat[i * cols + j] * vec[j];
    }
  }
}

void gru_forward(const float *input_seq, float *hidden_out)
{
  float h[GRU_UNITS] = {0};
  for (int t = 0; t < SEQ_LEN; t++)
  {
    float x_t[EMBED_DIM];
    memcpy(x_t, &input_seq[t * EMBED_DIM], sizeof(float) * EMBED_DIM);
    float gates[3 * GRU_UNITS];
    float gates_r[3 * GRU_UNITS];

    // W*x
    matvec_mul(gru_kernel, x_t, gates, 3 * GRU_UNITS, EMBED_DIM);

    // U*h + b
    matvec_mul(gru_recurrent, h, gates_r, 3 * GRU_UNITS, GRU_UNITS);
    for (int i = 0; i < 3 * GRU_UNITS; i++)
    {
      gates[i] += gates_r[i] + gru_bias[i];
    }
    float *z = gates;
    float *r = gates + GRU_UNITS;
    float *h_hat_pre = gates + 2 * GRU_UNITS;
    sigmoid_vec(z, GRU_UNITS);
    sigmoid_vec(r, GRU_UNITS);

    float r_h[GRU_UNITS];
    for (int i = 0; i < GRU_UNITS; i++)
    {
      r_h[i] = h[i] * r[i];
    }

    float h_hat[GRU_UNITS];
    matvec_mul(&gru_recurrent[2 * GRU_UNITS * GRU_UNITS], r_h, h_hat, GRU_UNITS, GRU_UNITS);
    matvec_mul(&gru_kernel[2 * GRU_UNITS * EMBED_DIM], x_t, h_hat_pre, GRU_UNITS, EMBED_DIM);

    for (int i = 0; i < GRU_UNITS; i++)
    {
      h_hat[i] += h_hat_pre[i] + gru_bias[2 * GRU_UNITS + i];
    }

    tanh_vec(h_hat, GRU_UNITS);

    for (int i = 0; i < GRU_UNITS; i++)
    {
      h[i] = (1 - z[i]) * h[i] + z[i] * h_hat[i];
    }
  }

  memcpy(hidden_out, h, sizeof(float) * GRU_UNITS);
}
void dense_softmax(const float *hidden, float *output)
{
  float logits[OUTPUT_CLASSES];

  for (int i = 0; i < OUTPUT_CLASSES; i++)
  {
    logits[i] = 0;
    for (int j = 0; j < GRU_UNITS; j++)
    {
      logits[i] += dense_kernel[j * OUTPUT_CLASSES + i] * hidden[j];
    }
    logits[i] += dense_bias[i];
  }

  // softmax
  float max_logit = logits[0];
  for (int i = 1; i < OUTPUT_CLASSES; i++)
    if (logits[i] > max_logit)
      max_logit = logits[i];

  float sum = 0;
  for (int i = 0; i < OUTPUT_CLASSES; i++)
  {
    logits[i] = expf(logits[i] - max_logit); // stability
    sum += logits[i];
  }

  for (int i = 0; i < OUTPUT_CLASSES; i++)
  {
    output[i] = logits[i] / sum;
  }
}

int infer_intent(const int *input_sequence)
{
  float embedded[SEQ_LEN * EMBED_DIM];
  embed_input(input_sequence, embedded);

  float hidden[GRU_UNITS];
  gru_forward(embedded, hidden);

  float output[OUTPUT_CLASSES];
  dense_softmax(hidden, output);

  // Return argmax
  return output[0] > output[1] ? 0 : 1; // 0 = OFF, 1 = ON
}

void tokenize_input(const char *line, int *tokens)
{
  char copy[MAX_INPUT_LEN];
  strncpy(copy, line, MAX_INPUT_LEN);
  copy[MAX_INPUT_LEN - 1] = '\0'; // Ensure null-terminated

  int i = 0;
  char *word = strtok(copy, " ");
  while (word != NULL && i < SEQ_LEN)
  {
    tokens[i] = 0; // Default to PAD if not found
    for (int j = 1; j < VOCAB_SIZE; j++)
    { // skip PAD at index 0
      if (strcmp(word, vocab[j]) == 0)
      {
        tokens[i] = j;
        break;
      }
    }
    word = strtok(NULL, " ");
    i++;
  }

  // Pad remaining
  while (i < SEQ_LEN)
  {
    tokens[i++] = 0;
  }
}

void setup()
{
  Serial.begin(115200);
  Serial.println("ESP32 GRU Inference Ready!");
  Serial.println("Type a command like: turn on led");
  Serial.println();
  pinMode(LED_PIN, OUTPUT);
}
void loop()
{
  if (Serial.available())
  {
    int len = Serial.readBytesUntil('\n', input_line, MAX_INPUT_LEN - 1);
    input_line[len] = '\0';

    int input_sequence[SEQ_LEN];
    tokenize_input(input_line, input_sequence);

    Serial.print("Tokenized: ");
    for (int i = 0; i < SEQ_LEN; i++)
    {
      Serial.print(input_sequence[i]);
      Serial.print(" ");
    }
    Serial.println();

    int intent = infer_intent(input_sequence);
    if (intent == 1)
    {
      Serial.println("Intent: LED ON");
      digitalWrite(LED_PIN, HIGH);
    }
    else
    {
      Serial.println("Intent: LED OFF");
      digitalWrite(LED_PIN, LOW);
    }

    Serial.println("-----------------------------");
  }
}
