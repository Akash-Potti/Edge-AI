# Intent_ML Project

## Overview
Intent_ML is a machine learning-based intent classification system designed to recognize user intents from text input. The project leverages deep learning models (GRU and LSTM with attention) and custom tokenization to achieve high accuracy in intent recognition, suitable for conversational AI and edge deployment on microcontrollers such as the ESP32.

## Approach
1. **Data Preparation**: Raw training data is collected and preprocessed. Text samples are tokenized using a custom tokenizer, and labels are encoded for supervised learning.
2. **Model Selection**: The project explores both GRU (Gated Recurrent Unit) and LSTM (Long Short-Term Memory) architectures. Attention mechanisms are integrated to enhance context understanding.
3. **Training**: Models are trained on the processed dataset. Training scripts are provided in Jupyter notebooks for reproducibility and experimentation.
4. **Evaluation**: Model performance is evaluated using accuracy and loss metrics. The best-performing weights are saved for deployment.
5. **Export & Deployment**: Trained weights and tokenizer objects are exported for use in edge devices or embedded systems. Model weights are converted to formats suitable for C++ integration (see `IntentML/include/model_weights.h`), specifically targeting ESP32 compatibility.

## Steps to Build the Project

1. **Prepare the Dataset**
   - Place your training data (e.g., `train.csv`) in the appropriate folder.
   - Ensure the data is cleaned and formatted for intent classification.

2. **Tokenization**
   - Run the tokenizer script/notebook to generate `tokenizer.pkl`.
   - This file will be used for both training and inference.

3. **Model Training**
   - Use the provided Jupyter notebooks (`Training_GRU.ipynb`, etc.) to train the GRU or LSTM models.
   - Adjust hyperparameters as needed for your dataset.
   - Save the best model weights (e.g., `gru_led_intent_weights.weights.h5`).

4. **Model Evaluation**
   - Evaluate the trained model using validation data.
   - Review accuracy and loss plots in the notebook.

5. **Export for ESP32 Deployment**
   - Convert trained weights to C++ header format using the export script/notebook.
   - Place the exported weights in `IntentML/include/model_weights.h` for integration with embedded code.
   - Ensure the exported code and weights are optimized for ESP32 memory and performance constraints.

6. **Integration with ESP32**
   - Use the provided C++ source files (`src/main.cpp`) to run inference on the ESP32.
   - Use the ESP-IDF or Arduino framework to build and flash the firmware.
   - Ensure all dependencies and weights are correctly referenced.

## Folder Structure
- `Convert_Header.ipynb`: Notebook to convert model weights for C++/ESP32 integration.
- `gru_led_intent_weights.weights.h5`: Trained GRU model weights.
- `tokenizer.pkl`: Tokenizer object for text preprocessing.
- `Training_GRU.ipynb`: Notebook for training the GRU model.
- `IntentML/include/model_weights.h`: C++ header file with exported model weights for ESP32.
- `IntentML/src/main.cpp`: Main C++ source for running inference on ESP32.

## Requirements
- Python 3.x
- TensorFlow/Keras
- Jupyter Notebook
- ESP-IDF or Arduino framework (for ESP32 deployment)
- C++ compiler

## How to Run
1. Train the model using the notebook.
2. Export weights for C++/ESP32.
3. Build and flash the C++ code to your ESP32 device using ESP-IDF or Arduino IDE.

## License
This project is for educational and research purposes. See individual files for license details.
