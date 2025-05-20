# BiLSTM Project

This project explores the use of a Bidirectional Long Short-Term Memory (BiLSTM) network for audio classification, specifically focusing on distinguishing between real and fake audio samples. It incorporates both MFCC (Mel-Frequency Cepstral Coefficients) features and Mel-spectrogram images to leverage different aspects of the audio data.

## Project Structure

The repository contains the following files:

*   `CNN_BiLSTM.ipynb`:  A combined CNN-BiLSTM model, integrating both MFCC features and Mel-spectrogram images for audio classification. It includes data loading, preprocessing, model definition, training, and evaluation steps.
*    `CNN_BiLSTM_gelu.ipynb`: Implements a combined CNN-BiLSTM architecture, using MFCC features and Mel-spectrogram images. The CNN part of the model uses the GELU activation function.
*   `GDP_Data.csv`: Contains GDP data, potentially used as a feature or for analysis in relation to the audio data (details of its use are not explicitly defined in the provided code).
*   `Government_Debt_Data.csv`: Government Debt Data, potentially used as a feature or for analysis in relation to the audio data (details of its use are not explicitly defined in the provided code).
*   `Inflation_Rate_Data.csv`: Inflation Rate Data, potentially used as a feature or for analysis in relation to the audio data (details of its use are not explicitly defined in the provided code).
*   `Interest_Rate_Data.csv`: Interest Rate Data, potentially used as a feature or for analysis in relation to the audio data (details of its use are not explicitly defined in the provided code).
*   `LICENSE`: Contains the project's license information.
*   `README.md`: This file, providing an overview of the project.
*   `Unemployment_Rate_Data.csv`: Unemployment Rate Data, potentially used as a feature or for analysis in relation to the audio data (details of its use are not explicitly defined in the provided code).
*   `bilstmtest.py`:  Implementation of a BiLSTM model (it seems, this is not used in the Colab notebooks), focusing on text processing and classification using word embeddings. It includes functionalities for training, evaluation, and prediction.
*   `submission_maker.ipynb`: Notebook designed to generate a submission file based on the model's predictions.
*   `/content/drive/MyDrive/dataset/TeamDeepwave/dataset/open/train`: Directory containing the training audio files in `.ogg` format.
*   `/content/drive/MyDrive/dataset/TeamDeepwave/dataset/open/train.csv`: CSV file containing the training labels and file paths.
*   `/content/drive/MyDrive/dataset/TeamDeepwave/dataset/open/train_mfcc.npy`: Numpy array file storing the training MFCC (Mel-Frequency Cepstral Coefficients) features.
*   `/content/drive/MyDrive/dataset/TeamDeepwave/dataset/open/train_labels.npy`: Numpy array file storing the training labels.
*   `/content/drive/MyDrive/dataset/TeamDeepwave/dataset/open/val_mfcc.npy`: Numpy array file storing the validation MFCC features.
*   `/content/drive/MyDrive/dataset/TeamDeepwave/dataset/open/val_labels.npy`: Numpy array file storing the validation labels.
*   `/content/drive/MyDrive/dataset/TeamDeepwave/dataset/preprocessed/`: Directory to store preprocessed audio data.

## Key Components and Functionality

### 1. Data Loading and Preprocessing

*   **Audio Data:** The project uses `.ogg` audio files located in the `/content/drive/MyDrive/dataset/TeamDeepwave/dataset/open/train` directory.
*   **Feature Extraction:**
    *   **MFCC:** Mel-Frequency Cepstral Coefficients are extracted using `librosa`. The code pads or truncates MFCC sequences to a fixed length (`MAX_SEQ_LEN`).
    *   **Mel-spectrogram:** Mel-spectrogram images are generated from audio files. These images are resized, converted to grayscale, and normalized.
*   **Data Splitting:** The dataset is split into training and validation sets using `train_test_split` from `sklearn.model_selection`.
*   **Custom Dataset:** A `CustomDataset` class is defined using `torch.utils.data.Dataset` to handle loading and preprocessing of audio data and labels.

### 2. Model Architectures

*   **BiLSTM (bilstmtest.py):**
    *   A Bidirectional LSTM network is implemented for text classification.
    *   It uses pre-trained word embeddings (e.g., from `wiki-news-300d-1M`) to initialize the embedding layer.
    *   The network consists of an embedding layer, a BiLSTM layer, and a linear layer for classification.
*   **CNN (CNN\_BiLSTM.ipynb and CNN\_BiLSTM\_gelu.ipynb):**
    *   A Convolutional Neural Network is designed for processing Mel-spectrogram images.
    *   It includes convolutional layers, pooling layers, and fully connected layers.
    *   The `CNN_BiLSTM_gelu.ipynb` notebook utilizes the GELU activation function in the CNN layers.
*   **Combined CNN-BiLSTM (CNN\_BiLSTM.ipynb and CNN\_BiLSTM\_gelu.ipynb):**
    *   Combines the BiLSTM (for MFCC features) and CNN (for Mel-spectrogram images) models.
    *   Concatenates the outputs of the BiLSTM and CNN layers and feeds them into a fully connected layer for final classification.

### 3. Training and Evaluation

*   **Training Loop:** The training loop iterates through the training data, calculates the loss, and updates the model's parameters using an optimizer (e.g., Adam).
*   **Loss Function:** Binary Cross Entropy with Logits Loss (`BCEWithLogitsLoss`) is used as the loss function in `bilstmtest.py`, while `CrossEntropyLoss` is used in `CNN_BiLSTM.ipynb` and `CNN_BiLSTM_gelu.ipynb`.
*   **Evaluation:** The model's performance is evaluated on a validation set, and metrics such as accuracy and F1-score are calculated.
*   **Model Saving:** The best-performing model is saved during training.

### 4. Submission Generation

*   The `submission_maker.ipynb` notebook is used to generate a submission file in the required format, based on the model's predictions on the test data.

## Usage

1.  **Install Dependencies:** Ensure that you have the necessary libraries installed. You can install them using `pip`:

    ```bash  
    pip install librosa numpy pandas scikit-learn torch torchtext torchvision pillow  