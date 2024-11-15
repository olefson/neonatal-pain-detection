# Neonatal Pain Detection Using Deep Learning

## Project Overview
This project focuses on using deep learning techniques to classify pain levels in neonates based on NICU vital signs data. The dataset, collected over five years from USF's medical lab, includes key vital metrics for neonates, providing an essential foundation for building a model that can support early pain detection in neonatal care.

## Models
Two types of recurrent neural networks (RNNs) are utilized:
- **LSTM (Long Short-Term Memory)**: A model known for capturing long-term dependencies.
- **GRU (Gated Recurrent Unit)**: A simplified variant of LSTM, designed for similar tasks but with fewer parameters.

The models were tested with and without **peephole connections** in the LSTM to determine optimal architectures. Techniques like batch normalization, dropout, and L2 regularization were applied to prevent overfitting and improve model generalization.

## Data Preprocessing
Four different preprocessing techniques were implemented:
1. **Standard Scaling**: Normalizing features to have a mean of 0 and a standard deviation of 1.
2. **Data Shuffling**: Shuffling the dataset to ensure random distribution during training and validation splits.
3. **Time-Series Reshaping**: Adjusting the data shape for compatibility with the RNN models.
4. **Batch Normalization**: Applying batch normalization layers to stabilize training and improve convergence.

## Training Details
- **Loss Function**: Sparse Categorical Crossentropy
- **Optimizer**: Adam with dynamic learning rate adjustments
- **Metrics**: Accuracy for evaluation

Each model was trained and validated with different architectures, dropout rates, and preprocessing techniques to achieve optimal performance.

## Repository Structure
- `saved_model/`: Directory to store the trained model files
- `data/`: Contains example dataset files
- `scripts/`: Python scripts for model training, testing, and evaluation
- `notebooks/`: Jupyter notebooks for exploratory data analysis and model experimentation

## How to Run the Project

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   
2. **Train the Model To train the model with the LSTM architecture:**
    python scripts/train_lstm.py

3. **Test the Model To test the saved model on new data:**
    python scripts/test_model.py --data_path <path_to_test_data.csv>

## Results
The model achieved notable accuracy in classifying pain levels across the training and validation datasets. Training and validation metrics are available in the results/ directory, showing accuracy and loss trends over epochs. Batch normalization and dropout layers significantly improved generalization.

## Acknowledgements
This project utilizes data collected from USF’s medical lab and applies advanced deep learning techniques for neonatal pain detection, contributing to the field of neonatal care.