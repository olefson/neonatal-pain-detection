import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import os

# Load Model Function
def load_saved_model(model_dir='saved_model', model_name='project2_part1_model.h5'):
    model_path = os.path.join(model_dir, model_name)
    model = load_model(model_path) # load model
    print(f"Model loaded from: {model_path}")
    return model

# Load/Preprocess test data function
def load_n_preprocess_test_data(data_path="./Data/sub01/Vd01_Sg_001_VS_Set1.csv"): # Put the path to the test data here
    # load test data
    test_data = pd.read_csv(data_path)
    # feature/label extraction
    x_test = test_data.iloc[:, 1:4] # adjust based on data structure
    # normalize
    scaler = StandardScaler()
    x_test = scaler.fit_transform(x_test)
    x_test = np.expand_dims(x_test, axis=1).astype('float32')
    return x_test

# Main test function
def test_model(data_path):
    model = load_saved_model(model_dir='saved_model', model_name='project2_part1_model.h5')
    x_test = load_n_preprocess_test_data(data_path=data_path) # load test data
    predictions = model.predict(x_test) # predict
    predicted_classes = np.argmax(predictions, axis=1)  # Get class predictions
    print("Predicted classes:", predicted_classes)

test_model(data_path="./Data/sub01/Vd01_Sg_001_VS_Set1.csv") # Put path to test data here