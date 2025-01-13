from fastapi import FastAPI
from autoencoder import Autoencoder
import torch

app = FastAPI()

# Load the saved PyTorch model
model_path = "autoencoder_weights.pth"
model = Autoencoder()
model.load_state_dict(torch.load(model_path))
model.eval()

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI application!"}

# Tasks
# Get the data and return whether anomaly or not. Single data point
# Get the data and return whether anomaly or not. Multiple data points

@app.post("/predict")
def predict(data: dict):
    print(model.state_dict())
    return {"message": "Predicting the anomaly for the given data"}

@app.post("/predict_batch")
def predict_batch(data: list):
    return {"message": "Predicting the anomaly for the given data"}

