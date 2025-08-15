from fastapi import FastAPI, UploadFile, File
import pandas as pd
import numpy as np
import joblib
import io
from tensorflow import keras

app = FastAPI()

# Load model
ann = keras.models.load_model("ann.keras")

# Load preprocessing objects
le_gender = joblib.load("label_encoder_gender.pkl")
ct_geo = joblib.load("column_transformer_geo.pkl")

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    # Select same features as training
    X = df.iloc[:, 3:-1].values  # Geography, Gender, Age, etc.

    # Apply preprocessing
    X[:, 2] = le_gender.transform(X[:, 2])  # Gender encoding
    X = np.array(ct_geo.transform(X))       # Geography one-hot

    # Predict
    probabilities = ann.predict(X).flatten()
    predictions = (probabilities > 0.5).astype(int)

    # Append predictions to DataFrame
    df["churn_prediction"] = predictions
    df["churn_probability"] = probabilities

    # Return results
    return df.to_dict(orient="records")
