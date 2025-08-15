from fastapi import FastAPI, UploadFile, File 
import pandas as pd 
import numpy as np 
import joblib 
import io 
from tensorflow import keras

app = FastAPI() # Load model 
ann = keras.models.load_model("ann.keras") 
# Load preprocessing objects 
le_gender = joblib.load("label_encoder_gender.pkl") 
ct_geo = joblib.load("column_transformer_geo.pkl")

from fastapi.responses import HTMLResponse

@app.get("/")
def home():
    return HTMLResponse("""
    <html>
        <body>
            <h2>Upload CSV for Prediction</h2>
            <form action="/predict_csv" method="post" enctype="multipart/form-data">
                <input type="file" name="file">
                <input type="submit" value="Upload">
            </form>
        </body>
    </html>
    """)

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    # Select same features as in training
    X = df.iloc[:, 3:-1].copy()  # Keep as DataFrame

    # Encode Gender
    X["Gender"] = le_gender.transform(X["Gender"])

    # One-hot encode Geography using the trained column transformer
    X = ct_geo.transform(X)

    # Convert to numpy float32
    X = np.array(X, dtype=np.float32)

    # Predict
    probabilities = ann.predict(X).flatten()
    predictions = (probabilities > 0.5).astype(int)

    # Append predictions
    df["churn_prediction"] = predictions
    df["churn_probability"] = probabilities

    return df.to_dict(orient="records")
