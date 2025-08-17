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
sc = joblib.load("scaler.pkl")
from fastapi.responses import HTMLResponse

@app.get("/")
def home():
    return HTMLResponse("""
    <html>
        <body>
            <h2>Upload CSV for Prediction</h2>
            <form action="/predict_csv" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept=".csv">
                <input type="submit" value="Upload">
            </form>
            <h3>Expected CSV Format</h3>
            <table border="1" cellpadding="5" cellspacing="0">
                <tr><th>Column Name</th><th>Data Type</th></tr>
                <tr><td>RowNumber</td><td>int64</td></tr>
                <tr><td>CustomerId</td><td>int64</td></tr>
                <tr><td>Surname</td><td>object (string)</td></tr>
                <tr><td>CreditScore</td><td>int64</td></tr>
                <tr><td>Geography</td><td>object (string)</td></tr>
                <tr><td>Gender</td><td>object (string: Male/Female)</td></tr>
                <tr><td>Age</td><td>int64</td></tr>
                <tr><td>Tenure</td><td>int64</td></tr>
                <tr><td>Balance</td><td>float64</td></tr>
                <tr><td>NumOfProducts</td><td>int64</td></tr>
                <tr><td>HasCrCard</td><td>int64 (0 or 1)</td></tr>
                <tr><td>IsActiveMember</td><td>int64 (0 or 1)</td></tr>
                <tr><td>EstimatedSalary</td><td>float64</td></tr>
                <tr><td>Exited</td><td>int64 (Target column: 0 or 1)</td></tr>
            </table>
            <p><b>Note:</b> Ensure the CSV has all the above columns in the same format.</p>
        </body>
    </html>
    """)

from fastapi.responses import HTMLResponse

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    # Select same features as in training
    X = df.iloc[:, 3:-1].copy()

    # Encode Gender
    X["Gender"] = le_gender.transform(X["Gender"])

    # One-hot encode Geography
    X = ct_geo.transform(X)
    X = sc.transform(X)
    X = np.array(X, dtype=np.float32)

    # Predict
    probabilities = ann.predict(X).flatten()
    predictions = (probabilities > 0.5).astype(int)

    # Append predictions
    df["churn_prediction"] = predictions
    df["churn_probability"] = probabilities
    df["Result"] = df["churn_prediction"].apply(
        lambda x: "✅ The Customer is NOT expected to leave" if x == 0 
                  else "⚠️ The Customer is expected to leave"
    )

    # Convert to HTML table
    html_table = df.to_html(classes="table table-bordered", index=False)

    return HTMLResponse(content=f"""
    <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .table {{ border-collapse: collapse; width: 100%; }}
                .table th, .table td {{ border: 1px solid #ddd; padding: 8px; }}
                .table th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h2>Prediction Results</h2>
            {html_table}
        </body>
    </html>
    """)

