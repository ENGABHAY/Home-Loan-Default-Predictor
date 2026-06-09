from fastapi import FastAPI, UploadFile, File, HTTPException
from predictor import LoanDefaultPredictor
import pandas as pd
from typing import List
import uvicorn

app = FastAPI(title="Home Credit Loan Default Predictor")

predictor = LoanDefaultPredictor()

@app.post("/predict")
async def predict_loan_default(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        
        if 'SK_ID_CURR' not in df.columns:
            raise HTTPException(status_code=400, detail="SK_ID_CURR column is required")
            
        probabilities = predictor.predict_proba(df)
        predictions = (probabilities >= 0.5).astype(int)
        
        result = pd.DataFrame({
            'SK_ID_CURR': df['SK_ID_CURR'],
            'default_probability': probabilities.round(6),
            'will_default': predictions
        })
        
        return result.to_dict(orient='records')
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    return await predict_loan_default(file)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)