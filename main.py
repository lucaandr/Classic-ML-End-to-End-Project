from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predict import load_model, predict_transaction

app = FastAPI(
    title="Fraud Detection API",
    description="API pentru detectarea tranzactiilor frauduloase",
    version="1.0",
)
model = load_model()


class TransactionData(BaseModel):

    features: list[float]


@app.get("/")
def home():
    return {"message": "API-ul de Detectie Frauda este activ!"}


@app.post("/predict")
def predict(data: TransactionData):
    if len(data.features) != model.n_features_in_:
        raise HTTPException(
            status_code=400,
            detail=f"Tranzactia trebuie sa contina exact {model.n_features_in_} caracteristici.",
        )

    prediction = predict_transaction(
        data.features, model=model, threshold=0.15)
    return prediction
