from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
import boto3
import joblib
import os
import io

app = FastAPI(
    title="Getaround rental pricing optimization",
    description="""
    Bienvenue sur l'API de prédiction des prix Getaround ! 🚗

    Cette API vous permet de prédire le prix de location journalier recommandé pour un véhicule.

    Valeurs acceptées pour les champs :
    * model_key: Citroën, Peugeot, PGO, Renault, Audi, BMW, Ford, Mercedes, Opel, Porsche, Volkswagen, KIA Motors, 
      Alfa Romeo, Ferrari, Fiat, Lamborghini, Maserati, Lexus, Honda, Mazda, Mini, Mitsubishi, Nissan, SEAT, 
      Subaru, Suzuki, Toyota, Yamaha
    * fuel: diesel, petrol, hybrid_petrol, electro
    * paint_color: black, grey, white, red, silver, blue, orange, beige, brown, green
    * car_type: convertible, coupe, estate, hatchback, sedan, subcompact, suv, van
    * mileage: 0 à 300000 km
    * engine_power: 0 à 300 ch

    Les autres champs sont des booléens (true/false) indiquant la présence ou non des options.
    """,
    version="1.0",
    contact={
        "url": "https://github.com/Rom1Legrand"
    }
)

# Schéma pour les données d'entrée
class CarFeatures(BaseModel):
    model_key: str = Field(pattern='^(Citroën|Peugeot|PGO|Renault|Audi|BMW|Ford|Mercedes|Opel|Porsche|Volkswagen|KIA Motors|Alfa Romeo|Ferrari|Fiat|Lamborghini|Maserati|Lexus|Honda|Mazda|Mini|Mitsubishi|Nissan|SEAT|Subaru|Suzuki|Toyota|Yamaha)$')
    mileage: int = Field(ge=0, le=300000)
    engine_power: int = Field(ge=0, le=300)
    fuel: Literal['diesel', 'petrol', 'hybrid_petrol', 'electro']
    paint_color: Literal['black', 'grey', 'white', 'red', 'silver', 'blue', 'orange', 'beige', 'brown', 'green']
    car_type: Literal['convertible', 'coupe', 'estate', 'hatchback', 'sedan', 'subcompact', 'suv', 'van']
    private_parking_available: bool
    has_gps: bool
    has_air_conditioning: bool
    automatic_car: bool
    has_getaround_connect: bool
    has_speed_regulator: bool
    winter_tires: bool

# Configuration S3
bucket = os.getenv('S3_BUCKET')
model_path = "mlflow/models/xgboost_model_4e1d09c075954401b4323321c1c84fc5.joblib"
loaded_model = None
s3 = boto3.client('s3')

@app.on_event("startup")
def load_model():
    global loaded_model
    try:
        model_response = s3.get_object(Bucket=bucket, Key=model_path)
        model_bytes = io.BytesIO(model_response['Body'].read())
        loaded_model = joblib.load(model_bytes)
        print("Modèle chargé avec succès")
    except Exception as e:
        print(f"Erreur lors du chargement : {e}")
        raise RuntimeError(f"Erreur lors du chargement : {e}")

@app.get("/")
def root():
    return {"message": "Bienvenue sur l'API de prédiction de prix pour GetAround 🚗"}

@app.post("/predict")
async def predict(features: CarFeatures):
    try:
        data = pd.DataFrame([features.dict()])
        prediction = loaded_model.predict(data)
        predicted_price = float(prediction[0])
        return {"rental_price": round(predicted_price, 2)}
    except Exception as e:
        print(f"Erreur lors de la prédiction : {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4000, reload=True)