from fastapi import FastAPI
from model.modeloKeras import ModeloKeras
from pydantic import BaseModel

app = FastAPI()

modelo = ModeloKeras()

class InputData(BaseModel):
    features: list[float]

@app.get("/")
async def home():
    
    return {"msg": "Hola mundo."}

@app.get("/modelo/info")
async def info_modelo():
    resumen = modelo.obtener_resumen()
    return {"resumen": resumen}


@app.post("/predict")
def predict(data: InputData):
    try:
        prediction = modelo.predecir(data.features)
        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}