from fastapi import FastAPI
from model.modeloKeras import ModeloStockRF
from pydantic import BaseModel
from datetime import date
from model.registro_advanced import obtener_registro_advanced_por_fecha,all_registers_priductos

app = FastAPI()

# Cargar modelo
modelo = ModeloStockRF()

# Modelo del body del POST
class PredictInput(BaseModel):
    
    prioridad_proveedor: int
    quantity_on_hand: float
    quantity_reserved: float
    
    minimum_stock_level: float
    reorder_point: float
    optimal_stock_level: float
    reorder_quantity: float
    average_daily_usage: float
    unit_cost: float
    total_value: float
    
    is_active: int
    anio: int
    mes: int
    vacaciones_o_no: int
    es_feriado: int
    temporada_alta: int
    semana_del_anio: int
    dia_del_mes: int
    dia_de_la_semana: int
    es_fin_de_semana: int
    trimestre: int
    lag_1: float
    lag_7: float
    lag_30: float
    media_movil_7d: float
    media_movil_30d: float
    media_movil_exponencial: float
    std_movil_30d: float
    variacion_stock_diaria: float
    tendencia_stock: float
    dias_desde_ultimo_pedido: int
    ratio_reservado_disponible: float
    anomalia_stock: float
    rotacion_estimada_30d: float
    estacion_Otoño: bool
    estacion_Primavera: bool
    estacion_Verano: bool


@app.get("/")
async def home():
    return {"msg": "Hola mundo."}

@app.get("/modelo/info")
async def info_modelo():
    resumen = modelo.obtener_resumen()
    return {"resumen": resumen}

@app.get("/predict")
def predict(fecha: str, id: str):
    features = obtener_registro_advanced_por_fecha(sku=id,fecha=fecha)
    pred = modelo.predecir(features)

    return {
        "sku": id,
        "prediction": pred,
    }
    
@app.get("/predictAll")
def predict(fecha: str):
    productos = all_registers_priductos()
    resultados = []

    for prod in productos:
        features = obtener_registro_advanced_por_fecha(sku=prod, fecha=fecha)
        if features:  # validar que exista registro
            pred = modelo.predecir(features)
            resultados.append({
                "sku": prod,
                "prediction": pred
            })
    # lista de productos y sus predicciones 
    return {
        "predictions": resultados
    }
