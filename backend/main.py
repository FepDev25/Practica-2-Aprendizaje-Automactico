from fastapi import FastAPI, UploadFile, File, HTTPException
from model.modeloKeras import ModeloStockKeras
from pydantic import BaseModel
from datetime import date
from model.registro_advanced import preparar_input_desde_dataset_procesado,all_registers_priductos,procesar_dataset_inventario
import pandas as pd
import os

app = FastAPI()

# Cargar modelo
modelo = ModeloStockKeras()

@app.get("/")
async def home():
    return {"msg": "Hola mundo."}

@app.get("/modelo/info")
async def info_modelo():
    resumen = modelo.obtener_resumen()
    return {"resumen": resumen}

@app.get("/predict")
def predict(fecha: str, id: str):
    features = preparar_input_desde_dataset_procesado(
    "BCA-115",
     "2025-12-05"
    )
    
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
        features = preparar_input_desde_dataset_procesado(sku=prod, fecha=fecha)
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


@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    
    DATASET_PATH = "dataset.csv"
    
    REQUIRED_COLUMNS = [
        "id","created_at","product_id","product_name","product_sku","supplier_id",
        "supplier_name","prioridad_proveedor","quantity_on_hand","quantity_reserved",
        "quantity_available","minimum_stock_level","reorder_point","optimal_stock_level",
        "reorder_quantity","average_daily_usage","last_order_date","last_stock_count_date",
        "unit_cost","total_value","expiration_date","batch_number","warehouse_location",
        "shelf_location","region_almacen","stock_status","is_active","last_updated_at",
        "created_by_id","record_sequence_number","categoria_producto","subcategoria_producto",
        "anio","mes","vacaciones_o_no","es_feriado","temporada_alta"
    ]
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="El archivo debe ser .csv")

    # Leer CSV entrante
    try:
        df_new = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer CSV: {str(e)}")

    # Validar columnas
    missing = set(REQUIRED_COLUMNS) - set(df_new.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Columnas faltantes: {', '.join(missing)}"
        )

    # Si existe dataset.csv → cargar
    if os.path.exists(DATASET_PATH):
        df_existing = pd.read_csv(DATASET_PATH)

        # Validar que las columnas coincidan exactamente
        if list(df_existing.columns) != REQUIRED_COLUMNS:
            raise HTTPException(
                status_code=500,
                detail="El dataset.csv existente no coincide con el esquema requerido."
            )

    else:
        # Si no existe, iniciarlo vacío con columnas correctas
        df_existing = pd.DataFrame(columns=REQUIRED_COLUMNS)

    rows_before = len(df_existing)

    # Ordenar columnas del CSV cargado según definición
    df_new = df_new[REQUIRED_COLUMNS]

    # Agregar nuevas filas al final del dataset
    df_final = pd.concat([df_existing, df_new], ignore_index=True)

    # Guardar dataset actualizado
    df_final.to_csv(DATASET_PATH, index=False)
    procesar_dataset_inventario()
    
    return {
        "message": "Nuevas filas agregadas correctamente al final del dataset.",
        "rows_before": rows_before,
        "rows_added": len(df_new),
        "rows_after": len(df_final),
        "path": DATASET_PATH
    }
    
    
@app.get("/reentrenarModelo")
def contar_diferencias():

    # Leer directamente desde archivos locales
    df1 = pd.read_csv("dataset_processed_advanced.csv")
    df2 = pd.read_csv("dataset_processed_advanced2.csv")

    # Alinear columnas
    for col in df1.columns:
        if col not in df2.columns:
            df2[col] = None

    for col in df2.columns:
        if col not in df1.columns:
            df1[col] = None

    df1 = df1[df2.columns]

    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)

    # Detectar filas distintas
    filas_diferentes = df1.ne(df2).any(axis=1)

    numero_diferencias = filas_diferentes.sum()

    return {
        "filas_diferentes": int(numero_diferencias)
    }