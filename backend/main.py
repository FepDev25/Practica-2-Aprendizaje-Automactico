from fastapi import FastAPI, UploadFile, File, HTTPException
from model.modeloKeras import ModeloStockKeras,reentrenar_modelo_con_diferencias
from pydantic import BaseModel
from datetime import date
from model.registro_advanced import preparar_input_desde_dataset_procesado,all_registers_priductos,procesar_dataset_inventario,buscar_producto_por_id,buscar_producto_por_nombre,buscar_nombre_por_sku
from llm_service import get_llm_service
import pandas as pd
import os
import numpy as np
from paths import resolve_file
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite a todos (para la práctica está bien)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar servicio LLM
try:
    llm_service = get_llm_service()
except Exception as e:
    print(f"Advertencia: No se pudo inicializar el servicio LLM: {e}")
    llm_service = None

# Cargar modelo
modelo = ModeloStockKeras()

@app.get("/")
async def home():
    return {"msg": "Hola mundo."}

@app.get("/modelo/info")
async def info_modelo():
    resumen = modelo.obtener_resumen()
    return {"resumen": resumen}

@app.get("/predictPornombre")
def predict(fecha: str, nombre: str):

    try:
        # 1. Obtener SKU desde el CSV
        sku = buscar_producto_por_nombre(nombre)
        print(sku)
        if sku is None:
            raise HTTPException(
                status_code=404,
                detail=f"No se encontró ningún producto con nombre '{nombre}'"
            )

        # 2. Preparar input del modelo
        features = preparar_input_desde_dataset_procesado(
            sku=sku,
            fecha_override=fecha
        )

        # 3. Realizar predicción
        pred = modelo.predecir(features)
        
        # 4. Obtener nombre completo del producto
        nombre_completo = buscar_nombre_por_sku(sku)
        
        # 5. Generar mensaje amigable con LLM
        mensaje_llm = None
        if llm_service:
            try:
                mensaje_llm = llm_service.generar_mensaje_prediccion(
                    nombre_producto=nombre_completo,
                    sku=sku,
                    fecha=fecha,
                    prediccion=float(pred)
                )
            except Exception as llm_error:
                print(f"Error generando mensaje LLM: {llm_error}")

        return {
            "nombre_ingresado": nombre,
            "nombre_producto": nombre_completo,
            "sku_detectado": sku,
            "fecha_prediccion": fecha,
            "prediction": float(pred),
            "mensaje": mensaje_llm or f"Predicción para {nombre_completo}: {pred:.2f} unidades disponibles para {fecha}"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al predecir: {str(e)}"
        )

@app.get("/predictPorID")
def predict(fecha: str, id: int):

    try:
        # 1. Obtener SKU desde el CSV
        sku = buscar_producto_por_id(id)
        
        if sku is None:
            raise HTTPException(
                status_code=404,
                detail=f"No se encontró ningún producto con id '{id}'"
            )

        # 2. Preparar input del modelo
        features = preparar_input_desde_dataset_procesado(
            sku=sku,
            fecha_override=fecha
        )

        # 3. Realizar predicción
        pred = modelo.predecir(features)
        
        # 4. Obtener nombre del producto
        nombre_producto = buscar_nombre_por_sku(sku)
        
        # 5. Generar mensaje amigable con LLM
        mensaje_llm = None
        if llm_service:
            try:
                mensaje_llm = llm_service.generar_mensaje_prediccion(
                    nombre_producto=nombre_producto,
                    sku=sku,
                    fecha=fecha,
                    prediccion=float(pred)
                )
            except Exception as llm_error:
                print(f"Error generando mensaje LLM: {llm_error}")

        return {
            "id_ingresado": id,
            "nombre_producto": nombre_producto,
            "sku_detectado": sku,
            "fecha_prediccion": fecha,
            "prediction": float(pred),
            "mensaje": mensaje_llm or f"Predicción para {nombre_producto}: {pred:.2f} unidades disponibles para {fecha}"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al predecir: {str(e)}"
        )

@app.get("/predictAll")
def predict(fecha: str):
    productos = all_registers_priductos()
    resultados = []
    
    for prod in productos:
        features = preparar_input_desde_dataset_procesado(sku=prod,fecha_override=fecha)
        if features is not None and np.any(features):  # validar que exista registro
            pred = modelo.predecir(features)
            nombre = buscar_nombre_por_sku(prod)
            resultados.append({
                "sku": prod,
                "nombre": nombre,
                "prediction": float(pred)
            })
    
    # Generar mensaje resumen con LLM
    mensaje_resumen = None
    if llm_service and resultados:
        try:
            # Ordenar por predicción (menor a mayor para destacar críticos)
            resultados_ordenados = sorted(resultados, key=lambda x: x['prediction'])
            
            mensaje_resumen = llm_service.generar_mensaje_multiple(
                fecha=fecha,
                total_productos=len(resultados),
                predicciones_destacadas=resultados_ordenados[:10]  # Top 10 más críticos
            )
        except Exception as llm_error:
            print(f"Error generando mensaje resumen: {llm_error}")
    
    # lista de productos y sus predicciones 
    return {
        "fecha_prediccion": fecha,
        "total_productos": len(resultados),
        "predictions": resultados,
        "mensaje_resumen": mensaje_resumen or f"Se analizaron {len(resultados)} productos para la fecha {fecha}"
    }


@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    
    DATASET_PATH = "model/files/dataset.csv"
    
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
def reentrenar_modelo():
    try:
        resultado = reentrenar_modelo_con_diferencias()

        return {
            "status": "ok",
            "mensaje": "Reentrenamiento completado",
            "resultado": resultado
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante el reentrenamiento: {str(e)}"
        )