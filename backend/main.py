from fastapi import FastAPI, UploadFile, File, HTTPException,Depends ,Query
from model.modeloKeras import ModeloStockKeras,reentrenar_modelo_con_diferencias
from pydantic import BaseModel
from datetime import date
from model.registro_advanced import preparar_input_desde_dataset_procesado,all_registers_priductos,procesar_dataset_inventario,buscar_producto_por_id,buscar_producto_por_nombre,buscar_nombre_por_sku,obtener_minimum_stock_level
from llm_service import get_llm_service
from rag_service import get_rag_service
import pandas as pd
import os
import numpy as np
from paths import resolve_file
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from model.database import SessionLocal
from model.registro import Registro
from dias_stock_service import DiasStockService
from rag_service import get_rag_service, crear_router_integrado
# from regex_handlers import procesar_mensaje_simple
from model.funciones import ACTIONS_MAP
app = FastAPI()
dias_stock_service = DiasStockService()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite a todos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# mis_funciones = [
#     {"id": "saludo", "docstring": "hola buenos días saludo inicial bienvenida"},
#     {"id": "despedida", "docstring": "adiós hasta luego cerrar chat terminar"},
#     {"id": "enviar_correo", "docstring": "enviar mandar correo email redactar mensaje electronico"},
#     {"id": "calculo_stock", "docstring": "calcular días stock restante inventario cuanto queda mercadería bodega"}
# ]

# mis_faqs = [
#     {"text": "horario atencion hora abren", "answer": "Atendemos de 9 a 18hs."},
#     {"text": "precio costo valor", "answer": "Los precios dependen del catálogo actual."}
# ]


# router_engine = SemanticRouter(mis_funciones, mis_faqs)
# Inicializar servicio LLM
rag_service = get_rag_service()
router = crear_router_integrado(rag_service)

try:
    llm_service = get_llm_service()
except Exception as e:
    print(f"Advertencia: No se pudo inicializar el servicio LLM: {e}")
    llm_service = None

# Inicializar servicio RAG
try:
    rag_service = get_rag_service()
    print("Servicio RAG inicializado correctamente")
except Exception as e:
    print(f"Advertencia: No se pudo inicializar el servicio RAG: {e}")
    rag_service = None
class ChatInput(BaseModel):
    mensaje: str
# Cargar modelo
modelo = ModeloStockKeras()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/chat")
def chat_endpoint(request: ChatInput):
    """
    Endpoint principal del chatbot con routing inteligente
    """
    decision = router.buscar_intencion(request.mensaje)

    # Caso 1: Conversaciones naturales (saludo/despedida) -> Usar RAG
    if decision["tipo"] == "conversacional":
        subtipo = decision["subtipo"]
        resultado_rag = rag_service.generar_respuesta_conversacional(
            tipo=subtipo,
            mensaje_usuario=request.mensaje
        )
        return {
            "tipo": "conversacional",
            "subtipo": subtipo,
            "score": decision["score"],
            "resultado": {
                    "message": resultado_rag["message"]
                },
            "metadata": {
                "confianza": resultado_rag["confianza"]
            }
        }

    # Caso 2: Acciones del sistema (predicciones, correos, alertas)
    if decision["tipo"] == "accion":
        accion_id = decision["funcion"]

        if accion_id in ACTIONS_MAP:
            resultado = ACTIONS_MAP[accion_id]()
            return {
                "tipo": "accion",
                "accion": accion_id,
                "score": decision["score"],
                "resultado": resultado
            }
        else:
            return {
                "error": f"No existe la acción '{accion_id}' en ACTIONS_MAP"
            }

    # Caso 3: Consulta general (RAG - FAQs o info de empresa)
    resultado = rag_service.responder_pregunta_general(request.mensaje)
    return {
        "tipo": "rag",
        "score": decision.get("score", 0.0),
        "resultado": resultado
            
    }

@app.get("/predictPornombre")
def predict(fecha: str, nombre: str):

    try:
        # Obtener SKU desde el CSV
        sku = buscar_producto_por_nombre(nombre)
        print(sku)
        if sku is None:
            raise HTTPException(
                status_code=404,
                detail=f"No se encontró ningún producto con nombre '{nombre}'"
            )

        # Preparar input del modelo
        features = preparar_input_desde_dataset_procesado(
            sku=sku,
            fecha_override=fecha
        )

        # Realizar predicción
        pred = modelo.predecir(features)
        
        # Obtener nombre completo del producto
        nombre_completo = buscar_nombre_por_sku(sku)
        
        # Obtener minimum_stock_level del producto
        minimum_stock = obtener_minimum_stock_level(sku) or 20.0
        
        # Generar mensaje con LLM
        mensaje_llm = None
        if llm_service:
            try:
                mensaje_llm = llm_service.generar_mensaje_prediccion(
                    nombre_producto=nombre_completo,
                    sku=sku,
                    fecha=fecha,
                    prediccion=float(pred),
                    minimum_stock_level=minimum_stock
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
        sku = buscar_producto_por_id(id)
        
        if sku is None:
            raise HTTPException(
                status_code=404,
                detail=f"No se encontró ningún producto con id '{id}'"
            )

        features = preparar_input_desde_dataset_procesado(
            sku=sku,
            fecha_override=fecha
        )

        pred = modelo.predecir(features)
        
        nombre_producto = buscar_nombre_por_sku(sku)
        
        # Obtener minimum_stock_level del producto
        minimum_stock = obtener_minimum_stock_level(sku) or 20.0
        
        mensaje_llm = None
        if llm_service:
            try:
                mensaje_llm = llm_service.generar_mensaje_prediccion(
                    nombre_producto=nombre_producto,
                    sku=sku,
                    fecha=fecha,
                    prediccion=float(pred),
                    minimum_stock_level=minimum_stock
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
            minimum_stock = obtener_minimum_stock_level(prod) or 20.0
            
            resultados.append({
                "sku": prod,
                "nombre": nombre,
                "prediccion": float(pred),
                "prediction": float(pred),  # mantener compatibilidad
                "minimum_stock": minimum_stock
            })
    
    # Generar mensaje resumen con LLM
    mensaje_resumen = None
    if llm_service and resultados:
        try:
            # Ordenar por predicción (menor a mayor para destacar críticos)
            resultados_ordenados = sorted(resultados, key=lambda x: x['prediccion'])
            
            # Clasificar productos por nivel de stock
            stock_critico = [r for r in resultados if r['prediccion'] < r['minimum_stock']]
            stock_precaucion = [r for r in resultados if r['minimum_stock'] <= r['prediccion'] < r['minimum_stock'] * 1.5]
            stock_adecuado = [r for r in resultados if r['prediccion'] >= r['minimum_stock'] * 1.5]
            
            # Calcular estadísticas
            predicciones_valores = [r['prediccion'] for r in resultados]
            min_pred = min(predicciones_valores)
            max_pred = max(predicciones_valores)
            
            producto_min = next(r for r in resultados if r['prediccion'] == min_pred)
            producto_max = next(r for r in resultados if r['prediccion'] == max_pred)
            
            estadisticas = {
                'promedio': sum(predicciones_valores) / len(predicciones_valores),
                'minimo': min_pred,
                'maximo': max_pred,
                'producto_minimo': producto_min['nombre'],
                'producto_maximo': producto_max['nombre']
            }
            
            mensaje_resumen = llm_service.generar_mensaje_multiple(
                fecha=fecha,
                total_productos=len(resultados),
                predicciones_destacadas=resultados_ordenados,
                stock_critico=stock_critico,
                stock_adecuado=stock_adecuado,
                estadisticas=estadisticas
            )
        except Exception as llm_error:
            print(f"Error generando mensaje resumen: {llm_error}")
            import traceback
            traceback.print_exc()
    
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


        
@app.post("/cargar-csv")
def cargar_csv(db: SessionLocal = Depends(get_db)):
    try:
        df = pd.read_csv("model/files/dataset.csv")

        # Si existe la columna id, eliminarla porque es autoincremental
        if "id" in df.columns:
            df = df.drop(columns=["id"])

        # Convertir columnas fecha
        date_cols = [
            "created_at", "last_order_date", "last_stock_count_date",
            "last_updated_at", "expiration_date"
        ]

        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                df[col] = df[col].astype("object").where(df[col].notna(), None)
                df[col] = df[col].apply(lambda x: x.date() if x is not None else None)

        # Convertir booleanos
        bool_cols = ["vacaciones_o_no", "es_feriado", "temporada_alta", "is_active"]
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().map(
                    {"true": True, "1": True, "false": False, "0": False}
                )

        registros = [
            Registro(**row.dropna().to_dict()) for _, row in df.iterrows()
        ]

        db.add_all(registros)
        db.commit()

        return {"message": "CSV cargado con éxito", "total": len(registros)}

    except Exception as e:
        db.rollback()
        raise HTTPException(500, f"Error cargando CSV: {str(e)}")


@app.get("/check-db-full")
def check_db_full():
    try:
        db = SessionLocal()
        result = db.execute(text("SELECT COUNT(*) FROM registros"))
        total = result.scalar()
        return {"status": "OK", "tabla": "registros", "total_registros": total}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
    finally:
        db.close()
        

@app.get("/analisis-stock")
def analizar_stock_desde_bd(db: SessionLocal = Depends(get_db)):
    
    # 1. Obtener solo el registro más reciente por producto
    registros = (
        db.query(
            Registro.product_name,
            Registro.average_daily_usage,
            Registro.quantity_on_hand,
            Registro.created_at
        )
        .distinct(Registro.product_name)  # ← evita duplicados
        .order_by(Registro.product_name, Registro.created_at.desc())  # ← selecciona el más reciente
        .all()
    )

    if not registros:
        return {"mensaje": "No hay datos para analizar."}

    # 2. Construir lote
    productos_lote = [
        {
            "nombre": r.product_name,
            "stock": r.quantity_on_hand,
            "ventas_diarias": r.average_daily_usage
        }
        for r in registros
    ]

    # 3. Analizar lote
    resultado = dias_stock_service.analizar_lote_productos(productos_lote)

    return resultado

