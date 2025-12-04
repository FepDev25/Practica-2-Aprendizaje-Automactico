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
from model.semantic_engine import SemanticRouter
from model.database import SessionLocal
from model.registro import Registro
from dias_stock_service import DiasStockService
from rag_service import get_rag_service, crear_router_integrado
from regex_handlers import procesar_mensaje_simple

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
import time
import re
from model.funciones import FUNCIONES_DISPONIBLES


from rag_service import get_rag_service, crear_router_integrado
from company_info import COMPANY_INFO

def extraer_parametros_del_mensaje(mensaje: str) -> Dict:
    """
    Extrae parámetros útiles del mensaje del usuario
    
    Returns:
        Dict con parámetros encontrados
    """
    params = {}
    
    # Extraer SKU
    sku_match = re.search(r'SKU[:\s-]*(\w+)', mensaje, re.IGNORECASE)
    if sku_match:
        params['sku'] = sku_match.group(1)
    
    # Extraer emails
    emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', mensaje)
    if emails:
        params['destinatarios'] = emails
    
    # Extraer números (podrían ser días, cantidades, etc.)
    numeros = re.findall(r'\b(\d+)\s*días?\b', mensaje, re.IGNORECASE)
    if numeros:
        params['dias'] = int(numeros[0])
    
    # Detectar palabras clave de urgencia
    if any(palabra in mensaje.lower() for palabra in ['urgente', 'crítico', 'inmediato']):
        params['urgencia'] = 'alta'
    
    # Detectar si pide envío de email
    if any(palabra in mensaje.lower() for palabra in ['enviar', 'mandar', 'correo', 'email', 'notificar']):
        params['enviar_email'] = True
    
    return params


def procesar_mensaje_simple(mensaje: str) -> Optional[Dict]:
    """
    Procesa mensajes simples con regex (Nivel 1)
    """
    mensaje_lower = mensaje.lower().strip()
    
    # Saludos
    if re.match(r'^(hola|buenos días|buenas tardes|buenas noches|hey|hi)$', mensaje_lower):
        return {
            "respuesta": f"¡Hola! 👋 Soy el asistente virtual de UPS Tuti. Puedo ayudarte con:\n\n" +
                        "• Predicciones de stock\n" +
                        "• Alertas de inventario\n" +
                        "• Búsqueda de productos\n" +
                        "• Envío de reportes\n\n" +
                        "¿En qué puedo ayudarte?",
            "tipo": "saludo"
        }
    
    # Despedidas
    if re.match(r'^(adiós|chao|hasta luego|bye|nos vemos)$', mensaje_lower):
        return {
            "respuesta": "¡Hasta pronto! 👋 Si necesitas algo más, aquí estaré.",
            "tipo": "despedida"
        }
    
    # Agradecimientos
    if re.match(r'^(gracias|muchas gracias|thanks|thank you)$', mensaje_lower):
        return {
            "respuesta": "¡De nada! 😊 ¿Hay algo más en lo que pueda ayudarte?",
            "tipo": "agradecimiento"
        }
    
    # Help
    if re.match(r'^(ayuda|help|qué puedes hacer|opciones)$', mensaje_lower):
        return {
            "respuesta": "Puedo ayudarte con:\n\n" +
                        "**Predicciones**: 'predecir stock del SKU-123'\n" +
                        "**Alertas**: 'generar alerta de bajo stock'\n" +
                        "**Búsquedas**: 'buscar producto galletas'\n" +
                        "**Reportes**: 'enviar reporte a gerencia@upstuti.com'\n\n" +
                        "También puedo responder preguntas sobre la empresa.",
            "tipo": "ayuda"
        }
    
    return None


class ChatInput(BaseModel):
    mensaje: str = Field(..., min_length=1, max_length=500, description="Mensaje del usuario")
    usuario_id: Optional[str] = Field(None, description="ID del usuario para tracking")
    contexto: Optional[Dict[str, Any]] = Field(None, description="Contexto adicional")
    
    @validator('mensaje')
    def mensaje_no_vacio(cls, v):
        if not v.strip():
            raise ValueError('El mensaje no puede estar vacío')
        return v.strip()


class ChatResponse(BaseModel):
    respuesta: str
    metodo: str
    tipo: Optional[str] = None
    confianza: Optional[float] = None
    fuentes: Optional[List] = None
    resultado_funcion: Optional[Dict] = None
    metadata: Optional[Dict] = None
    tiempo_procesamiento: Optional[float] = None


class ErrorResponse(BaseModel):
    error: str
    detalle: Optional[str] = None
    codigo: str
    timestamp: str

app = FastAPI()

# Variables globales para servicios
dias_stock_service = None
router_engine = None
rag_service = None
router = None
llm_service = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite a todos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """
    Inicializa todos los servicios pesados durante el startup
    para evitar segfaults por carga durante requests
    """
    global dias_stock_service, router_engine, rag_service, router, llm_service
    
    print("Iniciando carga de servicios...")
    
    # 1. Servicio de días de stock (ligero)
    try:
        dias_stock_service = DiasStockService()
        print("DiasStockService cargado")
    except Exception as e:
        print(f"Error cargando DiasStockService: {e}")
    
    # 2. Semantic Router (ligero)
    try:
        mis_funciones = [
            {"id": "saludo", "docstring": "hola buenos días saludo inicial bienvenida"},
            {"id": "despedida", "docstring": "adiós hasta luego cerrar chat terminar"},
            {"id": "enviar_correo", "docstring": "enviar mandar correo email redactar mensaje electronico"},
            {"id": "calculo_stock", "docstring": "calcular días stock restante inventario cuanto queda mercadería bodega"}
        ]
        mis_faqs = [
            {"text": "horario atencion hora abren", "answer": "Atendemos de 9 a 18hs."},
            {"text": "precio costo valor", "answer": "Los precios dependen del catálogo actual."}
        ]
        router_engine = SemanticRouter(mis_funciones, mis_faqs)
        print("SemanticRouter cargado")
    except Exception as e:
        print(f"Error cargando SemanticRouter: {e}")
    
    # 3. Servicio RAG (pesado - FAISS + embeddings)
    try:
        rag_service = get_rag_service()
        print("RAG Service cargado")
    except Exception as e:
        print(f"Error cargando RAG Service: {e}")
        rag_service = None
    
    # 4. Router integrado (requiere RAG)
    try:
        if rag_service:
            router = crear_router_integrado(rag_service)
            print("Router integrado cargado")
    except Exception as e:
        print(f"Error cargando Router integrado: {e}")
        router = None
    
    # 5. Servicio LLM (opcional)
    try:
        llm_service = get_llm_service()
        print("LLM Service cargado")
    except Exception as e:
        print(f"LLM Service no disponible: {e}")
        llm_service = None
    
    print("🎉 Todos los servicios iniciados")

def convertir_numpy_a_python(obj):
    """Convierte tipos numpy a tipos nativos de Python para serialización JSON"""
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convertir_numpy_a_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convertir_numpy_a_python(item) for item in obj]
    return obj

# Variable global para el modelo
modelo = None

@app.on_event("startup")
async def load_keras_model():
    """
    Carga el modelo Keras durante el startup en un hilo separado
    para evitar bloquear la inicialización
    """
    global modelo
    try:
        import gc
        import tensorflow as tf
        
        # Configurar TensorFlow para usar menos memoria
        tf.config.set_soft_device_placement(True)
        
        # Limitar memoria GPU si está disponible
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        
        print("Cargando modelo Keras...")
        modelo = ModeloStockKeras()
        print("Modelo Keras cargado exitosamente")
        
        # Forzar garbage collection
        gc.collect()
        
    except Exception as e:
        print(f"Error cargando modelo Keras: {e}")
        modelo = None
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
async def home():
    return {"msg": "Hola mundo."}

@app.get("/modelo/info")
async def info_modelo():
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    resumen = modelo.obtener_resumen()
    return {"resumen": resumen}

@app.get("/predictPornombre")
def predict(fecha: str, nombre: str):
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

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
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

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
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
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

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(
    data: ChatInput,
    db: Session = Depends(get_db)
):
    """
    🤖 Endpoint principal de chat con routing inteligente de 3 niveles
    """
    
    inicio = time.time()
    mensaje = data.mensaje
    
    try:
        # NIVEL 1: REGEX - Respuestas instantáneas
        respuesta_simple = procesar_mensaje_simple(mensaje)
        
        if respuesta_simple:
            print(f"[NIVEL 1 - REGEX] Respuesta rápida: {respuesta_simple['tipo']}")
            return ChatResponse(
                respuesta=respuesta_simple["respuesta"],
                metodo="regex",
                tipo=respuesta_simple["tipo"],
                confianza=1.0,
                tiempo_procesamiento=time.time() - inicio
            )
        
        # NIVEL 2: ROUTER SEMÁNTICO - Detección de intenciones
        print(f"🎯 [NIVEL 2 - ROUTER] Analizando intención...")
        intencion = router.buscar_intencion(
            mensaje,
            umbral_func=0.60,
            umbral_faq=0.55
        )
        
        # Validación defensiva: Asegurar que score siempre exista
        if 'score' not in intencion:
            intencion['score'] = max(
                intencion.get('score_func', 0.0),
                intencion.get('score_faq', 0.0)
            )
        
        print(f"   └─ Tipo: {intencion['tipo']} | Score: {intencion['score']:.3f}")
        
        # CASO 2A: ACCIÓN DETECTADA → Ejecutar función con BD
        if intencion["tipo"] == "accion" and intencion["score"] > 0.58:
            nombre_func = intencion["funcion"]
            print(f"   └─ 🎯 Función detectada: {nombre_func}")
            
            if nombre_func not in FUNCIONES_DISPONIBLES:
                raise HTTPException(
                    status_code=501,
                    detail=f"La función '{nombre_func}' no está implementada"
                )
            
            try:
                # Extraer parámetros del mensaje
                params_extraidos = extraer_parametros_del_mensaje(mensaje)
                
                # Combinar con params del request
                if data.contexto:
                    params_extraidos.update(data.contexto)
                
                print(f"🎯 Ejecutando función: {nombre_func}")
                print(f"📋 Parámetros: {params_extraidos}")
                
                # EJECUTAR FUNCIÓN REAL CON BD
                resultado_funcion = FUNCIONES_DISPONIBLES[nombre_func](
                    mensaje=mensaje,
                    db=db,
                    params=params_extraidos
                )
                
                print(f"Función ejecutada: {resultado_funcion.get('exito', 'N/A')}")
                
                # Verificar si la función fue exitosa
                if not resultado_funcion.get('exito', True):
                    return ChatResponse(
                        respuesta=resultado_funcion.get('mensaje', resultado_funcion.get('error', 'Error desconocido')),
                        metodo="router_accion_error",
                        tipo="error",
                        resultado_funcion=resultado_funcion,
                        tiempo_procesamiento=time.time() - inicio
                    )
                
                # Generar explicación en lenguaje natural con LLM
                prompt_explicacion = f"""
Eres un asistente de UPS Tuti. El usuario solicitó: "{mensaje}"

Se ejecutó la función: {nombre_func}

Resultado de la función:
{resultado_funcion}

INSTRUCCIONES:
1. Explica de forma clara y profesional qué se hizo
2. Resume los resultados principales (2-3 puntos clave)
3. Si hay alertas o problemas, destácalos
4. Sugiere próximos pasos si es relevante
5. Máximo 5 oraciones
6. Usa emojis sutilmente para claridad visual

Responde en español, tono profesional pero amigable.
                """
                
                respuesta_llm = rag_service.llm.invoke(prompt_explicacion)
                
                return ChatResponse(
                    respuesta=respuesta_llm.content,
                    metodo="router_accion_ejecutada",
                    tipo="accion",
                    confianza=float(intencion["score"]),
                    resultado_funcion=resultado_funcion,
                    metadata={
                        "funcion": nombre_func,
                        "score_router": float(intencion["score"]),
                        "parametros_usados": params_extraidos
                    },
                    tiempo_procesamiento=time.time() - inicio
                )
            
            except Exception as e:
                print(f"Error ejecutando {nombre_func}: {e}")
                
                return ChatResponse(
                    respuesta=f"Ocurrió un error al procesar tu solicitud. " +
                              f"Por favor intenta de nuevo o contacta a soporte: {COMPANY_INFO['contacto']['email_soporte']}",
                    metodo="router_accion_error",
                    tipo="error",
                    metadata={
                        "error": str(e),
                        "funcion": nombre_func
                    },
                    tiempo_procesamiento=time.time() - inicio
                )
        
        # CASO 2B: FAQ DETECTADA → Respuesta directa
        if intencion["tipo"] == "faq" and intencion["score"] > 0.60:
            return ChatResponse(
                respuesta=intencion["respuesta"],
                metodo="router_faq",
                tipo="faq",
                confianza=float(intencion["score"]),
                metadata={
                    "pregunta_original": intencion.get("pregunta_original", "N/A"),
                    "score_router": float(intencion["score"])
                },
                tiempo_procesamiento=time.time() - inicio
            )
        
        # NIVEL 3: RAG - Búsqueda contextual avanzada
        print(f"[NIVEL 3 - RAG] Procesando: {mensaje}")
        
        resultado_rag = rag_service.responder_pregunta_general(mensaje)
        
        print(f"   └─ Tipo búsqueda: {resultado_rag.get('tipo_busqueda', 'N/A')}")
        print(f"   └─ Confianza: {resultado_rag.get('confianza', 'N/A')}")
        print(f"   └─ # Fuentes: {resultado_rag.get('num_fuentes', 0)}")
        print(f"   └─ Longitud respuesta: {len(resultado_rag.get('respuesta', ''))} caracteres")
        
        # Convertir todos los valores numpy a tipos nativos de Python
        resultado_rag = convertir_numpy_a_python(resultado_rag)
        
        return ChatResponse(
            respuesta=resultado_rag["respuesta"],
            metodo="rag",
            tipo=resultado_rag.get("tipo_busqueda", "general"),
            confianza=resultado_rag.get("confianza"),
            fuentes=resultado_rag.get("fuentes"),
            metadata={
                "num_fuentes": resultado_rag.get("num_fuentes", 0),
                "mejor_score": resultado_rag.get("mejor_score"),
                "razon": resultado_rag.get("razon")
            },
            tiempo_procesamiento=time.time() - inicio
        )
    
    except HTTPException as he:
        raise he
    
    except Exception as e:
        print(f"Error general en chat_endpoint: {e}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Error interno del servidor. Contacta a soporte: {COMPANY_INFO['contacto']['email_soporte']}"
        )




@app.get("/rag/faq")
async def responder_faq(pregunta: str = Query(..., description="Pregunta del usuario sobre FAQs")):
    # para responder preguntas frecuentes usando RAG
    
    if not rag_service:
        raise HTTPException(
            status_code=503,
            detail="Servicio RAG no disponible. Verifica configuración de credenciales."
        )
    
    try:
        resultado = rag_service.responder_faq(pregunta)
        return {
            "pregunta": pregunta,
            "respuesta": resultado['respuesta'],
            "fuentes": resultado['fuentes'],
            "confianza": resultado['confianza'],
            "tipo": "faq"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar pregunta FAQ: {str(e)}"
        )


@app.get("/rag/empresa")
async def consultar_empresa(consulta: str = Query(..., description="Consulta sobre información de la empresa")):
    # consultar información general de la empresa usando RAG

    if not rag_service:
        raise HTTPException(
            status_code=503,
            detail="Servicio RAG no disponible. Verifica configuración de credenciales."
        )
    
    try:
        resultado = rag_service.responder_sobre_empresa(consulta)
        return {
            "consulta": consulta,
            "respuesta": resultado['respuesta'],
            "fuentes": resultado['fuentes'],
            "confianza": resultado['confianza'],
            "tipo": "empresa"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar consulta de empresa: {str(e)}"
        )


@app.get("/rag/pregunta")
async def responder_pregunta_general(pregunta: str = Query(..., description="Pregunta general (auto-detecta FAQ o empresa)")):
    """
    analiza la pregunta y decide si buscar en FAQs o en información de la empresa
    como:
        ¿Cómo puedo contactarlos? → FAQ
        ¿Qué productos venden? → Empresa
        ¿Cuál es su horario? → FAQ
        Cuéntame sobre UPS Tuti → Empresa
    """
    if not rag_service:
        raise HTTPException(
            status_code=503,
            detail="Servicio RAG no disponible. Verifica configuración de credenciales."
        )
    
    try:
        resultado = rag_service.responder_pregunta_general(pregunta)
        return {
            "pregunta": pregunta,
            "respuesta": resultado['respuesta'],
            "fuentes": resultado['fuentes'],
            "confianza": resultado['confianza'],
            "tipo_busqueda": resultado['tipo_busqueda']
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar pregunta: {str(e)}"
        )


@app.get("/rag/info")
async def info_rag():
    # info del sistema RAG

    if not rag_service:
        return {
            "status": "no_disponible",
            "mensaje": "Servicio RAG no inicializado"
        }
    
    try:
        from company_info import COMPANY_INFO, FAQS
        
        return {
            "status": "activo",
            "empresa": {
                "nombre": COMPANY_INFO['nombre'],
                "nicho": COMPANY_INFO['nicho'],
                "fundacion": COMPANY_INFO['fundacion'],
                "sede": COMPANY_INFO['sede']
            },
            "knowledge_base": {
                "num_faqs": len(FAQS),
                "num_categorias_productos": len(COMPANY_INFO['productos']),
                "num_servicios": len(COMPANY_INFO['servicios']),
                "num_proveedores": len(COMPANY_INFO['proveedores'])
            },
            "endpoints": {
                "faq": "/rag/faq?pregunta=tu_pregunta",
                "empresa": "/rag/empresa?consulta=tu_consulta",
                "general": "/rag/pregunta?pregunta=tu_pregunta",
                "info": "/rag/info"
            },
            "modelo_embeddings": "text-embedding-005",
            "modelo_llm": "gemini-2.0-flash-exp"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener información RAG: {str(e)}"
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
        
from sqlalchemy import distinct

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

