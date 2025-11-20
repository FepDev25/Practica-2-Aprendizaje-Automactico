import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Cargar variables de entorno
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# Configuración
PROJECT_ID = os.getenv("PROJECT_ID")
CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
LLM_MODEL = "gemini-2.0-flash-exp"  # Modelo más rápido y económico


class LLMPrediccionService:
    def __init__(self):
        self._validar_configuracion()
        self.llm = self._crear_llm()
        self.chain = self._crear_chain()
        print(f"✓ Servicio LLM inicializado con modelo '{LLM_MODEL}'")
    
    def _validar_configuracion(self):
        if not PROJECT_ID:
            raise ValueError(
                "Error: PROJECT_ID no configurado. Verifica .env o la variable de entorno PROJECT_ID"
            )

        creds = CREDENTIALS_PATH
        if creds:
            creds_path = Path(creds).expanduser()
            if not creds_path.is_absolute():
                creds_path = (Path(__file__).parent / creds_path).resolve()
            if creds_path.exists():
                # Asegurar que Google client lo vea
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path)
                return

        env_dir = Path(__file__).parent / "env"
        if env_dir.exists() and env_dir.is_dir():
            json_files = list(env_dir.glob("*.json"))
            if json_files:
                sel = json_files[0]
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(sel)
                print(f"Usando credenciales encontradas en: {sel}")
                return

        raise FileNotFoundError(
            f"Error: No se encontró el archivo de credenciales. Buscado: {CREDENTIALS_PATH} y en {env_dir}"
        )
    
    def _crear_llm(self):
        return ChatVertexAI(
            model=LLM_MODEL,
            project=PROJECT_ID,
            temperature=0.7,  
            max_tokens=300, 
        )
    
    def _crear_chain(self):
        template = """
Eres un asistente experto en gestión de inventarios de supermercado.

Tu tarea es explicar una predicción de stock de manera clara y profesional.

INFORMACIÓN DE LA PREDICCIÓN:
- Producto: {nombre_producto}
- SKU: {sku}
- Fecha de predicción: {fecha}
- Stock predicho: {prediccion} unidades
- Nivel mínimo de stock: {minimum_stock_level} unidades
{contexto_adicional}

INSTRUCCIONES:
1. Genera un mensaje amigable y profesional (máximo 3-4 oraciones)
2. Interpreta la predicción comparando con el nivel mínimo de stock:
   - Si el stock predicho está por debajo del nivel mínimo: advierte sobre stock crítico
   - Si está cerca del nivel mínimo (entre mínimo y 1.5x mínimo): indica precaución
   - Si está por encima de 1.5x el nivel mínimo: indica nivel adecuado o abundante
3. Da una recomendación breve y accionable basada en esta comparación
4. Usa un tono profesional pero cercano

Respuesta:
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Crear cadena usando LCEL
        chain = prompt | self.llm | StrOutputParser()
        
        return chain
    
    def generar_mensaje_prediccion(
        self, 
        nombre_producto: str,
        sku: str,
        fecha: str,
        prediccion: float,
        minimum_stock_level: float = 20.0,
        contexto_adicional: str = ""
    ) -> str:
        try:
            # Formatear predicción con 2 decimales
            prediccion_formateada = f"{prediccion:.2f}"
            
            # Invocar la cadena
            mensaje = self.chain.invoke({
                "nombre_producto": nombre_producto,
                "sku": sku,
                "fecha": fecha,
                "prediccion": prediccion_formateada,
                "minimum_stock_level": minimum_stock_level,
                "contexto_adicional": contexto_adicional or ""
            })
            
            return mensaje.strip()
        
        except Exception as e:
            # Fallback en caso de error del LLM
            return self._mensaje_fallback(nombre_producto, prediccion, fecha, minimum_stock_level)
    
    def generar_mensaje_multiple(
        self,
        fecha: str,
        total_productos: int,
        predicciones_destacadas: list,
        stock_critico: list,
        stock_adecuado: list,
        estadisticas: dict
    ) -> str:

        template = """
Eres un analista experto de inventarios de supermercado. Genera un análisis profesional y completo.

DATOS DEL ANÁLISIS:
- Fecha de predicción: {fecha}
- Total de productos analizados: {total_productos}

ESTADÍSTICAS GENERALES:
- Stock promedio predicho: {promedio:.2f} unidades
- Stock mínimo encontrado: {minimo:.2f} unidades (producto: {producto_minimo})
- Stock máximo encontrado: {maximo:.2f} unidades (producto: {producto_maximo})
- Productos con stock crítico: {total_critico}
- Productos con stock adecuado: {total_adecuado}

PRODUCTOS CON STOCK CRÍTICO (requieren atención inmediata):
{productos_criticos}

PRODUCTOS CON MEJOR NIVEL DE STOCK:
{productos_adecuados}

PRODUCTOS EN ORDEN DE CRITICIDAD:
{productos_destacados}

INSTRUCCIONES:
1. Genera un análisis ejecutivo profesional (4-6 oraciones)
2. Interpreta las estadísticas y proporciona contexto
3. Identifica claramente productos en riesgo de desabastecimiento
4. Menciona productos con buen nivel de stock
5. Da recomendaciones específicas y priorizadas para gestión de inventario
6. Usa un tono profesional pero claro

Análisis:
"""
        
        try:
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.llm | StrOutputParser()
            
            # Formatear productos destacados
            productos_str = "\n".join([
                f"  • {p['nombre']} (SKU: {p['sku']}): {p['prediccion']:.2f} unidades"
                for p in predicciones_destacadas[:10]
            ])
            
            # Formatear productos críticos
            criticos_str = "\n".join([
                f"  • {p['nombre']} (SKU: {p['sku']}): {p['prediccion']:.2f} unidades ⚠️"
                for p in stock_critico[:5]
            ]) if stock_critico else "  • Ninguno"
            
            # Formatear productos con buen stock
            adecuados_str = "\n".join([
                f"  • {p['nombre']} (SKU: {p['sku']}): {p['prediccion']:.2f} unidades ✓"
                for p in stock_adecuado[:5]
            ]) if stock_adecuado else "  • Ninguno"
            
            mensaje = chain.invoke({
                "fecha": fecha,
                "total_productos": total_productos,
                "promedio": estadisticas['promedio'],
                "minimo": estadisticas['minimo'],
                "maximo": estadisticas['maximo'],
                "producto_minimo": estadisticas['producto_minimo'],
                "producto_maximo": estadisticas['producto_maximo'],
                "total_critico": len(stock_critico),
                "total_adecuado": len(stock_adecuado),
                "productos_destacados": productos_str,
                "productos_criticos": criticos_str,
                "productos_adecuados": adecuados_str
            })
            
            return mensaje.strip()
        
        except Exception as e:
            print(f"Error en generar_mensaje_multiple: {e}")
            return self._mensaje_fallback_multiple(total_productos, fecha, predicciones_destacadas)
    
    def _mensaje_fallback(self, nombre: str, prediccion: float, fecha: str, minimum_stock_level: float = 20.0) -> str:
        if prediccion < minimum_stock_level:
            nivel = "crítico (por debajo del mínimo)"
        elif prediccion < minimum_stock_level * 1.5:
            nivel = "bajo (cerca del mínimo)"
        else:
            nivel = "adecuado"
        return (
            f"Predicción para {nombre} en fecha {fecha}: {prediccion:.2f} unidades. "
            f"Nivel de stock: {nivel} (mínimo requerido: {minimum_stock_level:.0f} unidades)."
        )
    
    def _mensaje_fallback_multiple(self, total: int, fecha: str, predicciones: list = None) -> str:
        if predicciones and len(predicciones) > 0:
            # Calcular estadísticas básicas
            valores = [p['prediccion'] for p in predicciones]
            promedio = sum(valores) / len(valores)
            criticos = [p for p in predicciones if p['prediccion'] < 50]
            
            msg = f"Análisis de inventario para {fecha}:\n"
            msg += f"• Total de productos: {total}\n"
            msg += f"• Stock promedio predicho: {promedio:.2f} unidades\n"
            msg += f"• Productos con stock crítico (<50 unidades): {len(criticos)}\n"
            
            if criticos:
                msg += f"\nProductos que requieren atención:\n"
                for p in criticos[:3]:
                    msg += f"  - {p['nombre']}: {p['prediccion']:.2f} unidades\n"
            
            return msg
        
        return (
            f"Se completó el análisis de {total} productos para la fecha {fecha}. "
            f"Revise los resultados detallados para más información."
        )


_llm_service = None


def get_llm_service() -> LLMPrediccionService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMPrediccionService()
    return _llm_service
