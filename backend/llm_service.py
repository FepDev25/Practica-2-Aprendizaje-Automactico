"""
Servicio LLM para generar mensajes amigables de predicción usando Google Gemini.

Este módulo integra LangChain con Google Vertex AI para transformar
predicciones numéricas en mensajes explicativos y amigables para el usuario.
"""

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
    """
    Servicio para generar mensajes amigables de predicción usando LLM.
    """
    
    def __init__(self):
        """Inicializa el servicio LLM con Gemini."""
        self._validar_configuracion()
        self.llm = self._crear_llm()
        self.chain = self._crear_chain()
        print(f"✓ Servicio LLM inicializado con modelo '{LLM_MODEL}'")
    
    def _validar_configuracion(self):
        """Valida que las variables de entorno estén configuradas."""
        if not PROJECT_ID or not CREDENTIALS_PATH:
            raise ValueError(
                "Error: Variables de entorno no configuradas. "
                "Verifica PROJECT_ID y GOOGLE_APPLICATION_CREDENTIALS en .env"
            )
        
        if not Path(CREDENTIALS_PATH).exists():
            raise FileNotFoundError(
                f"Error: No se encontró el archivo de credenciales en: {CREDENTIALS_PATH}"
            )
    
    def _crear_llm(self):
        """Crea y configura el modelo LLM de Gemini."""
        return ChatVertexAI(
            model=LLM_MODEL,
            project=PROJECT_ID,
            temperature=0.7,  # Balance entre creatividad y precisión
            max_tokens=300,   # Limitar longitud de respuesta
        )
    
    def _crear_chain(self):
        """Crea la cadena LangChain para generar mensajes."""
        template = """
Eres un asistente experto en gestión de inventarios de supermercado.

Tu tarea es explicar una predicción de stock de manera clara y profesional.

INFORMACIÓN DE LA PREDICCIÓN:
- Producto: {nombre_producto}
- SKU: {sku}
- Fecha de predicción: {fecha}
- Stock predicho: {prediccion} unidades
{contexto_adicional}

INSTRUCCIONES:
1. Genera un mensaje amigable y profesional (máximo 3-4 oraciones)
2. Interpreta la predicción:
   - Si es < 20 unidades: advierte sobre stock bajo
   - Si está entre 20-100: indica nivel normal
   - Si es > 100: menciona stock abundante
3. Da una recomendación breve y accionable
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
        contexto_adicional: str = ""
    ) -> str:
        """
        Genera un mensaje amigable explicando la predicción.
        
        Args:
            nombre_producto: Nombre del producto
            sku: SKU del producto
            fecha: Fecha de la predicción
            prediccion: Valor numérico predicho
            contexto_adicional: Información adicional opcional
            
        Returns:
            str: Mensaje explicativo generado por el LLM
        """
        try:
            # Formatear predicción con 2 decimales
            prediccion_formateada = f"{prediccion:.2f}"
            
            # Invocar la cadena
            mensaje = self.chain.invoke({
                "nombre_producto": nombre_producto,
                "sku": sku,
                "fecha": fecha,
                "prediccion": prediccion_formateada,
                "contexto_adicional": contexto_adicional or ""
            })
            
            return mensaje.strip()
        
        except Exception as e:
            # Fallback en caso de error del LLM
            return self._mensaje_fallback(nombre_producto, prediccion, fecha)
    
    def generar_mensaje_multiple(
        self,
        fecha: str,
        total_productos: int,
        predicciones_destacadas: list
    ) -> str:
        """
        Genera un mensaje resumen para predicciones múltiples.
        
        Args:
            fecha: Fecha de las predicciones
            total_productos: Número total de productos analizados
            predicciones_destacadas: Lista con predicciones destacadas
            
        Returns:
            str: Mensaje resumen generado por el LLM
        """
        template = """
Eres un analista de inventarios de supermercado.

ANÁLISIS REALIZADO:
- Fecha de predicción: {fecha}
- Total de productos analizados: {total_productos}
- Productos destacados:
{productos_destacados}

INSTRUCCIONES:
1. Genera un resumen ejecutivo breve (3-4 oraciones)
2. Menciona el estado general del inventario
3. Destaca productos con stock crítico (< 20 unidades)
4. Da una recomendación general

Respuesta:
"""
        
        try:
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.llm | StrOutputParser()
            
            # Formatear productos destacados
            productos_str = "\n".join([
                f"  • {p['nombre']}: {p['prediccion']:.2f} unidades"
                for p in predicciones_destacadas[:5]  # Top 5
            ])
            
            mensaje = chain.invoke({
                "fecha": fecha,
                "total_productos": total_productos,
                "productos_destacados": productos_str
            })
            
            return mensaje.strip()
        
        except Exception as e:
            return self._mensaje_fallback_multiple(total_productos, fecha)
    
    def _mensaje_fallback(self, nombre: str, prediccion: float, fecha: str) -> str:
        """Mensaje de respaldo si el LLM falla."""
        nivel = "bajo" if prediccion < 20 else "normal" if prediccion < 100 else "alto"
        return (
            f"Predicción para {nombre} en fecha {fecha}: {prediccion:.2f} unidades. "
            f"Nivel de stock: {nivel}."
        )
    
    def _mensaje_fallback_multiple(self, total: int, fecha: str) -> str:
        """Mensaje de respaldo para predicciones múltiples."""
        return (
            f"Se completó el análisis de {total} productos para la fecha {fecha}. "
            f"Revise los resultados detallados para más información."
        )


# Instancia global del servicio (singleton)
_llm_service = None


def get_llm_service() -> LLMPrediccionService:
    """
    Obtiene la instancia singleton del servicio LLM.
    Se crea solo la primera vez que se llama.
    """
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMPrediccionService()
    return _llm_service
