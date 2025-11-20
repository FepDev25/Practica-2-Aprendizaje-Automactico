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
        nivel = "bajo" if prediccion < 20 else "normal" if prediccion < 100 else "alto"
        return (
            f"Predicción para {nombre} en fecha {fecha}: {prediccion:.2f} unidades. "
            f"Nivel de stock: {nivel}."
        )
    
    def _mensaje_fallback_multiple(self, total: int, fecha: str) -> str:
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
