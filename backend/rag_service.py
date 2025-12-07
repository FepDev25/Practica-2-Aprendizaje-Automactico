import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from company_info import COMPANY_INFO, FAQS
import datetime
env_path = Path(__file__).parent / 'env/.env'
load_dotenv(dotenv_path=env_path)
PROJECT_ID = os.getenv("PROJECT_ID")
CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
LLM_MODEL = "gemini-2.0-flash-exp"

class RAGKnowledgeService:
    # servicio RAG para responder preguntas usando knowledge base
    
    def __init__(self):
        self._validar_configuracion()
        
        # Inicializar LLM y embeddings
        self.llm = self._crear_llm()
        self.embeddings = self._crear_embeddings()
        
        # Crear vector stores
        self.vectorstore_company = None
        self.vectorstore_faqs = None
        
        # Inicializar knowledge base
        self._inicializar_knowledge_base()
        
        print("Servicio RAG inicializado correctamente")
    
    def _validar_configuracion(self):
        if not PROJECT_ID:
            raise ValueError("Error: PROJECT_ID no configurado")
        
        if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
            raise FileNotFoundError("Error: Credenciales de Google Cloud no encontradas")
    
    def _crear_llm(self):
        return ChatVertexAI(
            model=LLM_MODEL,
            project=PROJECT_ID,
            temperature=0.3, 
            max_tokens=400,
        )
    
    def _crear_embeddings(self):
        return VertexAIEmbeddings(
            model_name="text-embedding-005",  
            project=PROJECT_ID
        )
    
    def _crear_documentos_empresa(self) -> list[Document]:
        # convierte la información de la empresa en documentos para el vector store
        documentos = []
        
        # Información general
        info_general = f"""
        Empresa: {COMPANY_INFO['nombre']}
        Nicho: {COMPANY_INFO['nicho']}
        Slogan: {COMPANY_INFO['slogan']}
        
        Fundada en: {COMPANY_INFO['fundacion']}
        Sede: {COMPANY_INFO['sede']}
        
        Misión: {COMPANY_INFO['mision']}
        Visión: {COMPANY_INFO['vision']}
        """
        documentos.append(Document(
            page_content=info_general,
            metadata={"tipo": "info_general", "categoria": "empresa"}
        ))
        
        # Productos por categoría
        for categoria, productos in COMPANY_INFO['productos'].items():
            productos_texto = f"Categoría: {categoria.replace('_', ' ').title()}\n\n"
            for prod in productos:
                productos_texto += f"- {prod['nombre']} (SKU: {prod['sku']}, Categoría: {prod['categoria']})\n"
            
            documentos.append(Document(
                page_content=productos_texto,
                metadata={"tipo": "productos", "categoria": categoria}
            ))
        
        # Servicios
        servicios_texto = "Servicios que ofrece UPS Tuti:\n\n"
        for i, servicio in enumerate(COMPANY_INFO['servicios'], 1):
            servicios_texto += f"{i}. {servicio}\n"
        
        documentos.append(Document(
            page_content=servicios_texto,
            metadata={"tipo": "servicios", "categoria": "empresa"}
        ))
        
        # Horarios
        horarios_texto = f"""
        Horarios de atención:
        
        Atención al cliente: {COMPANY_INFO['horario']['atencion_cliente']}
        Operaciones de almacén: {COMPANY_INFO['horario']['operaciones_almacen']}
        Soporte técnico: {COMPANY_INFO['horario']['soporte_tecnico']}
        """
        documentos.append(Document(
            page_content=horarios_texto,
            metadata={"tipo": "horarios", "categoria": "operacion"}
        ))
        
        # Contacto
        contacto_texto = f"""
        Información de contacto de UPS Tuti:
        
        Email de ventas: {COMPANY_INFO['contacto']['email_ventas']}
        Email de soporte: {COMPANY_INFO['contacto']['email_soporte']}
        Teléfono: {COMPANY_INFO['contacto']['telefono']}
        WhatsApp: {COMPANY_INFO['contacto']['whatsapp']}
        Dirección: {COMPANY_INFO['contacto']['direccion']}
        """
        documentos.append(Document(
            page_content=contacto_texto,
            metadata={"tipo": "contacto", "categoria": "empresa"}
        ))
        
        # Hitos históricos
        hitos_texto = "Historia y logros de UPS Tuti:\n\n"
        for hito in COMPANY_INFO['hitos']:
            hitos_texto += f"• {hito}\n"
        
        documentos.append(Document(
            page_content=hitos_texto,
            metadata={"tipo": "historia", "categoria": "empresa"}
        ))
        
        # Políticas
        politicas_texto = "Políticas de UPS Tuti:\n\n"
        for nombre_politica, descripcion in COMPANY_INFO['politicas'].items():
            politicas_texto += f"{nombre_politica.replace('_', ' ').title()}: {descripcion}\n\n"
        
        documentos.append(Document(
            page_content=politicas_texto,
            metadata={"tipo": "politicas", "categoria": "empresa"}
        ))
        
        # Diferenciadores
        diferenciadores_texto = "Ventajas competitivas de UPS Tuti:\n\n"
        for i, dif in enumerate(COMPANY_INFO['diferenciadores'], 1):
            diferenciadores_texto += f"{i}. {dif}\n"
        
        documentos.append(Document(
            page_content=diferenciadores_texto,
            metadata={"tipo": "diferenciadores", "categoria": "empresa"}
        ))
        
        # Proveedores
        proveedores_texto = "Proveedores de UPS Tuti:\n\n"
        for nombre_prov, info in COMPANY_INFO['proveedores'].items():
            proveedores_texto += f"{nombre_prov} (Prioridad {info['prioridad']}): {', '.join(info['productos'])}\n"
        
        documentos.append(Document(
            page_content=proveedores_texto,
            metadata={"tipo": "proveedores", "categoria": "operacion"}
        ))
        
        return documentos
    
    def _crear_documentos_faqs(self) -> list[Document]:
        # convierte las FAQs en documentos para el vector store
        documentos = []
        
        for i, faq in enumerate(FAQS):
            # Combinar pregunta y respuesta para mejor contexto
            contenido = f"Pregunta: {faq['pregunta']}\n\nRespuesta: {faq['respuesta']}"
            
            documentos.append(Document(
                page_content=contenido,
                metadata={
                    "tipo": "faq",
                    "pregunta": faq['pregunta'],
                    "id": i
                }
            ))
        
        return documentos
    
    def _inicializar_knowledge_base(self):
        # inicializa los vector stores con los documentos de la empresa y FAQs
        try:
            # Crear documentos
            docs_empresa = self._crear_documentos_empresa()
            docs_faqs = self._crear_documentos_faqs()
            
            # Crear vector stores con FAISS
            print("Creando vector store de información de empresa...")
            self.vectorstore_company = FAISS.from_documents(
                docs_empresa,
                self.embeddings
            )
            
            print("Creando vector store de FAQs...")
            self.vectorstore_faqs = FAISS.from_documents(
                docs_faqs,
                self.embeddings
            )
            
            print(f"Knowledge base inicializada: {len(docs_empresa)} docs de empresa, {len(docs_faqs)} FAQs")
            
        except Exception as e:
            print(f"Error al inicializar knowledge base: {e}")
            raise
    
    def responder_faq(self, pregunta_usuario: str, k: int = 2) -> dict:
        # responde a una pregunta buscando en las FAQs usando RAG
      
        try:
            # Buscar FAQs similares
            docs_relevantes = self.vectorstore_faqs.similarity_search(
                pregunta_usuario,
                k=k
            )
            
            if not docs_relevantes:
                return {
                    "message": "Lo siento, no encontré información específica sobre esa pregunta. ¿Podrías reformularla o contactar a nuestro soporte?",
                    "fuentes": [],
                    "confianza": "baja"
                }
            
            # Construir contexto con las FAQs encontradas
            contexto = "\n\n".join([doc.page_content for doc in docs_relevantes])
            
            # Crear prompt para el LLM
            template = """
Eres un asistente de atención al cliente de UPS Tuti, un distribuidor mayorista de snacks saludables.

Responde la pregunta del usuario usando ÚNICAMENTE la información de las siguientes FAQs:

{contexto}

Pregunta del usuario: {pregunta}

INSTRUCCIONES:
1. Si la información está en las FAQs, responde de manera clara y amigable
2. Cita la información relevante de las FAQs
3. Si la pregunta NO está relacionada con las FAQs proporcionadas, di que no tienes esa información específica y sugiere contactar al equipo
4. Mantén un tono profesional pero cercano
5. Sé conciso (máximo 4-5 oraciones)

Respuesta:
            """
            
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.llm | StrOutputParser()
            
            respuesta = chain.invoke({
                "contexto": contexto,
                "pregunta": pregunta_usuario
            })
            
            # Extraer preguntas de las FAQs relevantes
            fuentes = [doc.metadata.get('pregunta', 'N/A') for doc in docs_relevantes]
            
            return {
                "message": respuesta.strip(),
                "fuentes": fuentes,
                "confianza": "alta",
                "num_fuentes": len(docs_relevantes)
            }
            
        except Exception as e:
            print(f"Error en responder_faq: {e}")
            return {
                "message": f"Disculpa, ocurrió un error al procesar tu pregunta. Por favor contacta a soporte: {COMPANY_INFO['contacto']['email_soporte']}",
                "fuentes": [],
                "confianza": "error"
            }
    
    def responder_sobre_empresa(self, consulta_usuario: str, k: int = 3) -> dict:
        #
        # responde preguntas generales sobre la empresa usando RAG
        
        try:
            # Buscar documentos relevantes
            docs_relevantes = self.vectorstore_company.similarity_search(
                consulta_usuario,
                k=k
            )
            
            if not docs_relevantes:
                return {
                    "message": "No encontré información específica sobre eso. ¿Podrías ser más específico?",
                    "fuentes": [],
                    "confianza": "baja"
                }
            
            # Construir contexto
            contexto = "\n\n---\n\n".join([doc.page_content for doc in docs_relevantes])
            
            # Crear prompt
            template = """
Eres un asistente experto de UPS Tuti, un distribuidor mayorista de snacks saludables en Ecuador.

Usa la siguiente información de la empresa para responder la consulta del usuario:

{contexto}

Consulta del usuario: {consulta}

INSTRUCCIONES:
1. Responde usando ÚNICAMENTE la información proporcionada
2. Sé preciso y específico
3. Si la información no está en el contexto, dilo claramente
4. Mantén un tono profesional y amigable
5. Incluye detalles relevantes (fechas, números, nombres) cuando estén disponibles

Respuesta:
            """
            
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.llm | StrOutputParser()
            
            respuesta = chain.invoke({
                "contexto": contexto,
                "consulta": consulta_usuario
            })
            
            # Extraer tipos de documentos usados
            tipos_docs = [doc.metadata.get('tipo', 'desconocido') for doc in docs_relevantes]
            
            return {
                "message": respuesta.strip(),
                "fuentes": tipos_docs,
                "confianza": "alta",
                "num_documentos": len(docs_relevantes)
            }
            
        except Exception as e:
            print(f"Error en responder_sobre_empresa: {e}")
            return {
                "message": f"Ocurrió un error al procesar tu consulta. Contacta a nuestro equipo: {COMPANY_INFO['contacto']['email_soporte']}",
                "fuentes": [],
                "confianza": "error"
            }
    
    def responder_pregunta_general(self, pregunta: str) -> dict:
        #
        # Método principal que decide si buscar en FAQs o información general de empresa
        #
        # Args:
        #     pregunta: Pregunta del usuario
        #     
        # Returns:
        #     dict con respuesta y metadata
        #
        # Palabras clave que sugieren búsqueda en FAQs
        keywords_faq = ['cómo', 'cuándo', 'dónde', 'horario', 'contacto', 'precio', 
                        'envío', 'devolución', 'pago', 'cliente']
        
        pregunta_lower = pregunta.lower()
        es_faq = any(keyword in pregunta_lower for keyword in keywords_faq)
        
        if es_faq:
            resultado = self.responder_faq(pregunta)
            resultado['tipo_busqueda'] = 'faq'
        else:
            resultado = self.responder_sobre_empresa(pregunta)
            resultado['tipo_busqueda'] = 'empresa'
        
        return resultado
    def generar_respuesta_conversacional(self, tipo: str, mensaje_usuario: str) -> dict:
        """
        Genera respuestas naturales y contextualizadas para saludos y despedidas
        usando el LLM con información de la empresa
        
        Args:
            tipo: 'saludo' o 'despedida'
            mensaje_usuario: Mensaje original del usuario
        
        Returns:
            dict con respuesta generada y metadata
        """
        try:
            # Obtener hora actual para contextualizar
            hora_actual = datetime.datetime.now().hour
            
            if hora_actual < 12:
                momento_dia = "mañana"
                emoji_momento = "🌅"
            elif hora_actual < 18:
                momento_dia = "tarde"
                emoji_momento = "☀️"
            else:
                momento_dia = "noche"
                emoji_momento = "🌙"
            
            # Templates según el tipo de interacción
            if tipo == "saludo":
                template = """
    Eres el asistente virtual de {nombre_empresa}, {descripcion_empresa}.
    
    El usuario te ha saludado diciendo: "{mensaje_usuario}"
    
    CONTEXTO DE LA EMPRESA:
    - Nombre: {nombre_empresa}
    - Especialidad: {nicho}
    - Slogan: "{slogan}"
    - Servicios principales: {servicios_breve}
    - Horario de atención: {horario_atencion}
    
    MOMENTO DEL DÍA: {momento_dia} (usar emoji: {emoji_momento})
    
    INSTRUCCIONES:
    1. Responde el saludo de manera cálida y profesional adaptada al momento del día
    2. Preséntate brevemente como el asistente de {nombre_empresa}
    3. Menciona de forma natural 2-3 cosas que puedes hacer:
       - Consultar información sobre productos y servicios
       - Analizar el estado del inventario y hacer predicciones de stock
       - Proporcionar información de contacto, horarios y políticas
       - Responder preguntas frecuentes
    4. Incluye el emoji apropiado del momento del día al inicio
    5. Máximo 4 oraciones
    6. Tono: Profesional, cercano y entusiasta
    
    EJEMPLO DE ESTRUCTURA:
    "{emoji_momento} ¡[Saludo apropiado]! Soy [nombre], el asistente virtual de {nombre_empresa}. Puedo ayudarte a [acción 1], [acción 2] y [acción 3]. ¿En qué te puedo ayudar hoy?"
    
    Respuesta:
                """
                
                servicios_breve = ", ".join(COMPANY_INFO['servicios'][:3])
                
                prompt = ChatPromptTemplate.from_template(template)
                chain = prompt | self.llm | StrOutputParser()
                
                respuesta = chain.invoke({
                    "nombre_empresa": COMPANY_INFO['nombre'],
                    "descripcion_empresa": COMPANY_INFO['nicho'],
                    "mensaje_usuario": mensaje_usuario,
                    "nicho": COMPANY_INFO['nicho'],
                    "slogan": COMPANY_INFO['slogan'],
                    "servicios_breve": servicios_breve,
                    "horario_atencion": COMPANY_INFO['horario']['atencion_cliente'],
                    "momento_dia": momento_dia,
                    "emoji_momento": emoji_momento
                })
                
            elif tipo == "despedida":
                template = """
    Eres el asistente virtual de {nombre_empresa}, {descripcion_empresa}.
    
    El usuario se está despidiendo diciendo: "{mensaje_usuario}"
    
    CONTEXTO DE LA EMPRESA:
    - Nombre: {nombre_empresa}
    - Email de soporte: {email_soporte}
    - WhatsApp: {whatsapp}
    - Horario de atención: {horario_atencion}
    
    INSTRUCCIONES:
    1. Responde la despedida de manera cordial y profesional
    2. Agradece genuinamente por usar el servicio
    3. Menciona disponibilidad futura de forma breve
    4. Ofrece UN medio de contacto alternativo (email O WhatsApp, no ambos)
    5. Incluye emoji de despedida (👋, 😊, o ✨)
    6. Máximo 3 oraciones
    7. Tono: Cálido, profesional y positivo
    
    EJEMPLO DE ESTRUCTURA:
    "👋 [Despedida apropiada]. Gracias por contactar a {nombre_empresa}. Si necesitas más ayuda, estamos disponibles en [contacto] durante nuestro horario de atención."
    
    Respuesta:
                """
                
                prompt = ChatPromptTemplate.from_template(template)
                chain = prompt | self.llm | StrOutputParser()
                
                respuesta = chain.invoke({
                    "nombre_empresa": COMPANY_INFO['nombre'],
                    "descripcion_empresa": COMPANY_INFO['nicho'],
                    "mensaje_usuario": mensaje_usuario,
                    "email_soporte": COMPANY_INFO['contacto']['email_soporte'],
                    "whatsapp": COMPANY_INFO['contacto']['whatsapp'],
                    "horario_atencion": COMPANY_INFO['horario']['atencion_cliente']
                })
            
            else:
                # Tipo no reconocido, usar respuesta genérica con emoji
                return {
                    "message": f"👋 ¡Hola! Soy el asistente virtual de {COMPANY_INFO['nombre']}. ¿En qué puedo ayudarte hoy?",
                    "confianza": "baja",
                    "tipo_respuesta": "fallback",
                    "momento_dia": momento_dia
                }
            
            return {
                "message": respuesta.strip(),
                "confianza": "alta",
                "tipo_respuesta": tipo,
                "momento_dia": momento_dia,
                "emoji_usado": emoji_momento if tipo == "saludo" else "👋"
            }
            
        except Exception as e:
            print(f"Error en generar_respuesta_conversacional: {e}")
            import traceback
            traceback.print_exc()
            
            # Determinar momento del día para fallback
            hora_actual = datetime.datetime.now().hour
            if hora_actual < 12:
                momento_dia = "mañana"
                emoji_momento = "🌅"
            elif hora_actual < 18:
                momento_dia = "tarde"
                emoji_momento = "☀️"
            else:
                momento_dia = "noche"
                emoji_momento = "🌙"
            
            # Fallback según el tipo
            if tipo == "saludo":
                respuesta_fallback = (
                    f"{emoji_momento} ¡Hola! Soy el asistente virtual de {COMPANY_INFO['nombre']}, "
                    f"tu aliado en la distribución de snacks saludables. "
                    f"Puedo ayudarte con consultas sobre productos, analizar el inventario, "
                    f"o responder tus preguntas sobre nuestros servicios. ¿En qué te puedo ayudar?"
                )
            else:  # despedida
                respuesta_fallback = (
                    f"👋 ¡Gracias por contactar a {COMPANY_INFO['nombre']}! "
                    f"Si necesitas más ayuda, escríbenos a {COMPANY_INFO['contacto']['whatsapp']} "
                    f"o {COMPANY_INFO['contacto']['email_soporte']}. ¡Que tengas un excelente día!"
                )
            
            return {
                "message": respuesta_fallback,
                "confianza": "media",
                "tipo_respuesta": f"{tipo}_fallback",
                "momento_dia": momento_dia,
                "error": str(e)
            }
_rag_service = None

def get_rag_service() -> RAGKnowledgeService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGKnowledgeService()
    return _rag_service

from typing import List, Dict
import numpy as np

class UnifiedSemanticRouter:
    """Router que usa los mismos embeddings del RAG"""
    
    def __init__(self, embeddings, functions_data: List[Dict], intents_data: List[Dict]):
        self.embeddings = embeddings
        self.functions_data = functions_data
        self.intents_data = intents_data  # Cambio: ahora son "intents" no "faqs"
        
        # Pre-calcular embeddings
        self.func_texts = [f"{f['id']} {f['docstring']}" for f in functions_data]
        self.intent_texts = [f["text"] for f in intents_data]
        
        print("Vectorizando funciones e intenciones con Google...")
        self.func_embeddings = np.array(self.embeddings.embed_documents(self.func_texts))
        self.intent_embeddings = np.array(self.embeddings.embed_documents(self.intent_texts))
    
    def _similitud_coseno(self, vec1, vec2):
        dot = np.dot(vec1, vec2)
        norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return float(dot / norm) if norm > 0 else 0.0
    
    def buscar_intencion(self, query: str, umbral: float = 0.50):
        query_emb = np.array(self.embeddings.embed_query(query))
        
        # Comparar con funciones
        func_scores = [self._similitud_coseno(query_emb, emb) for emb in self.func_embeddings]
        best_func_idx = int(np.argmax(func_scores))
        best_func_score = float(func_scores[best_func_idx])
        
        # Comparar con intenciones (saludos, despedidas, etc)
        intent_scores = [self._similitud_coseno(query_emb, emb) for emb in self.intent_embeddings]
        best_intent_idx = int(np.argmax(intent_scores))
        best_intent_score = float(intent_scores[best_intent_idx])
        
        # Decisión: ¿es una acción del sistema o una consulta general?
        if best_func_score < umbral and best_intent_score < umbral:
            return {"tipo": "rag", "score": max(best_func_score, best_intent_score)}
        
        if best_func_score >= best_intent_score:
            funcion_id = self.functions_data[best_func_idx]["id"]
            
            # Si es saludo o despedida, marcarlo como "conversacional"
            if funcion_id in ["saludo", "despedida"]:
                return {
                    "tipo": "conversacional",
                    "subtipo": funcion_id,
                    "score": best_func_score
                }
            
            # Si es una acción real del sistema
            return {
                "tipo": "accion",
                "funcion": funcion_id,
                "score": best_func_score
            }
        else:
            # Si coincide con una intención específica, usar RAG
            intent_category = self.intents_data[best_intent_idx].get("categoria", "general")
            return {
                "tipo": "rag",
                "categoria": intent_category,
                "score": best_intent_score
            }


def crear_router_integrado(rag_service) -> UnifiedSemanticRouter:
    """Crea router usando embeddings del RAG."""
    
    # Funciones/acciones del sistema
    FUNCTIONS = [
        {
            "id": "predecir_all_stock",
            "docstring": (
                "predecir inventario calcular stock futuro y dias restantes "
                "analizar consumo estimar agotamiento productos"
            )
        },
        {
            "id": "enviar_correo",
            "docstring": (
                "enviar mandar correo email notificación redactar mensaje electronico "
                "avisar cliente o administrador"
            )
        },
        {
            "id": "productos_criticos",
            "docstring": (
                "buscar productos con stock critico alerta agotamiento "
                "listado de productos urgentes"
            )
        },
        {
            "id": "saludo",
            "docstring": (
                "saludar hola buenos dias buenas tardes en que puedo ayudarte "
                "mensaje de bienvenida iniciar conversacion"
            )
        },
        {
            "id": "despedida",
            "docstring": (
                "despedir adios hasta luego gracias por usar servicio "
                "cerrar conversacion chao nos vemos"
            )
        }
    ]
    
    # Intenciones para mejorar el routing (sin respuestas fijas)
    INTENTS = [
        {"text": "horario atención trabajo", "categoria": "horarios"},
        {"text": "contacto teléfono whatsapp", "categoria": "contacto"},
        {"text": "correo soporte ventas", "categoria": "contacto"},
        {"text": "tiempo costo envío", "categoria": "envios"},
        {"text": "métodos pago facturación", "categoria": "pagos"},
        {"text": "productos disponibles categorías", "categoria": "productos"},
        {"text": "política devoluciones garantía", "categoria": "politicas"},
        {"text": "quienes son qué hacen empresa", "categoria": "empresa"}
    ]
    
    return UnifiedSemanticRouter(
        rag_service.embeddings,
        FUNCTIONS,
        INTENTS
    )

