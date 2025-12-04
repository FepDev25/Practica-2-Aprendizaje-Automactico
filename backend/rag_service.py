import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from company_info import COMPANY_INFO, FAQS

env_path = Path(__file__).parent / '.env'
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
                    "respuesta": "Lo siento, no encontré información específica sobre esa pregunta. ¿Podrías reformularla o contactar a nuestro soporte?",
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
                "respuesta": respuesta.strip(),
                "fuentes": fuentes,
                "confianza": "alta",
                "num_fuentes": len(docs_relevantes)
            }
            
        except Exception as e:
            print(f"Error en responder_faq: {e}")
            return {
                "respuesta": f"Disculpa, ocurrió un error al procesar tu pregunta. Por favor contacta a soporte: {COMPANY_INFO['contacto']['email_soporte']}",
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
                    "respuesta": "No encontré información específica sobre eso. ¿Podrías ser más específico?",
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
                "respuesta": respuesta.strip(),
                "fuentes": tipos_docs,
                "confianza": "alta",
                "num_documentos": len(docs_relevantes)
            }
            
        except Exception as e:
            print(f"Error en responder_sobre_empresa: {e}")
            return {
                "respuesta": f"Ocurrió un error al procesar tu consulta. Contacta a nuestro equipo: {COMPANY_INFO['contacto']['email_soporte']}",
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

_rag_service = None

def get_rag_service() -> RAGKnowledgeService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGKnowledgeService()
    return _rag_service

