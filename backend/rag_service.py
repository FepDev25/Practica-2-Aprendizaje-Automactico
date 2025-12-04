import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
import numpy as np
from functools import lru_cache

from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from company_info import COMPANY_INFO, FAQS

# Configuración
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)
PROJECT_ID = os.getenv("PROJECT_ID")
CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
LLM_MODEL = "gemini-2.0-flash-exp"

# ============================================================================
# CLASE RAG MEJORADA
# ============================================================================

class RAGKnowledgeService:
    """Servicio RAG para responder preguntas usando knowledge base optimizada"""
    
    def __init__(self):
        self._validar_configuracion()
        
        # Inicializar LLM y embeddings
        self.llm = self._crear_llm()
        self.embeddings = self._crear_embeddings()
        
        # Vector stores
        self.vectorstore_company = None
        self.vectorstore_faqs = None
        
        # Cache para consultas repetidas
        self._cache_consultas = {}
        
        # Inicializar knowledge base
        self._inicializar_knowledge_base()
        
        print("✅ Servicio RAG inicializado correctamente")
    
    def _validar_configuracion(self):
        if not PROJECT_ID:
            raise ValueError("❌ Error: PROJECT_ID no configurado")
        if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
            raise FileNotFoundError("❌ Error: Credenciales de Google Cloud no encontradas")
    
    def _crear_llm(self):
        return ChatVertexAI(
            model=LLM_MODEL,
            project=PROJECT_ID,
            temperature=0.3,
            max_tokens=500,  # Aumentado para respuestas más completas
        )
    
    def _crear_embeddings(self):
        return VertexAIEmbeddings(
            model_name="text-embedding-005",
            project=PROJECT_ID
        )
    
    def _crear_documentos_empresa(self) -> List[Document]:
        """Crea documentos optimizados con mejor estructura"""
        documentos = []
        
        # 1. Info General Estructurada
        info_general = f"""
        INFORMACIÓN CORPORATIVA - UPS TUTI
        
        Nombre: {COMPANY_INFO['nombre']}
        Nicho de Mercado: {COMPANY_INFO['nicho']}
        Lema Corporativo: {COMPANY_INFO['slogan']}
        
        Fundación: {COMPANY_INFO['fundacion']}
        Ubicación: {COMPANY_INFO['sede']}
        
        Misión: {COMPANY_INFO['mision']}
        Visión: {COMPANY_INFO['vision']}
        
        Valores: Calidad, Innovación, Sostenibilidad, Servicio al Cliente
        """
        documentos.append(Document(
            page_content=info_general,
            metadata={"tipo": "info_general", "categoria": "empresa", "prioridad": "alta"}
        ))
        
        # 2. Catálogo de Productos (mejorado)
        for categoria, productos in COMPANY_INFO['productos'].items():
            categoria_limpia = categoria.replace('_', ' ').title()
            
            # Documento resumen de categoría
            resumen = f"CATEGORÍA: {categoria_limpia}\n\n"
            resumen += f"Total de productos: {len(productos)}\n\n"
            
            # Documentos individuales por producto (mejor para búsquedas específicas)
            for prod in productos:
                prod_detalle = f"""
                PRODUCTO: {prod['nombre']}
                SKU: {prod['sku']}
                Categoría: {prod['categoria']}
                Línea: {categoria_limpia}
                
                Descripción: Producto premium de UPS Tuti disponible para distribución mayorista.
                """
                
                documentos.append(Document(
                    page_content=prod_detalle,
                    metadata={
                        "tipo": "producto",
                        "categoria": categoria,
                        "sku": prod['sku'],
                        "nombre": prod['nombre'],
                        "prioridad": "alta"
                    }
                ))
        
        # 3. Servicios
        servicios_texto = "SERVICIOS DISPONIBLES - UPS TUTI\n\n"
        for i, servicio in enumerate(COMPANY_INFO['servicios'], 1):
            servicios_texto += f"{i}. {servicio}\n"
        servicios_texto += "\n\nTodos nuestros servicios están diseñados para mayoristas y distribuidores."
        
        documentos.append(Document(
            page_content=servicios_texto,
            metadata={"tipo": "servicios", "categoria": "empresa", "prioridad": "media"}
        ))
        
        # 4. Horarios (estructurado)
        horarios_texto = f"""
        HORARIOS DE OPERACIÓN - UPS TUTI
        
        🕐 Atención al Cliente: {COMPANY_INFO['horario']['atencion_cliente']}
        📦 Operaciones de Almacén: {COMPANY_INFO['horario']['operaciones_almacen']}
        🔧 Soporte Técnico: {COMPANY_INFO['horario']['soporte_tecnico']}
        
        Nota: Para urgencias fuera de horario, contactar vía WhatsApp.
        """
        documentos.append(Document(
            page_content=horarios_texto,
            metadata={"tipo": "horarios", "categoria": "operacion", "prioridad": "alta"}
        ))
        
        # 5. Contacto (expandido)
        contacto_texto = f"""
        INFORMACIÓN DE CONTACTO - UPS TUTI
        
        📧 Ventas: {COMPANY_INFO['contacto']['email_ventas']}
        🛠️ Soporte: {COMPANY_INFO['contacto']['email_soporte']}
        📞 Teléfono: {COMPANY_INFO['contacto']['telefono']}
        💬 WhatsApp: {COMPANY_INFO['contacto']['whatsapp']}
        📍 Dirección Física: {COMPANY_INFO['contacto']['direccion']}
        
        Preferencia de contacto: WhatsApp para consultas rápidas, Email para pedidos formales.
        """
        documentos.append(Document(
            page_content=contacto_texto,
            metadata={"tipo": "contacto", "categoria": "empresa", "prioridad": "alta"}
        ))
        
        # 6. Historia
        hitos_texto = "HISTORIA Y LOGROS - UPS TUTI\n\n"
        for hito in COMPANY_INFO['hitos']:
            hitos_texto += f"✓ {hito}\n"
        
        documentos.append(Document(
            page_content=hitos_texto,
            metadata={"tipo": "historia", "categoria": "empresa", "prioridad": "baja"}
        ))
        
        # 7. Políticas
        politicas_texto = "POLÍTICAS EMPRESARIALES - UPS TUTI\n\n"
        for nombre_politica, descripcion in COMPANY_INFO['politicas'].items():
            politicas_texto += f"• {nombre_politica.replace('_', ' ').title()}:\n  {descripcion}\n\n"
        
        documentos.append(Document(
            page_content=politicas_texto,
            metadata={"tipo": "politicas", "categoria": "empresa", "prioridad": "media"}
        ))
        
        # 8. Ventajas Competitivas
        diferenciadores_texto = "VENTAJAS COMPETITIVAS - UPS TUTI\n\n"
        for i, dif in enumerate(COMPANY_INFO['diferenciadores'], 1):
            diferenciadores_texto += f"{i}. {dif}\n"
        
        documentos.append(Document(
            page_content=diferenciadores_texto,
            metadata={"tipo": "diferenciadores", "categoria": "empresa", "prioridad": "media"}
        ))
        
        # 9. Red de Proveedores
        proveedores_texto = "RED DE PROVEEDORES - UPS TUTI\n\n"
        for nombre_prov, info in COMPANY_INFO['proveedores'].items():
            proveedores_texto += f"• {nombre_prov} (Prioridad: {info['prioridad']})\n"
            proveedores_texto += f"  Productos: {', '.join(info['productos'])}\n\n"
        
        documentos.append(Document(
            page_content=proveedores_texto,
            metadata={"tipo": "proveedores", "categoria": "operacion", "prioridad": "baja"}
        ))
        
        return documentos
    
    def _crear_documentos_faqs(self) -> List[Document]:
        """Crea documentos de FAQs con mejor contexto"""
        documentos = []
        
        for i, faq in enumerate(FAQS):
            # Contexto enriquecido
            contenido = f"""
            PREGUNTA FRECUENTE #{i+1}
            
            Pregunta: {faq['pregunta']}
            
            Respuesta: {faq['respuesta']}
            
            Categoría: FAQ | Atención al Cliente
            """
            
            documentos.append(Document(
                page_content=contenido,
                metadata={
                    "tipo": "faq",
                    "pregunta": faq['pregunta'],
                    "id": i,
                    "prioridad": "alta"
                }
            ))
        
        return documentos
    
    def _inicializar_knowledge_base(self):
        """Inicializa vectorstores con manejo de errores"""
        try:
            docs_empresa = self._crear_documentos_empresa()
            docs_faqs = self._crear_documentos_faqs()
            
            print(f"📄 Creando {len(docs_empresa)} documentos de empresa...")
            self.vectorstore_company = FAISS.from_documents(
                docs_empresa,
                self.embeddings
            )
            
            print(f"❓ Creando {len(docs_faqs)} documentos de FAQs...")
            self.vectorstore_faqs = FAISS.from_documents(
                docs_faqs,
                self.embeddings
            )
            
            print(f"✅ Knowledge base lista: {len(docs_empresa)} docs empresa, {len(docs_faqs)} FAQs")
            
        except Exception as e:
            print(f"❌ Error al inicializar knowledge base: {e}")
            raise
    
    @lru_cache(maxsize=100)
    def _busqueda_con_cache(self, consulta: str, tipo: str, k: int) -> Tuple:
        """Búsqueda con cache para consultas repetidas"""
        vectorstore = self.vectorstore_faqs if tipo == "faq" else self.vectorstore_company
        docs = vectorstore.similarity_search_with_score(consulta, k=k)
        # Convertir a tupla para hacer hashable
        return tuple((doc.page_content, doc.metadata, score) for doc, score in docs)
    
    def responder_faq(self, pregunta_usuario: str, k: int = 3) -> dict:
        """Responde FAQs con búsqueda mejorada"""
        try:
            # Búsqueda con scores
            resultados = self._busqueda_con_cache(pregunta_usuario, "faq", k)
            
            if not resultados or resultados[0][2] < 0.5:  # Score mínimo
                return {
                    "respuesta": "No encontré una FAQ específica para tu consulta. ¿Podrías reformularla? También puedes contactar a nuestro equipo de soporte.",
                    "fuentes": [],
                    "confianza": 0.3,
                    "sugerencia": f"Escribe a: {COMPANY_INFO['contacto']['email_soporte']}"
                }
            
            # Construir contexto con scores
            contexto_items = []
            fuentes = []
            for content, metadata, score in resultados[:k]:
                if score > 0.5:  # Filtrar por relevancia
                    contexto_items.append(f"[Relevancia: {score:.2f}]\n{content}")
                    fuentes.append({
                        "pregunta": metadata.get('pregunta', 'N/A'),
                        "relevancia": round(score, 2)
                    })
            
            contexto = "\n\n---\n\n".join(contexto_items)
            
            # Prompt mejorado para FAQs más útiles
            template = """
Eres un asistente experto de UPS Tuti, distribuidor mayorista de snacks saludables en Ecuador.

PREGUNTAS FRECUENTES RELACIONADAS:
{contexto}

PREGUNTA DEL USUARIO: {pregunta}

INSTRUCCIONES IMPORTANTES:
1. Proporciona una respuesta COMPLETA Y ÚTIL basada en las FAQs
2. Incluye TODOS los datos relevantes: horarios exactos, montos, procesos paso a paso
3. Si hay varios puntos importantes, enuméralos claramente
4. Si la pregunta tiene múltiples aspectos, abordalos todos
5. Sé específico con nombres, números, contactos y URLs cuando aplique
6. Tono cercano y servicial, como un asesor comercial experto
7. Entre 3-6 oraciones dependiendo de la complejidad
8. Si mencionas contacto, incluye email y teléfono

RESPUESTA:
            """
            
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.llm | StrOutputParser()
            
            respuesta = chain.invoke({
                "contexto": contexto,
                "pregunta": pregunta_usuario
            })
            
            return {
                "respuesta": respuesta.strip(),
                "fuentes": fuentes,
                "confianza": 0.9 if resultados[0][2] > 0.7 else 0.7,
                "num_fuentes": len(fuentes),
                "mejor_score": round(resultados[0][2], 2)
            }
            
        except Exception as e:
            print(f"❌ Error en responder_faq: {e}")
            return {
                "respuesta": f"Disculpa, ocurrió un error técnico. Contacta a soporte: {COMPANY_INFO['contacto']['email_soporte']}",
                "fuentes": [],
                "confianza": "error"
            }
    
    def responder_sobre_empresa(self, consulta_usuario: str, k: int = 4) -> dict:
        """Responde sobre la empresa con búsqueda optimizada"""
        try:
            resultados = self._busqueda_con_cache(consulta_usuario, "empresa", k)
            
            if not resultados or resultados[0][2] < 0.4:
                return {
                    "respuesta": "No encontré información específica sobre esa consulta en nuestra base de conocimiento. ¿Podrías ser más específico o reformular tu pregunta?",
                    "fuentes": [],
                    "confianza": 0.3
                }
            
            # Construir contexto enriquecido
            contexto_items = []
            fuentes = []
            
            for content, metadata, score in resultados:
                if score > 0.4:
                    tipo_doc = metadata.get('tipo', 'general')
                    contexto_items.append(f"[Tipo: {tipo_doc} | Relevancia: {score:.2f}]\n{content}")
                    fuentes.append({
                        "tipo": tipo_doc,
                        "categoria": metadata.get('categoria', 'N/A'),
                        "relevancia": round(score, 2)
                    })
            
            contexto = "\n\n---\n\n".join(contexto_items)
            
            # Prompt mejorado para respuestas más completas
            template = """
Eres un asistente experto de UPS Tuti, distribuidor mayorista de snacks saludables en Cuenca, Ecuador.

INFORMACIÓN CORPORATIVA DISPONIBLE:
{contexto}

PREGUNTA DEL USUARIO: {consulta}

INSTRUCCIONES IMPORTANTES:
1. Proporciona una respuesta COMPLETA y DETALLADA usando toda la información relevante del contexto
2. Si preguntan "qué hacemos" o "a qué nos dedicamos", explica:
   - Nuestra actividad principal (distribución mayorista de snacks saludables)
   - Nuestros productos destacados
   - Nuestra propuesta de valor (tecnología IA para inventario)
3. Si mencionas productos, incluye ejemplos específicos con SKUs
4. Estructura la respuesta de forma clara con bullet points si hay múltiples elementos
5. Incluye datos concretos (fechas, ubicación, números) cuando estén disponibles
6. Tono profesional pero cálido y acogedor
7. Respuesta entre 4-8 oraciones según la complejidad de la pregunta

RESPUESTA:
            """
            
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.llm | StrOutputParser()
            
            respuesta = chain.invoke({
                "contexto": contexto,
                "consulta": consulta_usuario
            })
            
            return {
                "respuesta": respuesta.strip(),
                "fuentes": fuentes,
                "confianza": 0.9 if resultados[0][2] > 0.7 else 0.7,
                "num_fuentes": len(fuentes),
                "mejor_score": round(resultados[0][2], 2)
            }
            
        except Exception as e:
            print(f"❌ Error en responder_sobre_empresa: {e}")
            return {
                "respuesta": f"Error al procesar tu consulta. Contacta: {COMPANY_INFO['contacto']['email_soporte']}",
                "fuentes": [],
                "confianza": "error"
            }
    
    def responder_pregunta_general(self, pregunta: str) -> dict:
        """Router inteligente que decide dónde buscar"""
        
        # Análisis de keywords mejorado
        keywords_faq = {
            'preguntas': ['cómo', 'cuándo', 'dónde', 'por qué', 'quién', 'cuál'],
            'operaciones': ['horario', 'contacto', 'teléfono', 'email', 'dirección'],
            'comercial': ['precio', 'costo', 'pagar', 'comprar', 'pedido'],
            'logística': ['envío', 'entrega', 'devolución', 'garantía'],
            'soporte': ['problema', 'ayuda', 'soporte', 'reclamo']
        }
        
        keywords_empresa = {
            'corporativo': ['misión', 'visión', 'historia', 'fundación', 'dedica', 'hace', 'empresa', 'ups tuti', 'nosotros'],
            'productos': ['producto', 'sku', 'catálogo', 'snack', 'vende', 'venden', 'tienen', 'galleta', 'chip', 'barra'],
            'servicios': ['servicio', 'ofrece', 'provee', 'facilita', 'brinda'],
            'partners': ['proveedor', 'socio', 'alianza'],
            'ubicacion': ['dónde', 'donde', 'ubicación', 'dirección', 'ciudad', 'sede']
        }
        
        pregunta_lower = pregunta.lower()
        
        # Contar matches
        score_faq = sum(
            1 for categoria in keywords_faq.values()
            for keyword in categoria
            if keyword in pregunta_lower
        )
        
        score_empresa = sum(
            1 for categoria in keywords_empresa.values()
            for keyword in categoria
            if keyword in pregunta_lower
        )
        
        # Decisión inteligente
        if score_faq > score_empresa:
            resultado = self.responder_faq(pregunta, k=3)
            resultado['tipo_busqueda'] = 'faq'
            resultado['razon'] = 'Consulta operativa/FAQ detectada'
        elif score_empresa > score_faq:
            resultado = self.responder_sobre_empresa(pregunta, k=4)
            resultado['tipo_busqueda'] = 'empresa'
            resultado['razon'] = 'Consulta corporativa/productos detectada'
        else:
            # Búsqueda híbrida
            resultado_faq = self.responder_faq(pregunta, k=2)
            resultado_empresa = self.responder_sobre_empresa(pregunta, k=2)
            
            # Seleccionar mejor resultado
            if resultado_faq.get('mejor_score', 0) > resultado_empresa.get('mejor_score', 0):
                resultado = resultado_faq
                resultado['tipo_busqueda'] = 'faq'
            else:
                resultado = resultado_empresa
                resultado['tipo_busqueda'] = 'empresa'
            
            resultado['razon'] = 'Búsqueda híbrida (ambigüedad detectada)'
        
        return resultado


# ============================================================================
# ROUTER SEMÁNTICO MEJORADO
# ============================================================================

class UnifiedSemanticRouter:
    """Router optimizado con mejores umbrales y caching"""
    
    def __init__(self, embeddings, functions_data: List[Dict], faqs_data: List[Dict]):
        self.embeddings = embeddings
        self.functions_data = functions_data
        self.faqs_data = faqs_data
        
        # Pre-calcular embeddings
        self.func_texts = [
            f"{f['id']} {f['docstring']} {' '.join(f.get('keywords', []))}"
            for f in functions_data
        ]
        self.faq_texts = [f["text"] for f in faqs_data]
        
        print("🔄 Vectorizando funciones y FAQs...")
        self.func_embeddings = np.array(self.embeddings.embed_documents(self.func_texts))
        self.faq_embeddings = np.array(self.embeddings.embed_documents(self.faq_texts))
        print(f"✅ {len(self.func_embeddings)} funciones y {len(self.faq_embeddings)} FAQs vectorizadas")
        
        # Cache
        self._cache = {}
    
    @staticmethod
    def _similitud_coseno(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calcula similitud coseno optimizada"""
        dot = np.dot(vec1, vec2)
        norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return float(dot / norm) if norm > 0 else 0.0
    
    def buscar_intencion(
        self, 
        query: str, 
        umbral_func: float = 0.55,
        umbral_faq: float = 0.50
    ) -> Dict:
        """Busca intención con umbrales diferenciados"""
        
        # Check cache
        if query in self._cache:
            return self._cache[query]
        
        # Embeddings de la consulta
        query_emb = np.array(self.embeddings.embed_query(query))
        
        # Calcular similitudes
        func_scores = [
            self._similitud_coseno(query_emb, emb) 
            for emb in self.func_embeddings
        ]
        best_func_idx = int(np.argmax(func_scores))
        best_func_score = float(func_scores[best_func_idx])
        
        faq_scores = [
            self._similitud_coseno(query_emb, emb) 
            for emb in self.faq_embeddings
        ]
        best_faq_idx = int(np.argmax(faq_scores))
        best_faq_score = float(faq_scores[best_faq_idx])
        
        # Decisión con umbrales diferenciados
        resultado = None
        
        if best_func_score < umbral_func and best_faq_score < umbral_faq:
            resultado = {
                "tipo": "desconocido",
                "score": max(best_func_score, best_faq_score),  # FIX: Agregar score
                "score_func": best_func_score,
                "score_faq": best_faq_score,
                "mensaje": "No se detectó intención clara"
            }
        
        elif best_func_score >= umbral_func and best_func_score >= best_faq_score:
            resultado = {
                "tipo": "accion",
                "funcion": self.functions_data[best_func_idx]["id"],
                "score": best_func_score,
                "metadata": self.functions_data[best_func_idx]
            }
        
        else:
            resultado = {
                "tipo": "faq",
                "respuesta": self.faqs_data[best_faq_idx]["answer"],
                "score": best_faq_score,
                "pregunta_original": self.faqs_data[best_faq_idx]["text"]
            }
        
        # Guardar en cache
        self._cache[query] = resultado
        return resultado


# ============================================================================
# FUNCIONES Y CONFIGURACIÓN
# ============================================================================

def crear_router_integrado(rag_service: RAGKnowledgeService) -> UnifiedSemanticRouter:
    """Factory para crear router con configuración mejorada"""
    
    FUNCTIONS = [
        {
            "id": "predecir_stock",
            "docstring": "predecir inventario stock futuro días restantes agotamiento",
            "keywords": ["predecir", "stock", "inventario", "quedan", "días"]
        },
        {
            "id": "generar_alerta",
            "docstring": "crear alerta notificación bajo stock crítico",
            "keywords": ["alerta", "notificar", "avisar", "bajo"]
        },
        {
            "id": "buscar_producto",
            "docstring": "buscar producto SKU código encontrar consultar",
            "keywords": ["buscar", "producto", "sku", "código", "encontrar"]
        },
        {
            "id": "enviar_correo",
            "docstring": "enviar mandar correo email redactar mensaje electrónico notificar",
            "keywords": ["correo", "email", "enviar", "mensaje", "notificar"]
        }
    ]
    
    FAQS = [
        {
            "text": "horario atención servicio clientes abierto cerrado",
            "answer": "Atención al cliente: Lunes a Viernes 8AM-6PM"
        },
        {
            "text": "contacto teléfono whatsapp número llamar",
            "answer": f"WhatsApp: {COMPANY_INFO['contacto']['whatsapp']}"
        },
        {
            "text": "dirección ubicación dónde están oficina",
            "answer": f"Dirección: {COMPANY_INFO['contacto']['direccion']}"
        },
        {
            "text": "email correo ventas contactar escribir",
            "answer": f"Ventas: {COMPANY_INFO['contacto']['email_ventas']}"
        }
    ]
    
    return UnifiedSemanticRouter(rag_service.embeddings, FUNCTIONS, FAQS)


# ============================================================================
# SINGLETON PATTERN
# ============================================================================

_rag_service = None

def get_rag_service() -> RAGKnowledgeService:
    """Obtiene instancia singleton del servicio RAG"""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGKnowledgeService()
    return _rag_service