from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SemanticRouter:
    def __init__(self, functions_data, faqs_data, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        """
        Inicializa el modelo y pre-calcula los embeddings.
        """
        print("--- Iniciando Motor Semántico ---")
        self.model = SentenceTransformer(model_name)
        self.functions_data = functions_data
        self.faqs_data = faqs_data
        
        # Pre-calcular embeddings para optimizar velocidad
        # Unimos Nombre + Docstring para mayor contexto
        self.func_texts = [f"{f['id']} {f['docstring']}" for f in functions_data]
        self.faq_texts = [f["text"] for f in faqs_data]
        
        print("Vectorizando datos base...")
        self.func_embeddings = self.model.encode(self.func_texts)
        self.faq_embeddings = self.model.encode(self.faq_texts)
        print("--- Motor Listo ---")

    def buscar_intencion(self, query: str, umbral: float = 0.45):
        """
        Recibe texto, lo convierte a vector y busca la mejor coincidencia.
        """
        # 1. Vectorizar la consulta del usuario
        query_embedding = self.model.encode([query])

        # 2. Comparar con FUNCIONES
        func_scores = cosine_similarity(query_embedding, self.func_embeddings)[0]
        best_func_idx = np.argmax(func_scores)
        best_func_score = float(func_scores[best_func_idx]) # Convertir a float nativo

        # 3. Comparar con FAQs
        faq_scores = cosine_similarity(query_embedding, self.faq_embeddings)[0]
        best_faq_idx = np.argmax(faq_scores)
        best_faq_score = float(faq_scores[best_faq_idx])

        # 4. Lógica de Decisión
        
        # Caso A: Nadie supera el umbral (No entendí)
        if best_func_score < umbral and best_faq_score < umbral:
            return {
                "tipo": "desconocido",
                "mensaje": "No estoy seguro de cómo ayudarte con eso.",
                "score": max(best_func_score, best_faq_score)
            }

        # Caso B: Gana una Función
        if best_func_score >= best_faq_score:
            matched_func = self.functions_data[best_func_idx]
            return {
                "tipo": "accion",
                "funcion": matched_func["id"],
                "score": best_func_score,
                "meta": matched_func # Retornamos info extra si se necesita
            }
        
        # Caso C: Gana una FAQ
        else:
            matched_faq = self.faqs_data[best_faq_idx]
            return {
                "tipo": "faq",
                "respuesta": matched_faq["answer"],
                "score": best_faq_score
            }