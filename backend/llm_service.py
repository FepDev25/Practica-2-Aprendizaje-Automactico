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
LLM_MODEL = "gemini-2.0-flash-exp" 


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
    
    def _crear_llm_conclusion(self):
        """LLM específico para generar conclusiones más largas"""
        return ChatVertexAI(
            model=LLM_MODEL,
            project=PROJECT_ID,
            temperature=0.8,  # Más creativo para conclusiones
            max_tokens=500,   # Más tokens para conclusión detallada
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
    
    def _generar_conclusion_inteligente(
        self,
        total_productos: int,
        num_criticos: int,
        num_adecuados: int,
        promedio: float,
        minimo: float,
        maximo: float,
        producto_minimo: str,
        producto_maximo: str,
        productos_criticos: list
    ) -> str:
        """
        Genera una conclusión personalizada usando el LLM basándose en los datos del análisis
        """
        try:
            # Crear prompt para conclusión
            nombres_criticos = [p.get('nombre', 'Sin nombre') for p in productos_criticos[:3]]
            criticos_str = ", ".join(nombres_criticos) if nombres_criticos else "ninguno"
            
            porcentaje_criticos = (num_criticos / total_productos * 100) if total_productos > 0 else 0
            
            template = """
Eres un analista experto en gestión de inventarios de supermercado.

Basándote en el siguiente análisis de inventario, genera una conclusión profesional y recomendaciones accionables en formato de lista con viñetas (usando guiones -).

DATOS DEL ANÁLISIS:
- Total de productos analizados: {total_productos}
- Productos en estado CRÍTICO: {num_criticos} ({porcentaje_criticos:.1f}%)
- Productos con stock ADECUADO: {num_adecuados}
- Stock promedio predicho: {promedio:.2f} unidades
- Stock mínimo: {minimo:.2f} unidades (Producto: {producto_minimo})
- Stock máximo: {maximo:.2f} unidades (Producto: {producto_maximo})
- Productos más críticos: {criticos_str}

INSTRUCCIONES:
1. Evalúa la situación general del inventario (¿es crítica, moderada, estable?)
2. Identifica las prioridades inmediatas basándote en el porcentaje de productos críticos
3. Da 4-5 recomendaciones ESPECÍFICAS y accionables en formato de lista con guiones (-)
4. Menciona los productos más críticos por nombre si es relevante
5. Sugiere acciones concretas (no genéricas) para cada tipo de situación
6. Mantén un tono profesional pero directo

Genera SOLO la lista de recomendaciones (sin títulos adicionales):
            """
            
            prompt = ChatPromptTemplate.from_template(template)
            llm_conclusion = self._crear_llm_conclusion()
            chain = prompt | llm_conclusion | StrOutputParser()
            
            conclusion = chain.invoke({
                "total_productos": total_productos,
                "num_criticos": num_criticos,
                "num_adecuados": num_adecuados,
                "porcentaje_criticos": porcentaje_criticos,
                "promedio": promedio,
                "minimo": minimo,
                "maximo": maximo,
                "producto_minimo": producto_minimo,
                "producto_maximo": producto_maximo,
                "criticos_str": criticos_str
            })
            
            return conclusion.strip()
            
        except Exception as e:
            print(f"Error generando conclusión con LLM: {e}")
            # Fallback a conclusión estática
            return self._conclusion_fallback(num_criticos, num_adecuados, total_productos)
    
    def _conclusion_fallback(self, num_criticos: int, num_adecuados: int, total: int) -> str:
        """Conclusión de respaldo si falla el LLM"""
        porcentaje = (num_criticos / total * 100) if total > 0 else 0
        
        if porcentaje > 30:
            urgencia = "ALTA PRIORIDAD"
        elif porcentaje > 15:
            urgencia = "PRIORIDAD MODERADA"
        else:
            urgencia = "SEGUIMIENTO ESTÁNDAR"
        
        return f"""- **Nivel de urgencia:** {urgencia} ({num_criticos} productos críticos de {total})
- Priorizar reposición inmediata para productos marcados como CRÍTICOS
- Implementar monitoreo diario para productos en ADVERTENCIA
- Revisar tiempos de entrega con proveedores de productos críticos
- Mantener políticas de stock de seguridad para productos de alta rotación"""

    def generar_mensaje_multiple(
        self,
        fecha: str,
        total_productos: int,
        predicciones_destacadas: list,
        stock_critico: list,
        stock_adecuado: list,
        estadisticas: dict
    ) -> str:
        """
        Genera un informe profesional y estructurado para análisis de inventario múltiple.
        Formato determinístico para visualización consistente en el frontend.
        """
        try:
            # Extraer estadísticas
            promedio = float(estadisticas.get('promedio', 0.0))
            minimo = float(estadisticas.get('minimo', 0.0))
            maximo = float(estadisticas.get('maximo', 0.0))
            producto_minimo = estadisticas.get('producto_minimo', 'N/A')
            producto_maximo = estadisticas.get('producto_maximo', 'N/A')

            # Función auxiliar para clasificar productos
            def estado_producto(pred, ref_min):
                try:
                    p = float(pred)
                except Exception:
                    return 'Desconocido', 'Revisar dato'
                if p < ref_min:
                    return 'CRÍTICO', 'Reponer con prioridad. Revisar lead time.'
                if p < ref_min * 1.5:
                    return 'ADVERTENCIA', 'Monitorear ventas y considerar reposición.'
                return 'ADECUADO', 'Sin acción inmediata requerida.'

            # Construir informe estructurado
            lines = []
            lines.append(f"**Análisis Ejecutivo de Inventario - Fecha: {fecha}**")
            lines.append("")
            lines.append(f"Resumen: Este informe analiza {total_productos} productos y entrega prioridades de acción por producto, seguido de conclusiones operativas.")
            lines.append("")
            
            # Estadísticas generales
            lines.append("**Estadísticas Generales**")
            lines.append(f"- Stock promedio predicho: {promedio:,.2f} unidades")
            lines.append(f"- Stock mínimo: {minimo:,.2f} unidades (Producto: {producto_minimo})")
            lines.append(f"- Stock máximo: {maximo:,.2f} unidades (Producto: {producto_maximo})")
            lines.append(f"- Productos en riesgo (CRÍTICO): {len(stock_critico)}")
            lines.append(f"- Productos con stock adecuado: {len(stock_adecuado)}")
            lines.append("")

            # Detalle por producto
            lines.append("**Detalle por Producto (priorizado)**")
            detalle_lista = predicciones_destacadas or []
            if not detalle_lista:
                lines.append("No hay productos detallados disponibles.")
            else:
                for p in detalle_lista[:50]:  # Limitar a 50 productos
                    nombre = p.get('nombre', 'Sin nombre')
                    sku = p.get('sku', 'N/A')
                    pred = p.get('prediccion', 0.0)
                    estado, recomendacion = estado_producto(pred, minimo if minimo > 0 else 1.0)
                    lines.append(f"- {nombre} (SKU: {sku}) — Predicción: {float(pred):,.2f} unidades — Estado: {estado}")
                    lines.append(f"  Recomendación: {recomendacion}")

            lines.append("")
            # Productos críticos
            lines.append("**Productos Críticos (revisión inmediata)**")
            if stock_critico:
                for p in stock_critico[:20]:
                    lines.append(f"- {p.get('nombre','Sin nombre')} (SKU: {p.get('sku','N/A')}) — {float(p.get('prediccion',0.0)):,.2f} unidades")
            else:
                lines.append("- Ninguno")

            lines.append("")
            # Productos con buen stock
            lines.append("**Productos con Buen Nivel de Stock**")
            if stock_adecuado:
                for p in stock_adecuado[:20]:
                    lines.append(f"- {p.get('nombre','Sin nombre')} (SKU: {p.get('sku','N/A')}) — {float(p.get('prediccion',0.0)):,.2f} unidades")
            else:
                lines.append("- Ninguno")

            lines.append("")
            # Conclusiones y recomendaciones generadas por LLM
            lines.append("**Conclusión y Recomendaciones**")
            
            # Generar conclusión personalizada con LLM
            conclusion_llm = self._generar_conclusion_inteligente(
                total_productos=total_productos,
                num_criticos=len(stock_critico),
                num_adecuados=len(stock_adecuado),
                promedio=promedio,
                minimo=minimo,
                maximo=maximo,
                producto_minimo=producto_minimo,
                producto_maximo=producto_maximo,
                productos_criticos=stock_critico[:5]  # Top 5 críticos
            )
            
            lines.append(conclusion_llm)

            mensaje = "\n".join(lines)
            print(f"Mensaje generado (longitud: {len(mensaje)} caracteres)")
            return mensaje

        except Exception as e:
            print(f"Error en generar_mensaje_multiple (formateo local): {e}")
            import traceback
            traceback.print_exc()
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
