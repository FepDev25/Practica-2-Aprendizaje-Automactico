import smtplib
from email.mime.text import MIMEText
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import numpy as np

from model.registro_advanced import (
    preparar_input_desde_dataset_procesado,
    all_registers_priductos,
    procesar_dataset_inventario,
    buscar_producto_por_id,
    buscar_producto_por_nombre,
    buscar_nombre_por_sku,
    obtener_minimum_stock_level
)
from llm_service import get_llm_service
from model.database import SessionLocal
from model.modeloKeras import ModeloStockKeras
from dias_stock_service import DiasStockService
from model.registro import Registro
from email_service import EmailService

# Inicialización de servicios
dias_stock_service = DiasStockService()
modelo = ModeloStockKeras()
email_service = EmailService()


class APIResponse:
    """Clase para estandarizar respuestas de la API"""
    
    @staticmethod
    def success(message: str, data: Optional[Dict[str, Any]] = None, title: str = "Operación exitosa") -> Dict[str, Any]:
        """Respuesta exitosa estándar"""
        response = {
            "status": "success",
            "title": title,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "data": data or {}
        }
        return response
    
    @staticmethod
    def error(message: str, error_detail: Optional[str] = None, title: str = "Error") -> Dict[str, Any]:
        """Respuesta de error estándar"""
        response = {
            "status": "error",
            "title": title,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "data": {}
        }
        if error_detail:
            response["error_detail"] = error_detail
        return response
    
    @staticmethod
    def warning(message: str, data: Optional[Dict[str, Any]] = None, title: str = "Advertencia") -> Dict[str, Any]:
        """Respuesta de advertencia estándar"""
        response = {
            "status": "warning",
            "title": title,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "data": data or {}
        }
        return response
    
    @staticmethod
    def critical(message: str, data: Optional[Dict[str, Any]] = None, title: str = "Alerta Crítica") -> Dict[str, Any]:
        """Respuesta crítica estándar"""
        response = {
            "status": "critical",
            "title": title,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "data": data or {}
        }
        return response


def predecir_all_stock(data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Ejecuta predicción de inventario para todos los productos
    
    Args:
        data: dict opcional con 'fecha' (formato 'YYYY-MM-DD')
    
    Returns:
        dict con respuesta estandarizada de la API
    """
    try:
        # Obtener fecha
        fecha = data.get('fecha') if data else None
        if not fecha:
            fecha = datetime.now().strftime('%Y-%m-%d')
        
        # Obtener todos los productos
        productos = all_registers_priductos()
        resultados = []
        
        # Procesar cada producto
        for prod in productos:
            features = preparar_input_desde_dataset_procesado(sku=prod, fecha_override=fecha)
            
            if features is not None and np.any(features):
                pred = modelo.predecir(features)
                nombre = buscar_nombre_por_sku(prod)
                minimum_stock = obtener_minimum_stock_level(prod) or 20.0
                
                resultados.append({
                    "sku": prod,
                    "nombre": nombre,
                    "prediccion": float(pred),
                    "minimum_stock": minimum_stock
                })
        
        if not resultados:
            return APIResponse.warning(
                message="No se encontraron productos con datos suficientes para realizar predicciones.",
                title="⚠️ Sin datos de predicción",
                data={
                    "fecha_prediccion": fecha,
                    "total_productos": 0,
                    "predictions": []
                }
            )
        
        # Generar mensaje resumen con LLM
        mensaje_resumen = None
        llm_service = get_llm_service()
        
        if llm_service and resultados:
            try:
                resultados_ordenados = sorted(resultados, key=lambda x: x['prediccion'])
                
                stock_critico = [r for r in resultados if r['prediccion'] < r['minimum_stock']]
                stock_precaucion = [r for r in resultados 
                                   if r['minimum_stock'] <= r['prediccion'] < r['minimum_stock'] * 1.5]
                stock_adecuado = [r for r in resultados if r['prediccion'] >= r['minimum_stock'] * 1.5]
                
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
                    predicciones_destacadas=resultados_ordenados[:10],
                    stock_critico=stock_critico,
                    stock_adecuado=stock_adecuado,
                    estadisticas=estadisticas
                )
            except Exception as llm_error:
                print(f"Error generando mensaje resumen: {llm_error}")
                import traceback
                traceback.print_exc()
        
        # Mensaje fallback
        if not mensaje_resumen:
            num_criticos = len([r for r in resultados if r['prediccion'] < r['minimum_stock']])
            mensaje_resumen = (
                f"Se analizaron {len(resultados)} productos para la fecha {fecha}. "
                f"{num_criticos} productos requieren atención urgente por stock crítico."
            )
        
        return APIResponse.success(
            message=mensaje_resumen,
            title="📦 Predicción de inventario",
            data={
                "fecha_prediccion": fecha,
                "total_productos": len(resultados),
                "productos_criticos": len([r for r in resultados if r['prediccion'] < r['minimum_stock']]),
                "productos_analizados": len(resultados)
            }
        )
        
    except Exception as e:
        print(f"Error en predecir_all_stock: {e}")
        import traceback
        traceback.print_exc()
        return APIResponse.error(
            message="No se pudo completar el análisis de predicción de inventario.",
            error_detail=str(e),
            title="❌ Error en predicción"
        )


def productos_criticos(data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Identifica productos con stock crítico consultando la base de datos
    y genera un análisis con LLM
    
    Args:
        data: dict opcional (no se usa actualmente)
    
    Returns:
        dict con respuesta estandarizada de la API
    """
    try:
        db = SessionLocal()
        
        try:
            # Consultar registros de la base de datos
            registros = (
                db.query(
                    Registro.product_name,
                    Registro.average_daily_usage,
                    Registro.quantity_on_hand,
                    Registro.created_at
                )
                .distinct(Registro.product_name)
                .order_by(Registro.product_name, Registro.created_at.desc())
                .all()
            )

            if not registros:
                return APIResponse.warning(
                    message="No hay registros en la base de datos para analizar el estado del inventario.",
                    title="⚠️ Sin datos disponibles"
                )

            # Construir lote de productos
            productos_lote = [
                {
                    "nombre": r.product_name,
                    "stock": r.quantity_on_hand,
                    "ventas_diarias": r.average_daily_usage
                }
                for r in registros
            ]
            
            # Analizar con el servicio de días de stock
            resultado = dias_stock_service.analizar_lote_productos(productos_lote)
            
            # Generar mensaje con LLM
            llm_service = get_llm_service()
            mensaje_resumen = None
            
            if llm_service:
                try:
                    resultados_detallados = resultado.get('resultados_detallados', [])
                    criticos = [r for r in resultados_detallados if "CRÍTICO" in r.get('estado', '')]
                    alertas = [r for r in resultados_detallados if "ALERTA" in r.get('estado', '')]
                    
                    if criticos:
                        print(f"DEBUG - Ejemplo de producto crítico: {criticos[0]}")
                    
                    mensaje_resumen = llm_service.generar_mensaje_productos_criticos(
                        total_productos=resultado['total_productos'],
                        productos_criticos=criticos,
                        productos_alerta=alertas,
                        productos_ok=resultado['productos_ok']
                    )
                    
                except Exception as llm_error:
                    print(f"Error generando mensaje con LLM: {llm_error}")
                    import traceback
                    traceback.print_exc()
            
            # Mensaje fallback
            if not mensaje_resumen:
                mensaje_resumen = resultado.get('resumen', 'Análisis de inventario completado.')
            
            # Determinar respuesta según criticidad
            productos_criticos_count = resultado['productos_criticos']
            productos_alerta_count = resultado['productos_alerta']
            
            response_data = {
                "total_productos": resultado['total_productos'],
                "productos_criticos": productos_criticos_count,
                "productos_alerta": productos_alerta_count,
                "productos_ok": resultado['productos_ok']
            }
            
            if productos_criticos_count > 0:
                return APIResponse.critical(
                    message=mensaje_resumen,
                    title="🚨 Alerta: Productos críticos detectados",
                    data=response_data
                )
            elif productos_alerta_count > 0:
                return APIResponse.warning(
                    message=mensaje_resumen,
                    title="⚠️ Atención: Productos requieren reabastecimiento",
                    data=response_data
                )
            else:
                return APIResponse.success(
                    message=mensaje_resumen,
                    title="✅ Inventario bajo control",
                    data=response_data
                )
            
        finally:
            db.close()
        
    except Exception as e:
        print(f"Error en productos_criticos: {e}")
        import traceback
        traceback.print_exc()
        return APIResponse.error(
            message="No se pudo completar el análisis de productos críticos.",
            error_detail=str(e),
            title="❌ Error al analizar productos"
        )


def enviar_correo(data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Envía notificación por correo electrónico con análisis de inventario
    
    Args:
        data: dict opcional con:
            - 'tipo': 'prediccion' o 'criticos' (default: 'criticos')
            - 'destinatario': email destino (default: 'venotacu@gmail.com')
            - 'incluir_analisis': bool para incluir análisis LLM (default: True)
    
    Returns:
        dict con respuesta estandarizada de la API
    """
    try:
        # Parámetros
        tipo_reporte = data.get('tipo', 'criticos') if data else 'criticos'
        destinatario = data.get('destinatario', 'venotacu@gmail.com') if data else 'venotacu@gmail.com'
        incluir_analisis = data.get('incluir_analisis', True) if data else True
        
        # Obtener análisis de productos críticos
        resultado = productos_criticos()
        if resultado.get('status') == 'error':
            return resultado
        
        resumen = resultado.get('message', 'Análisis de productos críticos')
        asunto_tipo = "Alerta de Stock Crítico"
        
        # Obtener datos de la base de datos
        db = SessionLocal()
        try:
            registros = (
                db.query(
                    Registro.product_name,
                    Registro.average_daily_usage,
                    Registro.quantity_on_hand,
                    Registro.created_at
                )
                .distinct(Registro.product_name)
                .order_by(Registro.product_name, Registro.created_at.desc())
                .all()
            )
            
            productos_lote = [
                {
                    "nombre": r.product_name,
                    "stock": r.quantity_on_hand,
                    "ventas_diarias": r.average_daily_usage
                }
                for r in registros
            ]
            
            analisis = dias_stock_service.analizar_lote_productos(productos_lote)
            
            # Filtrar solo críticos y alertas para el email
            resultados_detallados = analisis.get('resultados_detallados', [])
            productos_importantes = [
                r for r in resultados_detallados 
                if "CRÍTICO" in r.get('estado', '') or "ALERTA" in r.get('estado', '')
            ]
            
            datos_email = [
                {
                    "nombre": p.get('producto', 'Desconocido'),
                    "stock_predicho": p.get('stock_actual', 0),
                    "estado": p.get('estado', 'OK')
                }
                for p in productos_importantes[:20]
            ]
            
        finally:
            db.close()
        
        # Enviar email
        fecha_actual = datetime.now().strftime('%Y-%m-%d')
        
        resultado_envio = email_service.enviar_reporte_prediccion(
            destinatario=destinatario,
            fecha=fecha_actual,
            predicciones=datos_email,
            resumen=resumen
        )
        
        # Generar mensaje con LLM sobre el resultado del envío
        if resultado_envio.get('exito'):
            llm_service = get_llm_service()
            mensaje_confirmacion = None
            
            if llm_service and incluir_analisis:
                try:
                    mensaje_confirmacion = llm_service.generar_mensaje_envio_email(
                        destinatario=destinatario,
                        tipo_reporte=asunto_tipo,
                        num_productos=len(datos_email),
                        fecha=fecha_actual,
                        resumen_contenido=resumen[:200]
                    )
                except Exception as llm_error:
                    print(f"Error generando mensaje con LLM: {llm_error}")
            
            # Mensaje fallback
            if not mensaje_confirmacion:
                mensaje_confirmacion = (
                    f"Reporte enviado exitosamente a {destinatario}. "
                    f"Se incluyó el análisis de {len(datos_email)} productos con fecha {fecha_actual}."
                )
            
            return APIResponse.success(
                message=mensaje_confirmacion,
                title=" Correo enviado",
                data={
                    "destinatario": destinatario,
                    "tipo_reporte": asunto_tipo,
                    "productos_enviados": len(datos_email),
                    "fecha": fecha_actual
                }
            )
        else:
            return APIResponse.error(
                message=f"No se pudo enviar el correo a {destinatario}.",
                error_detail=resultado_envio.get('error', 'Error desconocido'),
                title=" Error al enviar correo"
            )
        
    except Exception as e:
        print(f"Error en enviar_correo: {e}")
        import traceback
        traceback.print_exc()
        return APIResponse.error(
            message="No se pudo completar el envío del correo electrónico.",
            error_detail=str(e),
            title=" Error al procesar envío"
        )

def exportar_pdf(data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Genera reporte PDF de predicciones y opcionalmente lo envía por email"""
    try:
        from export_service import get_export_service
        from email_service import get_email_service
        
        # Parámetros
        fecha = data.get('fecha') if data else datetime.now().strftime('%Y-%m-%d')
        enviar_email = data.get('enviar_email', False) if data else False
        destinatario = data.get('destinatario') if data else None
        
        if enviar_email and not destinatario:
            destinatario = 'venotacu@gmail.com'
        
        # OBTENER PREDICCIONES DIRECTAMENTE (no usar predecir_all_stock)
        productos = all_registers_priductos()
        predicciones = []
        
        for prod in productos:
            features = preparar_input_desde_dataset_procesado(sku=prod, fecha_override=fecha)
            
            if features is not None and np.any(features):
                pred = modelo.predecir(features)
                nombre = buscar_nombre_por_sku(prod)
                minimum_stock = obtener_minimum_stock_level(prod) or 20.0
                
                # Determinar estado
                if pred < minimum_stock:
                    estado = "CRÍTICO"
                elif pred < minimum_stock * 1.5:
                    estado = "ALERTA"
                else:
                    estado = "OK"
                
                predicciones.append({
                    "sku": prod,
                    "nombre": nombre,
                    "stock_predicho": float(pred),
                    "minimum_stock": minimum_stock,
                    "estado": estado
                })
        
        if not predicciones:
            return APIResponse.error(
                message="No hay productos con datos suficientes para generar el PDF.",
                title=" Sin datos"
            )
        
        # Generar mensaje con LLM
        try:
            llm_service = get_llm_service()
            
            # Preparar datos para el LLM
            criticos = [p for p in predicciones if p['estado'] == 'CRÍTICO']
            alertas = [p for p in predicciones if p['estado'] == 'ALERTA']
            
            mensaje_llm = f"""
Resumen Ejecutivo - Predicción de Stock para {fecha}

TOTAL PRODUCTOS ANALIZADOS: {len(predicciones)}
- Productos en estado CRÍTICO: {len(criticos)}
- Productos en ALERTA: {len(alertas)}
- Productos en estado OK: {len(predicciones) - len(criticos) - len(alertas)}

PRODUCTOS PRIORITARIOS:
{chr(10).join([f"• {p['nombre']}: {p['stock_predicho']:.1f} unidades ({p['estado']})" for p in criticos[:5]])}

RECOMENDACIONES:
- Reabastecer inmediatamente los productos en estado CRÍTICO
- Monitorear productos en ALERTA para el próximo pedido
- Mantener niveles actuales para productos OK
            """
            
        except Exception as e:
            print(f"Error al generar mensaje LLM: {e}")
            mensaje_llm = f"Análisis de {len(predicciones)} productos para la fecha {fecha}"
        
        # Generar PDF
        export_service = get_export_service()
        pdf_path = export_service.generar_pdf_reporte(
            fecha=fecha,
            predicciones=predicciones,
            mensaje_llm=mensaje_llm,
            tipo_reporte="stock_predictions"
        )
        
        nombre_archivo = Path(pdf_path).name
        url_descarga = f"/descargar-pdf/{nombre_archivo}"
        
        # Enviar por email si se solicitó
        email_enviado = False
        if enviar_email and destinatario:
            try:
                email_service = get_email_service()
                pdf_bytes = export_service.leer_pdf_como_bytes(pdf_path)
                
                resultado_email = email_service.enviar_reporte_con_pdf(
                    destinatario=destinatario,
                    fecha=fecha,
                    pdf_bytes=pdf_bytes,
                    nombre_archivo=nombre_archivo,
                    resumen=f"{len(predicciones)} productos analizados"
                )
                
                email_enviado = resultado_email.get('exito', False)
            except Exception as e:
                print(f"Error al enviar email: {e}")
        
        # Construir respuesta
        if email_enviado:
            mensaje = f" PDF generado y enviado a {destinatario}. Descárgalo: {url_descarga}"
            title = " PDF enviado"
        else:
            mensaje = f" PDF generado. Descárgalo: {url_descarga}"
            title = " PDF generado"
        
        return APIResponse.success(
            message=mensaje,
            title=title,
            data={
                "archivo_generado": nombre_archivo,
                "url_descarga": url_descarga,
                "total_productos": len(predicciones),
                "email_enviado": email_enviado,
                "destinatario": destinatario if email_enviado else None
            }
        )
        
    except Exception as e:
        print(f"Error en exportar_pdf: {e}")
        import traceback
        traceback.print_exc()
        return APIResponse.error(
            message="No se pudo completar la exportación a PDF.",
            error_detail=str(e),
            title=" Error en exportación"
        )

# Mapeo de acciones disponibles
ACTIONS_MAP = {
    "predecir_all_stock": predecir_all_stock,
    "productos_criticos": productos_criticos,
    "enviar_correo": enviar_correo,
    "exportar_pdf": exportar_pdf
}