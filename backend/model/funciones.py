"""
Funciones de negocio completas integradas con la base de datos
UPS Tuti - Sistema de Gestión de Inventario
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
import re

# Importar tus modelos y servicios
from model.registro import Registro  # Tu modelo SQLAlchemy
from dias_stock_service import DiasStockService
from email_service import EmailService

# ============================================================================
# SERVICIOS SINGLETON
# ============================================================================

dias_stock_service = DiasStockService()
email_service = EmailService()


# ============================================================================
# FUNCIÓN 1: PREDECIR STOCK
# ============================================================================

def predecir_stock(mensaje: str, db: Session, params: Optional[Dict] = None) -> Dict:
    """Predice el stock futuro de un producto específico"""
    try:
        sku_extraido = None
        nombre_extraido = None
        dias_prediccion = 7
        
        sku_match = re.search(r'SKU[:\s-]*(\w+)', mensaje, re.IGNORECASE)
        if sku_match:
            sku_extraido = sku_match.group(1)
        
        if params:
            sku_extraido = params.get('sku', sku_extraido)
            nombre_extraido = params.get('nombre', nombre_extraido)
            dias_prediccion = params.get('dias', dias_prediccion)
        
        query = db.query(Registro).filter(Registro.is_active == True)
        
        if sku_extraido:
            query = query.filter(Registro.product_sku.ilike(f"%{sku_extraido}%"))
        elif nombre_extraido:
            query = query.filter(Registro.product_name.ilike(f"%{nombre_extraido}%"))
        else:
            palabras = mensaje.lower().split()
            for palabra in palabras:
                if len(palabra) > 3:
                    query = query.filter(Registro.product_name.ilike(f"%{palabra}%"))
                    break
        
        registro = query.order_by(desc(Registro.created_at)).first()
        
        if not registro:
            return {
                "exito": False,
                "mensaje": "No se encontró el producto en la base de datos",
                "sugerencia": "Verifica el SKU o nombre del producto"
            }
        
        analisis = dias_stock_service.calcular_dias_restantes(
            stock_actual=registro.quantity_available or registro.quantity_on_hand,
            ventas_diarias=registro.average_daily_usage or 0,
            nombre_producto=registro.product_name
        )
        
        stock_predicho = registro.quantity_on_hand - (registro.average_daily_usage * dias_prediccion)
        stock_predicho = max(0, stock_predicho)
        
        necesita_reorden = registro.quantity_on_hand <= registro.reorder_point
        
        return {
            "exito": True,
            "producto": {
                "id": registro.product_id,
                "nombre": registro.product_name,
                "sku": registro.product_sku,
                "categoria": registro.categoria_producto
            },
            "stock_actual": {
                "en_mano": registro.quantity_on_hand,
                "disponible": registro.quantity_available,
                "ubicacion": f"{registro.warehouse_location} - {registro.shelf_location}"
            },
            "analisis_dias": {
                "dias_restantes": analisis['dias_restantes'],
                "fecha_agotamiento": analisis['fecha_agotamiento_estimada'],
                "estado": analisis['estado'],
                "urgencia": analisis['urgencia']
            },
            "prediccion": {
                "dias_proyectados": dias_prediccion,
                "stock_estimado": round(stock_predicho, 2),
                "alcanza": stock_predicho > 0
            },
            "necesita_reorden": necesita_reorden,
            "recomendacion": analisis['recomendacion']
        }
    
    except Exception as e:
        return {
            "exito": False,
            "error": f"Error al predecir stock: {str(e)}"
        }


# ============================================================================
# FUNCIÓN 2: GENERAR ALERTA
# ============================================================================

def generar_alerta(mensaje: str, db: Session, params: Optional[Dict] = None) -> Dict:
    """Genera alertas de bajo stock para productos críticos"""
    try:
        tipo_alerta = params.get('tipo', 'critico') if params else 'critico'
        
        subquery = (
            db.query(
                Registro.product_name,
                func.max(Registro.created_at).label('max_date')
            )
            .filter(Registro.is_active == True)
            .group_by(Registro.product_name)
            .subquery()
        )
        
        registros = (
            db.query(Registro)
            .join(
                subquery,
                (Registro.product_name == subquery.c.product_name) &
                (Registro.created_at == subquery.c.max_date)
            )
            .all()
        )
        
        if not registros:
            return {
                "exito": False,
                "mensaje": "No hay registros activos para analizar"
            }
        
        productos_lote = [
            {
                "nombre": r.product_name,
                "stock": r.quantity_available or r.quantity_on_hand,
                "ventas_diarias": r.average_daily_usage or 0
            }
            for r in registros
        ]
        
        analisis = dias_stock_service.analizar_lote_productos(productos_lote)
        
        alertas_generadas = []
        
        for resultado in analisis['resultados_detallados']:
            incluir = False
            
            if tipo_alerta == 'critico':
                incluir = 'CRÍTICO' in resultado['estado']
            elif tipo_alerta == 'todo':
                incluir = True
            elif tipo_alerta == 'alerta':
                incluir = 'ALERTA' in resultado['estado'] or 'CRÍTICO' in resultado['estado']
            
            if incluir:
                reg = next((r for r in registros if r.product_name == resultado['producto']), None)
                
                alerta = {
                    "producto": resultado['producto'],
                    "sku": reg.product_sku if reg else "N/A",
                    "dias_restantes": resultado['dias_restantes'],
                    "estado": resultado['estado'],
                    "urgencia": resultado['urgencia'],
                    "stock_actual": resultado['stock_actual'],
                    "recomendacion": resultado['recomendacion']
                }
                alertas_generadas.append(alerta)
        
        alertas_generadas.sort(key=lambda x: x['dias_restantes'])
        
        return {
            "exito": True,
            "id_alerta": f"ALT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "fecha_generacion": datetime.now().isoformat(),
            "resumen": {
                "total_productos_analizados": analisis['total_productos'],
                "productos_criticos": analisis['productos_criticos'],
                "alertas_generadas": len(alertas_generadas)
            },
            "alertas": alertas_generadas,
            "mensaje_ejecutivo": analisis['resumen']
        }
    
    except Exception as e:
        return {
            "exito": False,
            "error": f"Error al generar alerta: {str(e)}"
        }


# ============================================================================
# FUNCIÓN 3: BUSCAR PRODUCTO
# ============================================================================

def buscar_producto(mensaje: str, db: Session, params: Optional[Dict] = None) -> Dict:
    """Busca productos en la base de datos"""
    try:
        sku_buscar = None
        nombre_buscar = None
        limit = 10
        
        if params:
            sku_buscar = params.get('sku')
            nombre_buscar = params.get('nombre')
            limit = params.get('limit', 10)
        
        if not any([sku_buscar, nombre_buscar]):
            sku_match = re.search(r'SKU[:\s-]*(\w+)', mensaje, re.IGNORECASE)
            if sku_match:
                sku_buscar = sku_match.group(1)
            else:
                nombre_buscar = mensaje.replace('buscar', '').replace('producto', '').strip()
        
        query = db.query(Registro).filter(Registro.is_active == True)
        
        if sku_buscar:
            query = query.filter(Registro.product_sku.ilike(f"%{sku_buscar}%"))
        
        if nombre_buscar:
            query = query.filter(Registro.product_name.ilike(f"%{nombre_buscar}%"))
        
        subquery = (
            query
            .with_entities(
                Registro.product_name,
                func.max(Registro.created_at).label('max_date')
            )
            .group_by(Registro.product_name)
            .subquery()
        )
        
        resultados = (
            db.query(Registro)
            .join(
                subquery,
                (Registro.product_name == subquery.c.product_name) &
                (Registro.created_at == subquery.c.max_date)
            )
            .limit(limit)
            .all()
        )
        
        if not resultados:
            return {
                "encontrado": False,
                "mensaje": "No se encontraron productos con esos criterios"
            }
        
        productos_encontrados = []
        
        for reg in resultados:
            dias_info = dias_stock_service.calcular_dias_restantes(
                stock_actual=reg.quantity_available or reg.quantity_on_hand,
                ventas_diarias=reg.average_daily_usage or 0,
                nombre_producto=reg.product_name
            )
            
            producto = {
                "id": reg.product_id,
                "nombre": reg.product_name,
                "sku": reg.product_sku,
                "categoria": reg.categoria_producto,
                "stock": {
                    "en_mano": reg.quantity_on_hand,
                    "disponible": reg.quantity_available,
                    "dias_restantes": dias_info['dias_restantes'],
                    "urgencia": dias_info['urgencia']
                },
                "ubicacion": {
                    "almacen": reg.warehouse_location,
                    "region": reg.region_almacen
                },
                "proveedor": reg.supplier_name
            }
            
            productos_encontrados.append(producto)
        
        return {
            "encontrado": True,
            "total_resultados": len(productos_encontrados),
            "productos": productos_encontrados
        }
    
    except Exception as e:
        return {
            "encontrado": False,
            "error": f"Error al buscar producto: {str(e)}"
        }


# ============================================================================
# FUNCIÓN 4: ENVIAR CORREO
# ============================================================================

def enviar_correo(mensaje: str, db: Session, params: Optional[Dict] = None) -> Dict:
    """Envía correo electrónico con reportes de inventario"""
    try:
        destinatarios = []
        
        if params and 'destinatarios' in params:
            destinatarios = params['destinatarios']
        else:
            emails_encontrados = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', mensaje)
            destinatarios = emails_encontrados
        
        if not destinatarios:
            destinatarios = ["gerencia@upstuti.com"]
        
        incluir_criticos = params.get('incluir_criticos', True) if params else True
        
        subquery = (
            db.query(
                Registro.product_name,
                func.max(Registro.created_at).label('max_date')
            )
            .filter(Registro.is_active == True)
            .group_by(Registro.product_name)
            .subquery()
        )
        
        registros = (
            db.query(Registro)
            .join(
                subquery,
                (Registro.product_name == subquery.c.product_name) &
                (Registro.created_at == subquery.c.max_date)
            )
            .all()
        )
        
        productos_lote = [
            {
                "nombre": r.product_name,
                "stock": r.quantity_available or r.quantity_on_hand,
                "ventas_diarias": r.average_daily_usage or 0
            }
            for r in registros
        ]
        
        analisis = dias_stock_service.analizar_lote_productos(productos_lote)
        
        productos_para_reporte = []
        
        for resultado in analisis['resultados_detallados']:
            if incluir_criticos:
                if 'CRÍTICO' in resultado['estado'] or 'ALERTA' in resultado['estado']:
                    productos_para_reporte.append({
                        "nombre": resultado['producto'],
                        "stock_predicho": resultado['stock_actual'],
                        "estado": resultado['estado']
                    })
            else:
                productos_para_reporte.append({
                    "nombre": resultado['producto'],
                    "stock_predicho": resultado['stock_actual'],
                    "estado": resultado['estado']
                })
        
        if not productos_para_reporte:
            return {
                "exito": False,
                "mensaje": "No hay productos que cumplan los criterios para el reporte"
            }
        
        resultados_envio = []
        
        for destinatario in destinatarios:
            resultado = email_service.enviar_reporte_prediccion(
                destinatario=destinatario,
                fecha=datetime.now().strftime("%Y-%m-%d"),
                predicciones=productos_para_reporte,
                resumen=analisis['resumen']
            )
            
            resultados_envio.append({
                "destinatario": destinatario,
                "exito": resultado['exito']
            })
        
        envios_exitosos = sum(1 for r in resultados_envio if r['exito'])
        
        return {
            "exito": envios_exitosos > 0,
            "total_destinatarios": len(destinatarios),
            "envios_exitosos": envios_exitosos,
            "destinatarios": destinatarios,
            "resultados": resultados_envio,
            "mensaje": f"Se enviaron {envios_exitosos} de {len(destinatarios)} emails correctamente"
        }
    
    except Exception as e:
        return {
            "exito": False,
            "error": f"Error al enviar correo: {str(e)}"
        }


# ============================================================================
# MAPEO DE FUNCIONES
# ============================================================================

FUNCIONES_DISPONIBLES = {
    "predecir_stock": predecir_stock,
    "generar_alerta": generar_alerta,
    "buscar_producto": buscar_producto,
    "enviar_correo": enviar_correo,
}
