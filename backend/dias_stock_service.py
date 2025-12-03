"""
Servicio de cálculo de días de stock restante
Miembro 3: Samantha Suquilanda
Empresa: UPS Tuti
"""
from typing import Dict, List
import math

class DiasStockService:
    def __init__(self):
        """Inicializa el servicio de cálculo de días de stock"""
        self.umbral_critico_dias = 7  # Menos de 7 días = crítico
        self.umbral_alerta_dias = 14  # Menos de 14 días = alerta
    
    def calcular_dias_restantes(
        self, 
        stock_actual: float, 
        ventas_diarias: float,
        nombre_producto: str = "Producto"
    ) -> Dict:
        """
        Calcula cuántos días durará el stock actual
        
        Args:
            stock_actual: Unidades disponibles ahora
            ventas_diarias: Promedio de unidades vendidas por día
            nombre_producto: Nombre del producto
        
        Returns:
            Dict con análisis de días restantes
        """
        # Evitar división por cero
        if ventas_diarias == 0:
            return {
                "producto": nombre_producto,
                "stock_actual": stock_actual,
                "ventas_diarias": ventas_diarias,
                "dias_restantes": float('inf'),
                "estado": "SIN MOVIMIENTO",
                "urgencia": "BAJA",
                "recomendacion": "Producto sin ventas. Revisar si está descontinuado."
            }
        
        # Calcular días
        dias_restantes = stock_actual / ventas_diarias
        
        # Clasificar urgencia
        if dias_restantes < self.umbral_critico_dias:
            estado = " CRÍTICO"
            urgencia = "URGENTE"
            recomendacion = f"¡ATENCIÓN! Solo quedan {math.ceil(dias_restantes)} días de stock. Pedir HOY al proveedor."
        elif dias_restantes < self.umbral_alerta_dias:
            estado = " ALERTA"
            urgencia = "MEDIA"
            recomendacion = f"Quedan {math.ceil(dias_restantes)} días. Planificar pedido para esta semana."
        else:
            estado = " OK"
            urgencia = "BAJA"
            recomendacion = f"Stock suficiente para {math.ceil(dias_restantes)} días. Sin acción requerida."
        
        # Calcular fecha estimada de agotamiento
        from datetime import datetime, timedelta
        fecha_agotamiento = datetime.now() + timedelta(days=dias_restantes)
        
        return {
            "producto": nombre_producto,
            "stock_actual": stock_actual,
            "ventas_diarias": ventas_diarias,
            "dias_restantes": round(dias_restantes, 1),
            "fecha_agotamiento_estimada": fecha_agotamiento.strftime("%Y-%m-%d"),
            "estado": estado,
            "urgencia": urgencia,
            "recomendacion": recomendacion
        }
    
    def analizar_lote_productos(self, productos: List[Dict]) -> Dict:
        """
        Analiza un lote completo de productos
        
        Args:
            productos: Lista de diccionarios con datos de productos
                Ejemplo: [{"nombre": "...", "stock": 150, "ventas_diarias": 12}, ...]
        
        Returns:
            Reporte completo con análisis
        """
        resultados = []
        
        for prod in productos:
            resultado = self.calcular_dias_restantes(
                stock_actual=prod['stock'],
                ventas_diarias=prod['ventas_diarias'],
                nombre_producto=prod['nombre']
            )
            resultados.append(resultado)
        
        # Ordenar por urgencia (críticos primero)
        resultados_ordenados = sorted(
            resultados, 
            key=lambda x: x['dias_restantes']
        )
        
        # Estadísticas
        criticos = sum(1 for r in resultados if "CRÍTICO" in r['estado'])
        alertas = sum(1 for r in resultados if "ALERTA" in r['estado'])
        ok = sum(1 for r in resultados if "OK" in r['estado'])
        
        return {
            "total_productos": len(resultados),
            "productos_criticos": criticos,
            "productos_alerta": alertas,
            "productos_ok": ok,
            "resultados_detallados": resultados_ordenados,
            "resumen": self._generar_resumen(criticos, alertas, ok)
        }
    
    def _generar_resumen(self, criticos: int, alertas: int, ok: int) -> str:
        """Genera resumen ejecutivo"""
        total = criticos + alertas + ok
        
        if criticos > 0:
            return f" ATENCIÓN: {criticos} producto(s) en estado CRÍTICO. {alertas} en alerta. {ok} con stock OK."
        elif alertas > 0:
            return f" ALERTA: {alertas} producto(s) requieren planificación de pedido. {ok} con stock suficiente."
        else:
            return f" OK: Todos los {total} productos tienen stock suficiente."
# ========================================
# PRUEBA (CON DATOS REALISTAS)
# ========================================

if __name__ == "__main__":
    print("===  PRUEBA DE CÁLCULO DE DÍAS DE STOCK ===\n")
    
    # Datos de ejemplo basados en el dataset
    productos_ejemplo = [
        {"nombre": "Barra Cereal Choco", "stock": 156.86, "ventas_diarias": 20},  # ~8 días
        {"nombre": "Galletas Chocolate Chip", "stock": 124, "ventas_diarias": 36},  # ~3 días 
        {"nombre": "Chips Verde Lima", "stock": 188.81, "ventas_diarias": 10},  # ~19 días
        {"nombre": "Palomitas Caramelo", "stock": 201, "ventas_diarias": 8},  # ~25 días
        {"nombre": "Barra Proteica Frutos", "stock": 71.25, "ventas_diarias": 15},  # ~5 días 
    ]
    
    servicio = DiasStockService()
    reporte = servicio.analizar_lote_productos(productos_ejemplo)
    
    print(" REPORTE DE DÍAS DE STOCK RESTANTE - UPS TUTI\n")
    print(f"{reporte['resumen']}\n")
    print(f"Total analizado: {reporte['total_productos']} productos")
    print(f" Críticos: {reporte['productos_criticos']}")
    print(f" Alerta: {reporte['productos_alerta']}")
    print(f" OK: {reporte['productos_ok']}\n")
    
    print("--- DETALLE POR PRODUCTO (Ordenado por Urgencia) ---\n")
    for r in reporte['resultados_detallados']:
        print(f"{r['estado']} {r['producto']}")
        print(f"   Stock actual: {r['stock_actual']:.0f} unidades")
        print(f"   Ventas diarias: {r['ventas_diarias']:.0f} unidades/día")
        print(f"   Días restantes: {r['dias_restantes']:.1f} días")
        print(f"   Se agotará: {r['fecha_agotamiento_estimada']}")
        print(f"    {r['recomendacion']}\n")