import pandas as pd
from datetime import datetime



def obtener_registro_advanced_por_fecha(sku,fecha):
    # Cargar el CSV
    df = pd.read_csv(("dataset_processed_advanced.csv"))
    formatos = ["%Y-%m-%d", "%d/%m/%Y", "%Y%m%d"]
    fecha_dt = None
    for f in formatos:
        try:
            fecha_dt = datetime.strptime(fecha, f)
            break
        except ValueError:
            pass
    if fecha_dt is None:
        raise ValueError("Formato de fecha no válido. Usa: YYYY-MM-DD, DD/MM/YYYY o YYYYMMDD")
    anio = fecha_dt.year
    mes = fecha_dt.month
    dia = fecha_dt.day
    # Filtrar por SKU
    filtrado = df[df["product_sku"] == sku]

    if filtrado.empty:
        return None  

    
    fila = filtrado.iloc[-1]
    features = {
        "prioridad_proveedor": fila["prioridad_proveedor"],
        "quantity_on_hand": fila["quantity_on_hand"],
        "quantity_reserved": fila["quantity_reserved"],
        "minimum_stock_level": fila["minimum_stock_level"],
        "reorder_point": fila["reorder_point"],
        "optimal_stock_level": fila["optimal_stock_level"],
        "reorder_quantity": fila["reorder_quantity"],
        "average_daily_usage": fila["average_daily_usage"],
        "unit_cost": fila["unit_cost"],
        "total_value": fila["total_value"],
        "is_active": fila["is_active"],
        "anio": anio,
        "mes": mes,
        "vacaciones_o_no": fila["vacaciones_o_no"],
        "es_feriado": fila["es_feriado"],
        "temporada_alta": fila["temporada_alta"],
        "semana_del_anio": fila["semana_del_anio"],
        "dia_del_mes": dia,
        "dia_de_la_semana": fila["dia_de_la_semana"],
        "es_fin_de_semana": fila["es_fin_de_semana"],
        "trimestre": fila["trimestre"],
        "lag_1": fila["lag_1"],
        "lag_7": fila["lag_7"],
        "lag_30": fila["lag_30"],
        "media_movil_7d": fila["media_movil_7d"],
        "media_movil_30d": fila["media_movil_30d"],
        "media_movil_exponencial": fila["media_movil_exponencial"],
        "std_movil_30d": fila["std_movil_30d"],
        "variacion_stock_diaria": fila["variacion_stock_diaria"],
        "tendencia_stock": fila["tendencia_stock"],
        "dias_desde_ultimo_pedido": fila["dias_desde_ultimo_pedido"],
        "ratio_reservado_disponible": fila["ratio_reservado_disponible"],
        "anomalia_stock": fila["anomalia_stock"],
        "rotacion_estimada_30d": fila["rotacion_estimada_30d"],
        "estacion_Otoño": fila["estacion_Otoño"],
        "estacion_Primavera": fila["estacion_Primavera"],
        "estacion_Verano": fila["estacion_Verano"],
    }
    
    return features

def all_registers_priductos():
    
    df = pd.read_csv("dataset_processed_advanced.csv")

    sku_unicos = df["product_sku"].unique()
    
    return sku_unicos



# def obtener_registro_advanced( sku):
#     # Cargar el CSV
#     df = pd.read_csv(("dataset_processed_advanced.csv"))

#     # Filtrar por SKU
#     filtrado = df[df["product_sku"] == sku]

#     if filtrado.empty:
#         return None  # no existe en el CSV

#     # Tomar la última fila que aparece
#     fila = filtrado.iloc[-1]
#     features = {
#         "prioridad_proveedor": fila["prioridad_proveedor"],
#         "quantity_on_hand": fila["quantity_on_hand"],
#         "quantity_reserved": fila["quantity_reserved"],
#         "minimum_stock_level": fila["minimum_stock_level"],
#         "reorder_point": fila["reorder_point"],
#         "optimal_stock_level": fila["optimal_stock_level"],
#         "reorder_quantity": fila["reorder_quantity"],
#         "average_daily_usage": fila["average_daily_usage"],
#         "unit_cost": fila["unit_cost"],
#         "total_value": fila["total_value"],
#         "is_active": fila["is_active"],
#         "anio": fila["ania"],
#         "mes": fila["mes"],
#         "vacaciones_o_no": fila["vacaciones_o_no"],
#         "es_feriado": fila["es_feriado"],
#         "temporada_alta": fila["temporada_alta"],
#         "semana_del_anio": fila["semana_del_anio"],
#         "dia_del_mes": fila["dia_del_mes"],
#         "dia_de_la_semana": fila["dia_de_la_semana"],
#         "es_fin_de_semana": fila["es_fin_de_semana"],
#         "trimestre": fila["trimestre"],
#         "lag_1": fila["lag_1"],
#         "lag_7": fila["lag_7"],
#         "lag_30": fila["lag_30"],
#         "media_movil_7d": fila["media_movil_7d"],
#         "media_movil_30d": fila["media_movil_30d"],
#         "media_movil_exponencial": fila["media_movil_exponencial"],
#         "std_movil_30d": fila["std_movil_30d"],
#         "variacion_stock_diaria": fila["variacion_stock_diaria"],
#         "tendencia_stock": fila["tendencia_stock"],
#         "dias_desde_ultimo_pedido": fila["dias_desde_ultimo_pedido"],
#         "ratio_reservado_disponible": fila["ratio_reservado_disponible"],
#         "anomalia_stock": fila["anomalia_stock"],
#         "rotacion_estimada_30d": fila["rotacion_estimada_30d"],
#         "estacion_Otoño": fila["estacion_Otoño"],
#         "estacion_Primavera": fila["estacion_Primavera"],
#         "estacion_Verano": fila["estacion_Verano"],
#     }
#     return features

    
