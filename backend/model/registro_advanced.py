import pandas as pd
from datetime import datetime
import numpy as np
from joblib import load
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.stats import zscore
from pathlib import Path
from paths import resolve_file, FILES_DIR


scaler_X = load(str(resolve_file("scaler_X.joblib")))
scaler_y = load(str(resolve_file("scaler_y.joblib")))

# Cargar CSV una vez (ruta resuelta)
DF_PRODUCTOS = pd.read_csv(str(resolve_file("dataset.csv")))


def all_registers_priductos():
    
    df = pd.read_csv("model/files/dataset_processed_advanced.csv")
    sku_unicos = df["product_sku"].unique()
    
    return sku_unicos

def buscar_nombre_por_sku(product_sku: str) -> str | dict:
    fila = DF_PRODUCTOS[DF_PRODUCTOS["product_sku"] == product_sku]
    if fila.empty:
        return {"mensaje": f"No se encontró el product_sku '{product_sku}'"}
    return fila["product_name"].iloc[0]

def buscar_producto_por_id(product_id: int) -> str | dict:
    fila = DF_PRODUCTOS[DF_PRODUCTOS["product_id"] == product_id]
    if fila.empty:
        return {"mensaje": f"No se encontró el product_id {product_id}"}
    # Retorna solo un product_sku
    return fila["product_sku"].iloc[0]

def buscar_producto_por_nombre(nombre: str) -> str | dict:
    coincidencias = DF_PRODUCTOS[DF_PRODUCTOS["product_name"].str.contains(nombre, case=False, na=False)]
    print(coincidencias)
    if coincidencias.empty:
        return {"mensaje": f"No se encontró el nombre '{nombre}'"}
    # Retorna solo el primer product_sku encontrado
    return coincidencias["product_sku"].iloc[0]

def preparar_input_desde_dataset_procesado(sku, fecha_override=None):
    scaler_X = load(str(resolve_file("scaler_X.joblib")))
    n_steps = 7

    # Cargar dataset procesado
    df = pd.read_csv(str(resolve_file("dataset_processed_advanced.csv")))

    # Filtrar SKU
    df_sku = df[df["product_sku"] == sku].copy()

    if len(df_sku) < n_steps:
        raise ValueError(f"SKU {sku} solo tiene {len(df_sku)} registroSe requieren {n_steps}.")

    # Ordenar por fecha (cronológicamente)
    df_sku = df_sku.sort_values(by=["anio", "mes", "dia_del_mes"])

    # Obtener últimos N días
    df_last = df_sku.tail(n_steps).copy()

    # Si el usuario quiere una fecha nueva, reemplazar la última fila
    if fecha_override is not None:
        fecha = pd.to_datetime(fecha_override)

        # Crear nueva fila copiando la última
        nueva_fila = df_last.iloc[-1].copy()

        # Reemplazar la fecha
        nueva_fila["anio"] = fecha.year
        nueva_fila["mes"] = fecha.month
        nueva_fila["dia_del_mes"] = fecha.day

        # Reemplazar la última fila del dataset por esta nueva
        df_last.iloc[-1] = nueva_fila

    # Eliminar columnas no usadas
    cols_excluir = ["product_sku", "region_almacen"]
    df_last = df_last.drop(columns=[c for c in cols_excluir if c in df_last.columns])

    # Mantener solo las columnas usadas en entrenamiento
    df_last = df_last[scaler_X.feature_names_in_]

    # Escalar
    df_scaled = scaler_X.transform(df_last)

    # Volver a forma GRU (batch_size = 1)
    X_input = df_scaled.reshape(1, n_steps, df_scaled.shape[1])
    
    return X_input


def procesar_dataset_inventario(ruta_csv="dataset.csv", 
                                 ruta_salida="dataset_processed_advanced2.csv",
                                 guardar=True):
    """
    Procesa el dataset de inventario aplicando feature engineering completo.
    
    Parámetros:
    -----------
    ruta_csv : str
        Ruta al archivo CSV original (default: "data/dataset.csv")
    ruta_salida : str
        Ruta donde guardar el dataset procesado (default: "data/dataset_processed_advanced.csv")
    guardar : bool
        Si True, guarda el dataset procesado en CSV (default: True)
    
    Retorna:
    --------
    pd.DataFrame
        DataFrame procesado con todas las features engineered
    
    """
    
    # ========== CARGA Y PREPARACIÓN BASE ==========
    print("Cargando dataset...")
    ruta_csv_path = resolve_file(ruta_csv)
    df = pd.read_csv(ruta_csv_path)
    
    # Conversión de fechas
    date_cols = ['created_at', 'last_order_date', 'expiration_date', 
                 'last_stock_count_date', 'last_updated_at']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Definición de claves y variable objetivo
    ID_PRODUCTO = 'product_sku'
    FECHA_PRINCIPAL = 'created_at'
    VAR_OBJETIVO = 'quantity_available'
    
    # Ordenamiento crítico para lags y medias móviles
    df[FECHA_PRINCIPAL] = df[FECHA_PRINCIPAL].fillna(method='ffill')
    df = df.sort_values(by=[ID_PRODUCTO, FECHA_PRINCIPAL]).reset_index(drop=True)
    
    print(f"✓ Dataset ordenado por '{ID_PRODUCTO}' y '{FECHA_PRINCIPAL}'")
    
    
    # ========== FEATURE ENGINEERING: VARIABLES TEMPORALES ==========
    print("Creando variables temporales...")
    
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Invierno'
        elif month in [3, 4, 5]:
            return 'Primavera'
        elif month in [6, 7, 8]:
            return 'Verano'
        else:
            return 'Otoño'
    
    fecha = df[FECHA_PRINCIPAL]
    df['semana_del_anio'] = fecha.dt.isocalendar().week
    df['dia_del_mes'] = fecha.dt.day
    df['dia_de_la_semana'] = fecha.dt.dayofweek
    df['es_fin_de_semana'] = (df['dia_de_la_semana'] >= 5).astype(int)
    df['trimestre'] = fecha.dt.quarter
    df['estacion'] = df['mes'].apply(get_season)
    
    # Conversión de booleanos a enteros
    df['vacaciones_o_no'] = df['vacaciones_o_no'].astype(int)
    df['es_feriado'] = df['es_feriado'].astype(int)
    df['temporada_alta'] = df['temporada_alta'].astype(int)
    
    print("Variables temporales creadas")
    
    
    # ========== FEATURE ENGINEERING: LAGS ==========
    print("Creando variables lag...")
    
    df['lag_1'] = df.groupby(ID_PRODUCTO)[VAR_OBJETIVO].shift(1)
    df['lag_7'] = df.groupby(ID_PRODUCTO)[VAR_OBJETIVO].shift(7)
    df['lag_30'] = df.groupby(ID_PRODUCTO)[VAR_OBJETIVO].shift(30)
    
    print("Variables lag (1, 7, 30) creadas")
    
    
    # ========== FEATURE ENGINEERING: HISTÓRICAS Y ESTADÍSTICAS ==========
    print("Creando medias móviles y estadísticas...")
    
    df['media_movil_7d'] = df.groupby(ID_PRODUCTO)[VAR_OBJETIVO].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    df['media_movil_30d'] = df.groupby(ID_PRODUCTO)[VAR_OBJETIVO].transform(
        lambda x: x.rolling(30, min_periods=1).mean()
    )
    df['media_movil_exponencial'] = df.groupby(ID_PRODUCTO)[VAR_OBJETIVO].transform(
        lambda x: x.ewm(span=7, adjust=False).mean()
    )
    df['std_movil_30d'] = df.groupby(ID_PRODUCTO)[VAR_OBJETIVO].transform(
        lambda x: x.rolling(30, min_periods=1).std()
    )
    df['std_movil_30d'] = df['std_movil_30d'].fillna(0)
    
    print("Medias móviles y desviación estándar creadas")
    
    
    # ========== FEATURE ENGINEERING: VARIABLES SINTÉTICAS ==========
    print("Creando variables sintéticas...")
    
    # Variación diaria
    df['variacion_stock_diaria'] = df.groupby(ID_PRODUCTO)[VAR_OBJETIVO].transform('diff')
    
    # Tendencia
    df['tendencia_stock'] = df['media_movil_7d'] - df['media_movil_30d']
    
    # Días desde último pedido
    df['dias_desde_ultimo_pedido'] = (df[FECHA_PRINCIPAL] - df['last_order_date']).dt.days
    df['dias_desde_ultimo_pedido'] = df['dias_desde_ultimo_pedido'].fillna(9999)
    
    # Ratio reservado/disponible
    df['ratio_reservado_disponible'] = np.where(
        df[VAR_OBJETIVO] > 0,
        df['quantity_reserved'] / df[VAR_OBJETIVO],
        0
    )
    
    # Anomalía (Z-score)
    df['anomalia_stock'] = np.where(
        df['std_movil_30d'] > 0,
        (df[VAR_OBJETIVO] - df['media_movil_30d']) / df['std_movil_30d'],
        0
    )
    
    # Rotación estimada
    df['rotacion_estimada_30d'] = np.where(
        df['media_movil_30d'] > 0,
        (df['average_daily_usage'] * 30) / df['media_movil_30d'],
        0
    )
    
    print("Variables sintéticas creadas")
    
    
    # ========== LIMPIEZA FINAL Y SELECCIÓN DE COLUMNAS ==========
    print("Limpiando datos y preparando dataset final...")
    
    # Eliminar filas con NaN en lags principales
    nulos_antes = df.isnull().sum().sum()
    df_processed = df.dropna(
        subset=['lag_1', 'lag_7', 'lag_30', 'variacion_stock_diaria']
    )
    nulos_despues = df_processed.isnull().sum().sum()
    
    print(f"Filas eliminadas por NaNs: {len(df) - len(df_processed):,}")
    
    # Codificación One-Hot para estación
    df_processed = pd.get_dummies(df_processed, columns=['estacion'], drop_first=True)
    
    # Columnas a excluir
    cols_a_excluir = [
        'id', 'product_id', 'supplier_id', 'created_by_id', 'record_sequence_number',
        'product_name', 'supplier_name', 'batch_number', 'warehouse_location',
        'shelf_location', 'stock_status', 'categoria_producto', 'subcategoria_producto',
        'created_at', 'last_order_date', 'last_stock_count_date', 
        'expiration_date', 'last_updated_at'
    ]
    
    # Filtrar solo columnas que existen
    cols_a_excluir = [col for col in cols_a_excluir if col in df_processed.columns]
    df_final = df_processed.drop(columns=cols_a_excluir, errors='ignore')
    
    
    # ========== GUARDADO ==========
    if guardar:
        try:
            ruta_salida_path = resolve_file(ruta_salida)
            df_final.to_csv(ruta_salida_path, index=False)
            print(f"\n ¡Éxito! Dataset procesado guardado en: {ruta_salida_path}")
        except Exception as e:
            print(f"\n Error al guardar: {e}")
    
    
    # ========== RESUMEN ==========
    print("\n" + "="*60)
    print("RESUMEN DEL PROCESAMIENTO")
    print("="*60)
    print(f"Filas originales:      {len(df):,}")
    print(f"Filas procesadas:      {len(df_final):,}")
    print(f"Columnas finales:      {len(df_final.columns)}")
    print(f"Variable objetivo:     {VAR_OBJETIVO}")
    print(f"Productos únicos:      {df_final[ID_PRODUCTO].nunique()}")
    print(f"Nulos restantes:       {df_final.isnull().sum().sum()}")
    print("="*60)
    
    return df_final