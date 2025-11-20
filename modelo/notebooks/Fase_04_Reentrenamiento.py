import marimo

__generated_with = "0.17.7"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Reentrenamiento del Modelo GRU con Nuevo Dataset

    Este notebook:
    1. **Procesa** el nuevo dataset usando el mismo Feature Engineering de Fase 01
    2. **Carga** el modelo previamente entrenado (`best_model.keras`)
    3. **Reentrena** el modelo con los nuevos datos procesados
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #  Índice de Contenidos

    Este notebook contiene las siguientes secciones:

    ## Fase 01: Feature Engineering
    - [Diccionario de Datos](#diccionario-de-datos)
    - [1. Carga y Preparación Base](#1-carga-y-preparación-base)
    - [2. Feature Engineering: Variables Temporales](#2-feature-engineering-variables-temporales)
    - [3. Feature Engineering: Lags (Variables Sintéticas)](#3-feature-engineering-lags-variables-sintéticas)
    - [4. Feature Engineering: Históricas y Estadísticas](#4-feature-engineering-históricas-y-estadísticas)
    - [5. Feature Engineering: Variables Sintéticas](#5-feature-engineering-variables-sintéticas)
    - [6. Limpieza Final y Guardado](#6-limpieza-final-y-guardado)
    - [7. Revisión del Dataset Final](#7-revisión-del-dataset-final)

    ## Fase 04: Reentrenamiento del Modelo
    - [8. Carga del Modelo Pre-entrenado y Configuración](#8-carga-del-modelo-pre-entrenado-y-configuración)
    - [9. Carga del Modelo y Escaladores Originales](#9-carga-del-modelo-y-escaladores-originales)
    - [10. Preparación de Datos para Reentrenamiento](#10-preparación-de-datos-para-reentrenamiento)
    - [11. Escalado y Creación de Secuencias Temporales](#11-escalado-y-creación-de-secuencias-temporales)
    - [12. División Train/Validation para Reentrenamiento](#12-división-trainvalidation-para-reentrenamiento)
    - [13. Reentrenamiento del Modelo](#13-reentrenamiento-del-modelo)
    - [14. Evaluación del Modelo Reentrenado](#14-evaluación-del-modelo-reentrenado)
    - [15. Visualización del Entrenamiento](#15-visualización-del-entrenamiento)

    ---

    **Variables clave:**
    - Variable objetivo: `quantity_available`
    - ID de producto: `product_sku`
    - Fecha principal: `created_at`
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    <a id="1-carga-y-preparación-base"></a>
    ## 1. Carga y Preparación Base

    1.  Cargamos el dataset.
    2.  Convertimos las columnas de fecha a `datetime`.
    3.  **Ordenamos el dataset** por producto (`product_sku`) y
        fecha (`created_at`). Esto es **fundamental** para
        que los cálculos de lags y medias móviles sean correctos.
    """)
    return


@app.cell
def _():
    # Manipulación y análisis de datos
    import pandas as pd
    import numpy as np

    # Configuración de warnings y pandas
    import warnings
    warnings.filterwarnings('ignore')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)

    print("Librerías importadas correctamente.")
    return np, pd


@app.cell
def _(mo):
    mo.md(r"""
    ## Diccionario de Datos del Dataset Original

    Basado en la estructura del dataset, este diccionario de datos describe cada columna:
    """)
    return


@app.cell
def _(mo, pd):
    data = {
        'Variable': [
            'id', 'created_at', 'product_id', 'product_name', 'product_sku',
            'supplier_id', 'supplier_name', 'prioridad_proveedor',
            'quantity_on_hand', 'quantity_reserved', 'quantity_available',
            'minimum_stock_level', 'reorder_point', 'optimal_stock_level',
            'reorder_quantity', 'average_daily_usage', 'last_order_date',
            'last_stock_count_date', 'unit_cost', 'total_value',
            'expiration_date', 'batch_number', 'warehouse_location',
            'shelf_location', 'region_almacen', 'stock_status', 'is_active',
            'last_updated_at', 'created_by_id', 'record_sequence_number',
            'categoria_producto', 'subcategoria_producto', 'anio', 'mes',
            'vacaciones_o_no', 'es_feriado', 'temporada_alta'
        ],
        'Tipo de Dato': [
            'int64', 'datetime64[ns]', 'int64', 'object', 'object',
            'int64', 'object', 'int64',
            'int64', 'int64', 'int64',
            'int64', 'int64', 'int64',
            'int64', 'float64', 'datetime64[ns]',
            'datetime64[ns]', 'float64', 'float64',
            'datetime64[ns]', 'object', 'object',
            'object', 'object', 'object', 'int64',
            'object', 'int64', 'int64',
            'object', 'object', 'int64', 'int64',
            'bool', 'bool', 'bool'
        ],
        'Descripción': [
            'Identificador único para cada registro o movimiento de inventario.',
            'Fecha y hora en que se creó el registro en el sistema.',
            'Identificador único para el producto.',
            'Nombre descriptivo del producto.',
            '(Stock Keeping Unit) Código único interno del producto.',
            'Identificador único del proveedor del producto.',
            'Nombre del proveedor.',
            'Nivel de prioridad asignado al proveedor (ej. 1=Alta, 5=Baja).',
            'Cantidad física total del producto actualmente en el almacén.',
            'Cantidad del producto que está apartada para pedidos pendientes.',
            'Cantidad real disponible para la venta (on_hand - reserved).',
            'Nivel mínimo de stock antes de que se considere "bajo stock".',
            'Nivel de stock en el cual se debe generar una nueva orden de compra.',
            'La cantidad ideal de stock que se desea mantener.',
            'Cantidad estándar que se pide en una nueva orden de compra.',
            'Promedio de unidades de este producto usadas o vendidas por día.',
            'Fecha en que se realizó la última orden de compra de este producto.',
            'Fecha del último conteo físico de este producto en el almacén.',
            'El costo de adquirir una sola unidad del producto.',
            'Valor total del stock a mano (quantity_on_hand * unit_cost).',
            'Fecha de caducidad del lote del producto (si aplica).',
            'Número de lote para trazabilidad.',
            'Ubicación general dentro del almacén (ej. "Bodega A", "Zona Fría").',
            'Ubicación específica en la estantería (ej. "Pasillo 3, Rack B").',
            'Región del almacén (ej. "Norte", "Sur").', #Descripción corregida
            'Estado actual del stock (ej. "activo", "obsoleto").',
            'Indicador binario de si el producto está activo (1) o inactivo (0).',
            'Fecha y hora de la última actualización del registro.',
            'Identificador del usuario que creó el registro.',
            'Número secuencial del registro dentro de un proceso.',
            'Categoría principal a la que pertenece el producto.',
            'Subcategoría específica del producto.',
            'Año de registro.',
            'Mes de registro.',
            'Indicador booleano si es período de vacaciones (True/False).',
            'Indicador booleano si la fecha es un feriado (True/False).',
            'Indicador booleano si la fecha corresponde a temporada alta (True/False).'
        ]
    }

    df_diccionario_original = pd.DataFrame(data)


    # Esta es la forma correcta de mostrar el DataFrame en la celda.
    mo.ui.dataframe(df_diccionario_original)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Carga y Preparación Base

    1.  Cargamos el dataset.
    2.  Convertimos las columnas de fecha a `datetime`.
    3.  **Ordenamos el dataset** por producto (`product_sku`) y
        fecha (`created_at`). Esto es **fundamental** para
        que los cálculos de lags y medias móviles sean correctos.
    """)
    return


@app.cell
def _(pd):

    df = pd.read_csv("C:/Users/samil/Desktop/APRENDIZAJE AUTOMATICO/PRIMER INTERCICLO/Practica-2-Aprendizaje-Automactico/data/dataset nuevo.csv")

    #  conversión de Fechas
    date_cols = ['created_at', 'last_order_date', 'expiration_date', 'last_stock_count_date', 'last_updated_at']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # definición de Claves
    ID_PRODUCTO = 'product_sku'      # SKU único del producto
    FECHA_PRINCIPAL = 'created_at'   # Fecha del registro

    # nuestra variable objetivo
    VAR_OBJETIVO = 'quantity_available' 

    # orden
    df[FECHA_PRINCIPAL] = df[FECHA_PRINCIPAL].fillna(method='ffill')
    df = df.sort_values(by=[ID_PRODUCTO, FECHA_PRINCIPAL])

    print(f"Dataset ordenado por '{ID_PRODUCTO}' y '{FECHA_PRINCIPAL}'.")
    print(f"Variable Objetivo para lags/medias: '{VAR_OBJETIVO}'")

    df.info()
    return FECHA_PRINCIPAL, ID_PRODUCTO, VAR_OBJETIVO, df


@app.cell
def _(mo):
    mo.md(r"""
    **Justificación:**
    - Cargamos los datos, convertimos las fechas y ordenamos por ´product_sku´ y created_at. Al ordenar, nos aseguramos de que todos los cálculos de tiempo (lags, medias) ocurran dentro del hilo de cada producto.
    - Con  ´.sort_values()´ aseguramos que los datos de cada producto sean una "línea de tiempo" individual. Sin esto, el lag_1 (valor de ayer) tomaría el valor de un producto totalmente diferente, y el modelo aprendería mal.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Feature Engineering: Variables Temporales

    Extraemos componentes de la fecha principal (`created_at`) para que
    el modelo pueda aprender patrones estacionales
    (ej. "los lunes se mueve más stock", "en verano baja la demanda").
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Explicación:** En esta parte creamos dia_de_la_semana, estación, trimestre...
    - Nuestro modelo de ML no entiende "Lunes" o "True". Al convertir todo a números (dia_de_la_semana=0, es_feriado=1), le damos un contexto como un calendario
    """)
    return


@app.cell
def _(FECHA_PRINCIPAL, df):

    def get_season(month):
        if month in [12, 1, 2]:
            return 'Invierno'
        elif month in [3, 4, 5]:
            return 'Primavera'
        elif month in [6, 7, 8]:
            return 'Verano'
        else:
            return 'Otoño'

    # variables nuevas extraídas de la fecha principal
    fecha = df[FECHA_PRINCIPAL]
    df['semana_del_anio'] = fecha.dt.isocalendar().week
    df['dia_del_mes'] = fecha.dt.day
    df['dia_de_la_semana'] = fecha.dt.dayofweek  # Lunes=0, Domingo=6
    df['es_fin_de_semana'] = (df['dia_de_la_semana'] >= 5).astype(int)
    df['trimestre'] = fecha.dt.quarter

    # creación de 'estacion' usando variable 'mes'
    df['estacion'] = df['mes'].apply(get_season)

    # conversión de variables booleanas existentes a enteros (1/0)
    # el modelo prefiere 1/0 que True/False.
    df['vacaciones_o_no'] = df['vacaciones_o_no'].astype(int)
    df['es_feriado'] = df['es_feriado'].astype(int)
    df['temporada_alta'] = df['temporada_alta'].astype(int)

    print("Variables temporales nuevas creadas (semana, dia, trimestre, estacion).")
    print("Variables booleanas existentes (vacaciones, feriado, temporada_alta) convertidas a 0/1.")

    print("\n--- df.info() después de procesar variables temporales ---")
    df.info()
    print("\n--- df.head().T después de procesar variables temporales ---")
    # mostramos las variables que acabamos de crear yconvertir
    print(df[['product_sku', 'created_at', 'mes', 'estacion', 'es_feriado', 'temporada_alta']].head().T)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Feature Engineering: Lags (Variables Sintéticas)

    Creamos variables "lag" (retrasadas) de nuestra `VAR_OBJETIVO`.
    Esto es **lo más importante** para predicción: le decimos al modelo
    cuál era el stock disponible "ayer", "la semana pasada" y "el mes pasado".
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Explicación:**
    - Creamos lag_1, lag_7, lag_30 usando quantity_available. Que son la MEMORIA del modelo
      1. lag_1: Le da al modelo memoria a corto plazo.
      2. lag_7: Le da memoria semanal (comparar este lunes con el lunes pasado).
      3. lag_30: Le da memoria mensual.
      4. El groupby(ID_PRODUCTO) garantiza que el lag_1 de la "Barra Cereal Choco" sea el de la "Barra Cereal Choco" de ayer, y no el de "Agua Mineral".
    """)
    return


@app.cell
def _(ID_PRODUCTO, VAR_OBJETIVO, df):

    # .groupby(ID_PRODUCTO) es crucial.

    df['lag_1'] = df.groupby(ID_PRODUCTO)[VAR_OBJETIVO].shift(1)
    df['lag_7'] = df.groupby(ID_PRODUCTO)[VAR_OBJETIVO].shift(7)
    df['lag_30'] = df.groupby(ID_PRODUCTO)[VAR_OBJETIVO].shift(30)

    print(f"Variables Lag (1, 7, 30) creadas para '{VAR_OBJETIVO}'.")

    # mostramos cómo se ven (los primeros serán NaN, es normal)
    df[['product_sku', 'created_at', VAR_OBJETIVO, 'lag_1', 'lag_7']].head()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Feature Engineering: Históricas y Estadísticas

    Calculamos estadísticas móviles para capturar la **tendencia** y
    la **volatilidad** del stock.
    -   **Medias Móviles:** Suavizan el ruido diario.
    -   **Media Exponencial (EWMA):** Da más peso a los datos recientes.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Justificación:**
    - Creamos media_movil_7d, media_movil_30d y media_movil_exponencial.
    - Dado que el stock diario puede ser "ruidoso" (subir y bajar mucho). Las medias móviles suavizan este ruido y le muestran al modelo la tendencia general.
    - Esto le da al modelo el CONTEXTO DE TENDENCIA Y VOLATILIDAD. El lag_1 le dice dónde estaba ayer, pero la media_movil_30d le dice si esa cifra es "normal" o si está muy por encima/debajo de la tendencia del mes. La std_movil_30d (desviación estándar) le dice qué tan estable o no es el stock de ese producto.
    """)
    return


@app.cell
def _(ID_PRODUCTO, VAR_OBJETIVO, df):
    # Usamos .transform() para que el resultado (la media)
    # se alinee con el índice original del dataframe.

    #  Variables solicitadas 
    df['media_movil_7d'] = df.groupby(ID_PRODUCTO)[VAR_OBJETIVO].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    df['media_movil_30d'] = df.groupby(ID_PRODUCTO)[VAR_OBJETIVO].transform(
        lambda x: x.rolling(30, min_periods=1).mean()
    )
    df['media_movil_exponencial'] = df.groupby(ID_PRODUCTO)[VAR_OBJETIVO].transform(
        lambda x: x.ewm(span=7, adjust=False).mean()
    )

    # ariable extra (necesaria para 'anomalia_stock')
    df['std_movil_30d'] = df.groupby(ID_PRODUCTO)[VAR_OBJETIVO].transform(
        lambda x: x.rolling(30, min_periods=1).std()
    )
    df['std_movil_30d'] = df['std_movil_30d'].fillna(0) 

    print("Medias móviles (7d, 30d, Exp) y Std (30d) creadas.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Feature Engineering: Variables Sintéticas

    Creamos variables de negocio combinando otras.
    Estas variables capturan conceptos más complejos.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Explicación:**
    - Creamos tendencia_stock (media corta vs. larga), anomalia_stock (qué tan raro es el valor de hoy), ratio_reservado_disponible (qué tan estresado está el stock) y ahora rotacion_estimada_30d (qué tan rápido se mueve).
    - Estas son "features de inteligencia de negocio". Son ATAJOS para el modelo. En lugar de que el modelo intente descubrir que restar la media de 7 días y la de 30 días es importante, se lo damos directamente (tendencia_stock). Le damos el "nivel de estrés" y el "nivel de velocidad" pre-calculados.
    """)
    return


@app.cell
def _(FECHA_PRINCIPAL, ID_PRODUCTO, VAR_OBJETIVO, df, np):
    # 'diff' calcula la diferencia con la fila anterior (lag_1)
    df['variacion_stock_diaria'] = df.groupby(ID_PRODUCTO)[VAR_OBJETIVO].transform('diff')

    # Tendencia: Compara la media de corto plazo vs la de largo plazo
    df['tendencia_stock'] = df['media_movil_7d'] - df['media_movil_30d']

    # Días desde el último pedido
    df['dias_desde_ultimo_pedido'] = (df[FECHA_PRINCIPAL] - df['last_order_date']).dt.days
    df['dias_desde_ultimo_pedido'] = df['dias_desde_ultimo_pedido'].fillna(9999)

    # Ratio de "estrés" del stock: Reservado vs Disponible
    df['ratio_reservado_disponible'] = np.where(
        df[VAR_OBJETIVO] > 0,          # Si quantity_available > 0
        df['quantity_reserved'] / df[VAR_OBJETIVO], # Calcula el ratio
        0                             # Si no, el ratio es 0
    )

    # Nivel de Anomalía (Z-score): Qué tan "raro" es el stock de hoy
    df['anomalia_stock'] = np.where(
        df['std_movil_30d'] > 0,      # Si la desviación es > 0
        (df[VAR_OBJETIVO] - df['media_movil_30d']) / df['std_movil_30d'], # Calcula Z-score
        0                             # Si no, no hay anomalía
    )

    # calculamos rotacion
    # Usamos 'average_daily_usage' para estimar las "ventas" mensuales
    df['rotacion_estimada_30d'] = np.where(
        df['media_movil_30d'] > 0, # Evitar división por cero
        (df['average_daily_usage'] * 30) / df['media_movil_30d'],
        0 # Si no hay stock promedio, no hay rotación
    )

    print("Variables sintéticas (variación, tendencia, ratios, anomalía) creadas.")
    print("Variable sintética 'rotacion_estimada_30d' CREADA.")

    print("\n--- df.info() después de crear variables sintéticas ---")
    df.info()
    print("\n--- df.head().T después de crear variables sintéticas ---")
    print(df.head().T)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Nota sobre `rotacion_promedio_mensual`


    No pudimos calcular la rotación *financiera* (que usa Costo de Ventas),
    pero sí calculamos la **rotación en unidades** (`rotacion_estimada_30d`)
    usando `average_daily_usage` y `media_movil_30d`.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Limpieza Final y Guardado

    Los cálculos de lags y medias móviles crean valores `NaN` (Nulos)
    al principio de la serie de cada producto (ej. los primeros 30 días).

    **Eliminamos estas filas incompletas** para que el modelo entrene
    solo con datos 100% completos.
    """)
    return


@app.cell
def _(df, mo, pd):
    # Ver cuántos nulos se crearon por los nuevos features
    nulos_antes = df.isnull().sum().sort_values(ascending=False)
    print("--- Nulos ANTES de limpiar (Top 10) ---")
    print(nulos_antes[nulos_antes > 0].head(10))

    # Eliminamos filas donde nuestros lags principales son nulos
    df_processed = df.dropna(
        subset=['lag_1', 'lag_7', 'lag_30', 'variacion_stock_diaria']
    )

    print("\n--- Nulos DESPUÉS de limpiar ---")
    print(f"Nulos restantes: {df_processed.isnull().sum().sum()}")
    print(f"\nFilas originales: {len(df):,}")
    print(f"Filas procesadas: {len(df_processed):,}")
    print(f"Filas eliminadas (por NaNs): {len(df) - len(df_processed):,}")

    # --- Codificación y Selección Final ---

    # Convertir 'estacion' (categórica) a números (One-Hot Encoding)
    df_processed = pd.get_dummies(df_processed, columns=['estacion'], drop_first=True)

    # Definir columnas a excluir.
    # ¡Fíjate que 'anio', 'mes', 'vacaciones_o_no', 'es_feriado',
    # 'temporada_alta' NO están en esta lista, por lo tanto se MANTIENEN!
    cols_a_excluir_final = [
        # IDs y Texto que no aportan valor numérico directo
        'id', 'product_id', 'supplier_id', 'created_by_id', 'record_sequence_number',
        'product_name', 'supplier_name', 'batch_number', 'warehouse_location', 
        'shelf_location', 'stock_status', 'categoria_producto', 'subcategoria_producto',

        # Columnas de fecha originales que ya fueron transformadas
        'created_at', 'last_order_date', 'last_stock_count_date', 'expiration_date',
        'last_updated_at'
    ]

    # Filtrar las columnas que realmente existen en df_processed
    cols_a_excluir_final = [col for col in cols_a_excluir_final if col in df_processed.columns]

    df_final = df_processed.drop(columns=cols_a_excluir_final, errors='ignore')

    # --- Guardado ---
    try:
        processed_path = "C:/Users/samil/Desktop/APRENDIZAJE AUTOMATICO/PRIMER INTERCICLO/Practica-2-Aprendizaje-Automactico/data/dataset_reentrenamiento_advanced.csv"
        df_final.to_csv(processed_path, index=False)
        print(f"\nÉxito Dataset procesado guardado en: {processed_path}")
        mo.md(f"**Dataset guardado en:** `{processed_path}`")
    except Exception as e:
        print(f"\nError al guardar el archivo: {e}")
    return (df_final,)


@app.cell
def _(df_final, mo):
    mo.md(r"""
    ## 7. Revisión del Dataset Final

    Este es el dataset final que usaremos para entrenar
    nuestro modelo predictivo.
    """)
    print(df_final.info())
    df_final.head()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Carga del Modelo Pre-entrenado y Configuración

    En esta sección:
    1. Cargamos el modelo GRU previamente entrenado (`best_model.keras`)
    2. Cargamos el escalador utilizado en el entrenamiento original
    3. Preparamos los nuevos datos con el mismo formato
    """)
    return


@app.cell
def _():
    # Librerías para Deep Learning
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.preprocessing import StandardScaler
    import joblib

    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {keras.__version__}")
    return joblib, keras


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Carga del Modelo y Escalador Original
    """)
    return


@app.cell
def _(joblib, keras, mo):
    # Rutas de los archivos guardados
    MODEL_PATH = "C:/Users/samil/Desktop/APRENDIZAJE AUTOMATICO/PRIMER INTERCICLO/Practica-2-Aprendizaje-Automactico/modelo/notebooks/best_model.keras"
    SCALER_X_PATH = "C:/Users/samil/Desktop/APRENDIZAJE AUTOMATICO/PRIMER INTERCICLO/Practica-2-Aprendizaje-Automactico/modelo/notebooks/scaler_X.joblib"
    SCALER_Y_PATH = "C:/Users/samil/Desktop/APRENDIZAJE AUTOMATICO/PRIMER INTERCICLO/Practica-2-Aprendizaje-Automactico/modelo/notebooks/scaler_y.joblib"    
    try:
        # Cargar el modelo pre-entrenado
        model = keras.models.load_model(MODEL_PATH)
        print("✓ Modelo cargado exitosamente")

        # Cargar los escaladores originales
        scaler_X = joblib.load(SCALER_X_PATH)
        scaler_y = joblib.load(SCALER_Y_PATH)
        print("✓ Escalador X cargado exitosamente")
        print("✓ Escalador y cargado exitosamente")

        # Mostrar resumen del modelo
        print("\n--- Resumen del Modelo ---")
        model.summary()

    except FileNotFoundError as e:
        print(f" Error: No se encontró el archivo - {e}")
        print("Verifica las rutas del modelo y los escaladores")
        mo.md("** ERROR:** Verifica que los archivos `best_model.keras`, `scaler_X.joblib` y `scaler_y.joblib` existan en las rutas especificadas.")

    return model, scaler_X, scaler_y


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. Preparación de Datos para Reentrenamiento

    Preparamos los datos exactamente como lo hicimos en el entrenamiento original:
    1. Separamos features (X) y target (y)
    2. Aplicamos el mismo escalador
    3. Creamos secuencias temporales para el modelo GRU
    """)
    return


@app.cell
def _(df_final):
    # Definir variable objetivo
    TARGET = 'quantity_available'

    # Separar features y target
    X_new = df_final.drop(columns=[TARGET, 'product_sku'], errors='ignore')
    y_new = df_final[TARGET].values

    print(f"Shape de X_new: {X_new.shape}")
    print(f"Shape de y_new: {y_new.shape}")
    print(f"\nColumnas de features ({len(X_new.columns)}):")
    print(X_new.columns.tolist())

    return X_new, y_new


@app.cell
def _(scaler_X):
    scaler_X.feature_names_in_

    return


@app.cell
def _(X_new):
    X_new.columns

    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 11. Escalado y Creación de Secuencias Temporales
    """)
    return


@app.cell
def _(X_new, np, scaler_X, scaler_y, y_new):

    # Columnas permitidas por el scaler original
    cols_allowed = scaler_X.feature_names_in_

    # Filtrar X_new para que solo incluya esas columnas
    X_new_filtered = X_new[cols_allowed]

    print("✓ Columnas alineadas con scaler_X")
    print(f"Columnas esperadas ({len(cols_allowed)}): {cols_allowed}")
    print(f"Columnas de X_new ahora: {X_new_filtered.columns.tolist()}")


    X_new_scaled = scaler_X.transform(X_new_filtered)

    print("\n✓ Features (X) escalados con scaler_X")
    print(f"Shape después de escalar X: {X_new_scaled.shape}")

    # Escalar el target (y) usando scaler_y
    y_new_scaled = scaler_y.transform(y_new.reshape(-1, 1)).ravel()

    print("✓ Target (y) escalado con scaler_y")
    print(f"Shape después de escalar y: {y_new_scaled.shape}")



    def create_sequences(X, y, time_steps=7):
        """
        Crea secuencias temporales para el modelo GRU.

        Args:
            X: Matriz de features escalados
            y: Vector target escalado
            time_steps: Tamaño de la ventana temporal

        Returns:
            X_seq: Secuencias 3D (samples, time_steps, features)
            y_seq: Targets alineados
        """
        X_seq, y_seq = [], []

        for i in range(len(X) - time_steps):
            X_seq.append(X[i:i + time_steps])
            y_seq.append(y[i + time_steps])

        return np.array(X_seq), np.array(y_seq)



    TIME_STEPS = 7  # igual que el modelo original

    X_seq_new, y_seq_new_scaled = create_sequences(X_new_scaled, y_new_scaled, TIME_STEPS)

    print("\n✓ Secuencias creadas correctamente")
    print(f"Shape de X_seq_new: {X_seq_new.shape}")
    print(f"Shape de y_seq_new_scaled: {y_seq_new_scaled.shape}")

    return X_seq_new, y_seq_new_scaled


@app.cell
def _(mo):
    mo.md(r"""
    ## 12. División Train/Validation para Reentrenamiento
    """)
    return


@app.cell
def _(X_seq_new, y_seq_new_scaled):
    from sklearn.model_selection import train_test_split

    # División 80/20 para train/validation
    X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(
        X_seq_new, y_seq_new_scaled, 
        test_size=0.2, 
        random_state=42,
        shuffle=False  # Importante: No mezclar datos temporales
    )

    print(" Datos divididos en Train/Validation")
    print(f"Train: {X_train_new.shape[0]} muestras")
    print(f"Validation: {X_val_new.shape[0]} muestras")

    return X_train_new, X_val_new, y_train_new, y_val_new


@app.cell
def _(mo):
    mo.md(r"""
    ## 13. Reentrenamiento del Modelo

    Reentrenamos el modelo con:
    - **Frozen Layers (opcional):** Podemos congelar las primeras capas para mantener el conocimiento previo
    - **Learning Rate bajo:** Para ajuste fino
    - **Early Stopping:** Para evitar overfitting
    """)
    return


@app.cell
def _(keras):
    # Configurar callbacks
    checkpoint_path = "C:/Users/samil/Desktop/APRENDIZAJE AUTOMATICO/PRIMER INTERCICLO/Practica-2-Aprendizaje-Automactico/modelo/best_model_retrained.keras"

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

    print(" Callbacks configurados")
    print(f"El mejor modelo se guardará en: {checkpoint_path}")

    return (callbacks,)


@app.cell
def _(X_train_new, X_val_new, callbacks, model, y_train_new, y_val_new):
    # Reentrenar el modelo
    print("\n Iniciando reentrenamiento...")

    history = model.fit(
        X_train_new, y_train_new,
        validation_data=(X_val_new, y_val_new),
        epochs=50,  # Ajusta según necesites
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    print("\n Reentrenamiento completado!")

    return (history,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 14. Evaluación del Modelo Reentrenado
    """)
    return


@app.cell
def _(X_val_new, model, np, scaler_y, y_val_new):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Predecir sobre el conjunto de validación (escalado)
    y_pred_new_scaled = model.predict(X_val_new)

    # Desescalar las predicciones y los valores reales
    y_pred_new = scaler_y.inverse_transform(y_pred_new_scaled).ravel()
    y_val_new_original = scaler_y.inverse_transform(y_val_new.reshape(-1, 1)).ravel()

    # Calcular métricas en escala original
    mae = mean_absolute_error(y_val_new_original, y_pred_new)
    mse = mean_squared_error(y_val_new_original, y_pred_new)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val_new_original, y_pred_new)

    print(" Métricas del Modelo Reentrenado (Escala Original):")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    return


@app.cell
def _(history, mo):
    import matplotlib.pyplot as plt

    mo.md(r"""
    ## 15. Visualización del Entrenamiento
    """)

    # Gráfico de pérdida
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Pérdida durante Reentrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Pérdida (Escala Log)')
    plt.xlabel('Época')
    plt.ylabel('Loss (log)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
