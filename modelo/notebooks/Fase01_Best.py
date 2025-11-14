import marimo

__generated_with = "0.17.7"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Fase 2: Feature Engineering (Objetivo: Stock Disponible)

    Notebook para la transformación de `dataset.csv` en un dataset
    listo para modelado, con el objetivo de predecir `quantity_available`.
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

    df = pd.read_csv("C:/Users/samil/Desktop/APRENDIZAJE AUTOMATICO/PRIMER INTERCICLO/Practica-2-Aprendizaje-Automactico/data/dataset.csv")

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
        processed_path = "C:/Users/samil/Desktop/APRENDIZAJE AUTOMATICO/PRIMER INTERCICLO/Practica-2-Aprendizaje-Automactico/data/dataset_processed_advanced.csv"
        df_final.to_csv(processed_path, index=False)
        print(f"\n¡Éxito! Dataset procesado guardado en: {processed_path}")
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
    ## Conclusiones de la Fase 1: Feature Engineering

        En esta fase, hemos transformado exitosamente el `dataset.csv` original en un
        dataset 100% optimizado para Machine Learning. El objetivo ya no es
        la exploración (EDA), sino la **preparación para la predicción**.

    **Puntos Principales:**

        **1. Dataset 100% Numérico y Limpio:**
        El resultado final (`dataset_processed_advanced.csv`) tiene **7,050 filas y 41 columnas**,
        sin valores nulos y en un formato puramente numérico. Está listo para ser
        consumido directamente por un modelo.

        **2. Creación de "Memoria" (Lags):**
        Añadimos `lag_1`, `lag_7` y `lag_30`. Estas son las variables más importantes,
        ya que le dan al modelo la "memoria" de dónde estaba el stock ayer, la semana
        pasada y el mes pasado.

        **3. Creación de "Contexto" (Medias Móviles y Estadísticas):**
        No solo le dijimos al modelo *dónde* estaba el stock (lags), sino que le dimos
        *contexto*. Variables como `media_movil_30d` (tendencia estable),
        `std_movil_30d` (volatilidad) y `anomalia_stock` (qué tan "raro" es el valor de hoy)
        le permiten al modelo tomar decisiones mucho más inteligentes.

        **4. Creación de "Inteligencia de Negocio" (Variables Sintéticas):**
        Creamos variables que capturan conceptos de negocio complejos en un solo número.
        Las más importantes son:
        - `rotacion_estimada_30d`: Le dice al modelo qué tan *rápido* se mueve este producto.
        - `ratio_reservado_disponible`: Mide el "estrés" o la presión sobre el stock disponible.
        - `tendencia_stock`: Indica si el stock está creciendo o disminuyendo (media corta vs. larga).

        **5. El Intercambio: "Calidad sobre Cantidad"**
        Decidimos conscientemente eliminar **450 filas** (el 6% de los datos).
        Este "costo" fue necesario para poder calcular los features de 30 días
        (como `lag_30` y `media_movil_30d`).

        **¿Por qué fue una buena decisión?** Sacrificamos el 6% de nuestros peores datos
        (el "arranque en frío" de cada producto, que no tenía historia) para hacer
        que el 94% restante de los datos sea más inteligente.
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
