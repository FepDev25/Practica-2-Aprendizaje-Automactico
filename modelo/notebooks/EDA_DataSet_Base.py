import marimo

__generated_with = "0.17.7"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Fase 1: Análisis, Preparación y Feature Engineering

        Notebook para la preparación completa de los datos de `dataset.csv`,
        incluyendo las nuevas variables temporales, históricas y sintéticas.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Parte 1: Análisis Exploratorio de Datos (EDA) sobre el Dataset Base
    ## Dataset de Inventario y Gestión de Stock

    Un análisis exploratorio exhaustivo del dataset de gestión de inventario. El objetivo es comprender la estructura de los datos, identificar patrones, detectar anomalías y generar insights que permitan una mejor toma de decisiones en la gestión de stock.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Importación de Librerías

    Importamos las librerías necesarias para el análisis exploratorio, incluyendo herramientas para manipulación de datos, visualización y análisis estadístico.
    """)
    return


@app.cell
def _():
    # Manipulación y análisis de datos
    import pandas as pd
    import numpy as np

    # Visualización de datos
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Análisis estadístico
    from scipy import stats
    from scipy.stats import normaltest, skew, kurtosis

    # Configuración de visualización
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)

    # Configuración de warnings
    import warnings
    warnings.filterwarnings('ignore')

    print("Librerías importadas correctamente.")
    return kurtosis, np, pd, plt, skew, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Carga de Datos

    Cargamos el dataset desde el archivo CSV y realizamos una primera inspección de la estructura de los datos.
    """)
    return


@app.cell
def _(pd):
    # Cargar el dataset
    df = pd.read_csv("C:/Users/samil/Desktop/APRENDIZAJE AUTOMATICO/PRIMER INTERCICLO/Practica-2-Aprendizaje-Automactico/data/dataset.csv")

    # Información básica del dataset
    print("INFORMACIÓN GENERAL DEL DATASET")
    print(f"\nDimensiones del dataset: {df.shape[0]:,} filas x {df.shape[1]} columnas")
    print(f"Tamaño en memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print("\nPrimeras 5 filas del dataset:")
    print("=" * 80)
    df.head()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Análisis de la Estructura de Datos

    Examinamos los tipos de datos, valores nulos y características generales de cada columna.
    """)
    return


@app.cell
def _(df, pd):
    # Información detallada de las columnas
    print("INFORMACIÓN DE COLUMNAS")
    print("\nTipos de datos y valores no nulos:")
    print(df.info())

    print("\n" + "=" * 80)
    print("RESUMEN DE VALORES NULOS")
    print("=" * 80)
    null_summary = pd.DataFrame({
        'Columna': df.columns,
        'Valores Nulos': df.isnull().sum(),
        'Porcentaje (%)': (df.isnull().sum() / len(df) * 100).round(2)
    })
    null_summary = null_summary[null_summary['Valores Nulos'] > 0].sort_values('Valores Nulos', ascending=False)

    if len(null_summary) > 0:
        print(null_summary.to_string(index=False))
    else:
        print("No se encontraron valores nulos en el dataset.")
    return


@app.cell
def _(df, np):
    # Identificar tipos de variables
    numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    categoricas = df.select_dtypes(include=['object']).columns.tolist()

    print("CLASIFICACIÓN DE VARIABLES")
    print(f"\nVariables Numéricas ({len(numericas)}):")
    print(", ".join(numericas))
    print(f"\nVariables Categóricas ({len(categoricas)}):")
    print(", ".join(categoricas))
    return categoricas, numericas


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Estadísticas Descriptivas

    Análisis estadístico de las variables numéricas para comprender la distribución, tendencia central y dispersión de los datos.
    """)
    return


@app.cell
def _(df, kurtosis, numericas, skew):
    # Estadísticas descriptivas de variables numéricas
    print("ESTADÍSTICAS DESCRIPTIVAS - VARIABLES NUMÉRICAS")
    desc_stats = df[numericas].describe().T
    desc_stats['skewness'] = df[numericas].apply(lambda x: skew(x.dropna()))
    desc_stats['kurtosis'] = df[numericas].apply(lambda x: kurtosis(x.dropna()))
    print(desc_stats)
    return


@app.cell
def _(categoricas, df):
    # Análisis de variables categóricas
    print('ANÁLISIS DE VARIABLES CATEGÓRICAS')
    for _col in categoricas[:]:
        unique_count = df[_col].nunique()  # 5 primeras para no saturar
        print(f'\n{_col}:')
        print(f'  - Valores únicos: {unique_count}')
        if unique_count <= 10:
            print(f'  - Distribución:')
            print(df[_col].value_counts().to_string(header=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Detección de Valores Atípicos (Outliers)

    Identificamos valores atípicos utilizando el método de rango intercuartílico (IQR) para las variables numéricas más relevantes.
    """)
    return


@app.cell
def _(df, numericas, pd):
    # Función para detectar outliers usando IQR
    def detect_outliers_iqr(data, column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        return (len(outliers), lower_bound, upper_bound)
    print('DETECCIÓN DE VALORES ATÍPICOS (IQR)')
    outlier_summary = []
    for _col in numericas:
        if df[_col].nunique() > 10:
            n_outliers, lower, upper = detect_outliers_iqr(df, _col)
            if n_outliers > 0:  # Solo para variables con suficiente variación
                outlier_summary.append({'Variable': _col, 'N° Outliers': n_outliers, 'Porcentaje (%)': round(n_outliers / len(df) * 100, 2), 'Límite Inferior': round(lower, 2), 'Límite Superior': round(upper, 2)})
    if outlier_summary:
        outlier_df = pd.DataFrame(outlier_summary).sort_values('N° Outliers', ascending=False)
        print(outlier_df.to_string(index=False))
    else:
        print('No se detectaron outliers significativos en las variables numéricas.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Visualización de Distribuciones

    Análisis visual de las distribuciones de las principales variables numéricas del dataset.
    """)
    return


@app.cell
def _(df, numericas, plt):
    # Seleccionar variables clave de inventario para visualización
    inventory_vars = ['quantity_on_hand', 'quantity_reserved', 'quantity_available', 'minimum_stock_level', 'reorder_point', 'optimal_stock_level']
    inventory_vars = [var for var in inventory_vars if var in numericas]
    if len(inventory_vars) > 0:
    # Filtrar solo las que existen en el dataset
        _fig, _axes = plt.subplots(3, 2, figsize=(15, 12))
        _axes = _axes.flatten()
        for _idx, _col in enumerate(inventory_vars[:6]):
            _axes[_idx].hist(df[_col].dropna(), bins=50, edgecolor='black', alpha=0.7)
            _axes[_idx].set_title(f'Distribución de {_col}', fontsize=11, fontweight='bold')
            _axes[_idx].set_xlabel(_col)
            _axes[_idx].set_ylabel('Frecuencia')
            _axes[_idx].grid(True, alpha=0.3)
            mean_val = df[_col].mean()
            _axes[_idx].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Media: {mean_val:.2f}')
            _axes[_idx].legend()
        plt.tight_layout()
        plt.show()
    else:  # Añadir línea de media
        print('No se encontraron variables de inventario para visualizar.')
    return (inventory_vars,)


@app.cell
def _(df, inventory_vars, plt):
    # Boxplots para identificar outliers visualmente
    if len(inventory_vars) > 0:
        _fig, _axes = plt.subplots(2, 3, figsize=(16, 10))
        _axes = _axes.flatten()
        for _idx, _col in enumerate(inventory_vars[:6]):
            bp = _axes[_idx].boxplot(df[_col].dropna(), patch_artist=True)
            _axes[_idx].set_title(f'Boxplot: {_col}', fontsize=11, fontweight='bold')
            _axes[_idx].set_ylabel('Valor')
            _axes[_idx].grid(True, alpha=0.3)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
        plt.tight_layout()  # Colorear el boxplot
        plt.show()
    else:
        print('No se encontraron variables de inventario para visualizar.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Análisis de Correlaciones

    Evaluamos las relaciones lineales entre variables numéricas mediante la matriz de correlación.
    """)
    return


@app.cell
def _(df, np, numericas, pd, plt, sns):
    # Calcular matriz de correlación
    correlation_matrix = df[numericas].corr()
    plt.figure(figsize=(14, 10))
    # Visualizar matriz de correlación con heatmap
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
    plt.title('Matriz de Correlación - Variables Numéricas', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
    print('CORRELACIONES MÁS FUERTES (|r| > 0.7)')
    strong_corr = []
    for _i in range(len(correlation_matrix.columns)):
        for j in range(_i + 1, len(correlation_matrix.columns)):
    # Identificar correlaciones fuertes
            if abs(correlation_matrix.iloc[_i, j]) > 0.7:
                strong_corr.append({'Variable 1': correlation_matrix.columns[_i], 'Variable 2': correlation_matrix.columns[j], 'Correlación': round(correlation_matrix.iloc[_i, j], 3)})
    if strong_corr:
        strong_corr_df = pd.DataFrame(strong_corr).sort_values('Correlación', ascending=False, key=abs)
        print(strong_corr_df.to_string(index=False))
    else:
        print('No se encontraron correlaciones fuertes entre las variables.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Análisis Temporal

    Análisis de patrones temporales en el inventario, considerando tendencias y estacionalidad.
    """)
    return


@app.cell
def _(df, pd, plt):
    # Convertir columnas de fecha a datetime si existen
    date_columns = ['created_at', 'last_order_date', 'last_stock_count_date', 'expiration_date', 'last_updated_at']
    for _col in date_columns:
        if _col in df.columns:
            df[_col] = pd.to_datetime(df[_col], errors='coerce')
    if 'created_at' in df.columns:
        df_sorted = df.sort_values('created_at')
        _fig, _axes = plt.subplots(2, 2, figsize=(16, 10))
    # Verificar si existe columna de fecha principal
        if 'quantity_available' in df.columns:
            daily_avg = df_sorted.groupby(df_sorted['created_at'].dt.date)['quantity_available'].mean()
            _axes[0, 0].plot(daily_avg.index, daily_avg.values, linewidth=2)
            _axes[0, 0].set_title('Evolución del Stock Disponible (Media Diaria)', fontweight='bold')  # Análisis de tendencia temporal del inventario
            _axes[0, 0].set_xlabel('Fecha')
            _axes[0, 0].set_ylabel('Cantidad Disponible')
            _axes[0, 0].grid(True, alpha=0.3)  # Evolución del stock disponible
            _axes[0, 0].tick_params(axis='x', rotation=45)
        if 'quantity_reserved' in df.columns:
            daily_reserved = df_sorted.groupby(df_sorted['created_at'].dt.date)['quantity_reserved'].mean()
            _axes[0, 1].plot(daily_reserved.index, daily_reserved.values, color='orange', linewidth=2)
            _axes[0, 1].set_title('Evolución del Stock Reservado (Media Diaria)', fontweight='bold')
            _axes[0, 1].set_xlabel('Fecha')
            _axes[0, 1].set_ylabel('Cantidad Reservada')
            _axes[0, 1].grid(True, alpha=0.3)
            _axes[0, 1].tick_params(axis='x', rotation=45)
        if 'mes' in df.columns:  # Evolución del stock reservado
            monthly_stock = df.groupby('mes')['quantity_on_hand'].mean()
            _axes[1, 0].bar(monthly_stock.index, monthly_stock.values, color='steelblue')
            _axes[1, 0].set_title('Stock Promedio por Mes', fontweight='bold')
            _axes[1, 0].set_xlabel('Mes')
            _axes[1, 0].set_ylabel('Cantidad en Mano')
            _axes[1, 0].grid(True, alpha=0.3, axis='y')
        if 'total_value' in df.columns:
            daily_value = df_sorted.groupby(df_sorted['created_at'].dt.date)['total_value'].sum()
            _axes[1, 1].plot(daily_value.index, daily_value.values, color='green', linewidth=2)
            _axes[1, 1].set_title('Valor Total del Inventario en el Tiempo', fontweight='bold')  # Análisis por mes
            _axes[1, 1].set_xlabel('Fecha')
            _axes[1, 1].set_ylabel('Valor Total ($)')
            _axes[1, 1].grid(True, alpha=0.3)
            _axes[1, 1].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print('No se encontró columna de fecha para análisis temporal.')  # Valor total del inventario en el tiempo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Análisis Categórico

    Exploración de las distribuciones de variables categóricas clave y su relación con variables numéricas.
    """)
    return


@app.cell
def _(df, plt):
    # Análisis de categorías de producto
    if 'categoria_producto' in df.columns:
        _fig, _axes = plt.subplots(1, 2, figsize=(16, 6))
        cat_counts = df['categoria_producto'].value_counts().head(10)
        _axes[0].barh(cat_counts.index, cat_counts.values, color='teal')  # Distribución de categorías
        _axes[0].set_title('Top 10 Categorías de Productos', fontweight='bold', fontsize=12)
        _axes[0].set_xlabel('Cantidad de Registros')
        _axes[0].grid(True, alpha=0.3, axis='x')
        if 'quantity_available' in df.columns:
            cat_stock = df.groupby('categoria_producto')['quantity_available'].mean().sort_values(ascending=False).head(10)
            _axes[1].barh(cat_stock.index, cat_stock.values, color='coral')
            _axes[1].set_title('Stock Disponible Promedio por Categoría (Top 10)', fontweight='bold', fontsize=12)  # Stock promedio por categoría
            _axes[1].set_xlabel('Cantidad Promedio')
            _axes[1].grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.show()
    else:
        print("No se encontró la columna 'categoria_producto'.")
    return


@app.cell
def _(df, plt, sns):
    # Análisis del estado del stock
    if 'stock_status' in df.columns:
        _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))
        status_counts = df['stock_status'].value_counts()
        _axes[0].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set2'))  # Distribución del estado del stock
        _axes[0].set_title('Distribución del Estado del Stock', fontweight='bold', fontsize=12)
        if 'region_almacen' in df.columns and 'quantity_on_hand' in df.columns:
            region_stock = df.groupby('region_almacen')['quantity_on_hand'].sum().sort_values(ascending=False)
            _axes[1].bar(region_stock.index, region_stock.values, color='steelblue')
            _axes[1].set_title('Stock Total por Región de Almacén', fontweight='bold', fontsize=12)
            _axes[1].set_xlabel('Región')  # Stock por región de almacén
            _axes[1].set_ylabel('Cantidad Total')
            _axes[1].grid(True, alpha=0.3, axis='y')
            _axes[1].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("No se encontró la columna 'stock_status'.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Análisis de Estacionalidad y Factores Externos

    Evaluación del impacto de factores temporales como temporada alta, feriados y vacaciones en el comportamiento del inventario.
    """)
    return


@app.cell
def _(df, plt):
    # Análisis del impacto de factores externos
    seasonal_factors = ['temporada_alta', 'es_feriado', 'vacaciones_o_no']
    available_factors = [f for f in seasonal_factors if f in df.columns]
    if available_factors and 'quantity_available' in df.columns:
        n_factors = len(available_factors)
        _fig, _axes = plt.subplots(1, n_factors, figsize=(6 * n_factors, 5))
        if n_factors == 1:
            _axes = [_axes]
        for _idx, factor in enumerate(available_factors):
            factor_data = df.groupby(factor)['quantity_available'].mean().sort_index()
            labels = ['No', 'Sí'] if len(factor_data) == 2 else factor_data.index
            _axes[_idx].bar(range(len(factor_data)), factor_data.values, color=['lightcoral', 'lightgreen'][:len(factor_data)])
            _axes[_idx].set_title(f"Stock Disponible vs {factor.replace('_', ' ').title()}", fontweight='bold', fontsize=11)  # Convertir a formato legible
            _axes[_idx].set_xlabel(factor.replace('_', ' ').title())
            _axes[_idx].set_ylabel('Cantidad Disponible Promedio')
            _axes[_idx].set_xticks(range(len(factor_data)))  # Crear etiquetas legibles
            _axes[_idx].set_xticklabels(labels)
            _axes[_idx].grid(True, alpha=0.3, axis='y')
            for _i, _v in enumerate(factor_data.values):
                _axes[_idx].text(_i, _v, f'{_v:.1f}', ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        plt.show()
        print('ANÁLISIS ESTADÍSTICO - IMPACTO DE FACTORES EXTERNOS')
        for factor in available_factors:
            print(f"\n{factor.replace('_', ' ').title()}:")
            factor_stats = df.groupby(factor)['quantity_available'].agg(['mean', 'std', 'count'])
            print(factor_stats)
    else:
        print('No se encontraron factores estacionales o columna de cantidad disponible.')  # Añadir valores sobre las barras  # Análisis estadístico
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Análisis de Costos y Valor del Inventario

    Examinamos la estructura de costos y el valor total del inventario, identificando productos de alto valor.
    """)
    return


@app.cell
def _(df, plt):
    # Análisis de costos y valor
    if 'unit_cost' in df.columns and 'total_value' in df.columns:
        _fig, _axes = plt.subplots(2, 2, figsize=(16, 10))
        _axes[0, 0].hist(df['unit_cost'].dropna(), bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        _axes[0, 0].set_title('Distribución de Costos Unitarios', fontweight='bold')  # Distribución de costos unitarios
        _axes[0, 0].set_xlabel('Costo Unitario ($)')
        _axes[0, 0].set_ylabel('Frecuencia')
        _axes[0, 0].grid(True, alpha=0.3)
        _axes[0, 0].axvline(df['unit_cost'].median(), color='red', linestyle='--', linewidth=2, label=f"Mediana: ${df['unit_cost'].median():.2f}")
        _axes[0, 0].legend()
        _axes[0, 1].hist(df['total_value'].dropna(), bins=50, edgecolor='black', alpha=0.7, color='lightcoral')
        _axes[0, 1].set_title('Distribución del Valor Total del Inventario', fontweight='bold')
        _axes[0, 1].set_xlabel('Valor Total ($)')
        _axes[0, 1].set_ylabel('Frecuencia')
        _axes[0, 1].grid(True, alpha=0.3)  # Distribución de valor total
        _axes[0, 1].axvline(df['total_value'].median(), color='red', linestyle='--', linewidth=2, label=f"Mediana: ${df['total_value'].median():.2f}")
        _axes[0, 1].legend()
        if 'product_name' in df.columns:
            top_value = df.groupby('product_name')['total_value'].sum().sort_values(ascending=False).head(10)
            _axes[1, 0].barh(top_value.index, top_value.values, color='mediumseagreen')
            _axes[1, 0].set_title('Top 10 Productos por Valor Total', fontweight='bold')
            _axes[1, 0].set_xlabel('Valor Total ($)')
            _axes[1, 0].grid(True, alpha=0.3, axis='x')
        if 'quantity_available' in df.columns:
            _sample_data = df[['unit_cost', 'quantity_available']].dropna().sample(min(1000, len(df)))  # Top productos por valor total (si existe product_name)
            _axes[1, 1].scatter(_sample_data['unit_cost'], _sample_data['quantity_available'], alpha=0.5, s=30, color='purple')
            _axes[1, 1].set_title('Relación: Costo Unitario vs Cantidad Disponible', fontweight='bold')
            _axes[1, 1].set_xlabel('Costo Unitario ($)')
            _axes[1, 1].set_ylabel('Cantidad Disponible')
            _axes[1, 1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        print('RESUMEN FINANCIERO DEL INVENTARIO')  # Relación entre costo unitario y cantidad disponible
        print(f"\nValor Total del Inventario: ${df['total_value'].sum():,.2f}")
        print(f"Valor Promedio por Registro: ${df['total_value'].mean():,.2f}")
        print(f"Costo Unitario Promedio: ${df['unit_cost'].mean():.2f}")
        print(f"Costo Unitario Mediano: ${df['unit_cost'].median():.2f}")
    else:
        print('No se encontraron columnas de costo o valor para el análisis.')  # Resumen estadístico de valor
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12. Análisis de Niveles de Stock y Reabastecimiento

    Evaluación de los niveles de stock en relación con los puntos de reorden y niveles óptimos establecidos.
    """)
    return


@app.cell
def _(df, plt):
    # Análisis de niveles de stock
    stock_levels = ['quantity_on_hand', 'minimum_stock_level', 'reorder_point', 'optimal_stock_level']
    available_levels = [s for s in stock_levels if s in df.columns]
    if len(available_levels) >= 2:
        print('ANÁLISIS DE NIVELES DE STOCK')
        stock_comparison = df[available_levels].describe()  # Comparación de niveles de stock
        print('\nEstadísticas de niveles de stock:')
        print(stock_comparison)
        if 'quantity_on_hand' in df.columns and 'reorder_point' in df.columns:
            _below_reorder = df[df['quantity_on_hand'] < df['reorder_point']]
            print(f'\n\nProductos por debajo del punto de reorden: {len(_below_reorder)} ({len(_below_reorder) / len(df) * 100:.2f}%)')
            if len(_below_reorder) > 0 and 'product_name' in df.columns:
                print('\nTop 10 productos críticos (por debajo del punto de reorden):')  # Identificar productos por debajo del punto de reorden
                critical_products = _below_reorder.groupby('product_name')[['quantity_on_hand', 'reorder_point']].mean()
                critical_products['deficit'] = critical_products['reorder_point'] - critical_products['quantity_on_hand']
                print(critical_products.sort_values('deficit', ascending=False).head(10))
        _fig, _axes = plt.subplots(2, 2, figsize=(16, 10))
        if len(available_levels) >= 3:
            avg_levels = df[available_levels[:4]].mean()
            _axes[0, 0].bar(range(len(avg_levels)), avg_levels.values, color=['steelblue', 'orange', 'green', 'red'][:len(avg_levels)])
            _axes[0, 0].set_title('Comparación de Niveles de Stock Promedio', fontweight='bold')
            _axes[0, 0].set_ylabel('Cantidad')
            _axes[0, 0].set_xticks(range(len(avg_levels)))
            _axes[0, 0].set_xticklabels([l.replace('_', '\n') for l in avg_levels.index], fontsize=9)  # Visualización de niveles de stock
            _axes[0, 0].grid(True, alpha=0.3, axis='y')
            for _i, _v in enumerate(avg_levels.values):
                _axes[0, 0].text(_i, _v, f'{_v:.1f}', ha='center', va='bottom', fontweight='bold')  # Comparación de promedios
        if 'quantity_on_hand' in df.columns and 'optimal_stock_level' in df.columns:
            df['stock_gap'] = df['optimal_stock_level'] - df['quantity_on_hand']
            _axes[0, 1].hist(df['stock_gap'].dropna(), bins=50, edgecolor='black', alpha=0.7, color='coral')
            _axes[0, 1].set_title('Brecha entre Stock Actual y Óptimo', fontweight='bold')
            _axes[0, 1].set_xlabel('Brecha (Óptimo - Actual)')
            _axes[0, 1].set_ylabel('Frecuencia')
            _axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='Balance perfecto')
            _axes[0, 1].grid(True, alpha=0.3)
            _axes[0, 1].legend()
        if 'quantity_available' in df.columns and 'optimal_stock_level' in df.columns:  # Añadir valores
            df['utilization_rate'] = (df['quantity_available'] / df['optimal_stock_level'] * 100).clip(0, 200)
            _axes[1, 0].hist(df['utilization_rate'].dropna(), bins=50, edgecolor='black', alpha=0.7, color='lightgreen')
            _axes[1, 0].set_title('Tasa de Utilización del Stock (%)', fontweight='bold')
            _axes[1, 0].set_xlabel('Tasa de Utilización (%)')  # Distribución de la brecha entre stock actual y óptimo
            _axes[1, 0].set_ylabel('Frecuencia')
            _axes[1, 0].axvline(100, color='red', linestyle='--', linewidth=2, label='Utilización óptima (100%)')
            _axes[1, 0].grid(True, alpha=0.3)
            _axes[1, 0].legend()
        if 'quantity_reserved' in df.columns and 'quantity_available' in df.columns:
            _sample_data = df[['quantity_reserved', 'quantity_available']].dropna().sample(min(1000, len(df)))
            _axes[1, 1].scatter(_sample_data['quantity_reserved'], _sample_data['quantity_available'], alpha=0.5, s=30, color='darkblue')
            _axes[1, 1].set_title('Stock Reservado vs Disponible', fontweight='bold')
            _axes[1, 1].set_xlabel('Cantidad Reservada')
            _axes[1, 1].set_ylabel('Cantidad Disponible')
            _axes[1, 1].grid(True, alpha=0.3)  # Tasa de utilización del stock
        plt.tight_layout()
        plt.show()
    else:
        print('No se encontraron suficientes columnas de niveles de stock para el análisis.')  # Stock reservado vs disponible
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 13. Análisis de Proveedores

    Evaluación del desempeño y distribución de proveedores en el sistema de inventario.
    """)
    return


@app.cell
def _(df, plt):
    # Análisis de proveedores
    if 'supplier_name' in df.columns:
        _fig, _axes = plt.subplots(2, 2, figsize=(16, 10))
        supplier_counts = df['supplier_name'].value_counts().head(15)
        _axes[0, 0].barh(supplier_counts.index, supplier_counts.values, color='teal')  # Distribución de proveedores
        _axes[0, 0].set_title('Top 15 Proveedores por Cantidad de Registros', fontweight='bold')
        _axes[0, 0].set_xlabel('Cantidad de Registros')
        _axes[0, 0].grid(True, alpha=0.3, axis='x')
        if 'quantity_on_hand' in df.columns:
            supplier_stock = df.groupby('supplier_name')['quantity_on_hand'].mean().sort_values(ascending=False).head(15)
            _axes[0, 1].barh(supplier_stock.index, supplier_stock.values, color='coral')
            _axes[0, 1].set_title('Top 15 Proveedores por Stock Promedio', fontweight='bold')  # Stock promedio por proveedor
            _axes[0, 1].set_xlabel('Cantidad Promedio en Mano')
            _axes[0, 1].grid(True, alpha=0.3, axis='x')
        if 'prioridad_proveedor' in df.columns:
            priority_dist = df['prioridad_proveedor'].value_counts().sort_index()
            _axes[1, 0].bar(priority_dist.index, priority_dist.values, color='steelblue')
            _axes[1, 0].set_title('Distribución por Prioridad de Proveedor', fontweight='bold')
            _axes[1, 0].set_xlabel('Nivel de Prioridad')
            _axes[1, 0].set_ylabel('Cantidad de Registros')  # Prioridad de proveedores
            _axes[1, 0].grid(True, alpha=0.3, axis='y')
        if 'total_value' in df.columns:
            supplier_value = df.groupby('supplier_name')['total_value'].sum().sort_values(ascending=False).head(15)
            _axes[1, 1].barh(supplier_value.index, supplier_value.values, color='mediumseagreen')
            _axes[1, 1].set_title('Top 15 Proveedores por Valor Total del Inventario', fontweight='bold')
            _axes[1, 1].set_xlabel('Valor Total ($)')
            _axes[1, 1].grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.show()  # Valor total por proveedor
        print('RESUMEN DE PROVEEDORES')
        print(f"\nTotal de proveedores únicos: {df['supplier_name'].nunique()}")
        if 'prioridad_proveedor' in df.columns:
            print('\nDistribución por prioridad:')
            print(df['prioridad_proveedor'].value_counts().sort_index())
    else:
        print("No se encontró la columna 'supplier_name'.")  # Resumen de proveedores
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Análisis de Estabilidad y Secuencialidad del Dataset

        Visualizaciones para verificar que el dataset es adecuado para modelado de series temporales:
        - estabilidad (rolling mean/std),
        - dependencia temporal (autocorrelación),
        - realismo en los cambios (primera diferencia).
    """)
    return


@app.cell
def _(df, mo, pd, plt, sns):
    try:
        # Importamos SOLO la función específica localmente
        from pandas.plotting import autocorrelation_plot
    except ImportError:
        autocorrelation_plot = None

    mo.md(r"""
    ## Análisis de Estabilidad y Secuencialidad
    Verificamos si el dataset sirve para series temporales:
    """)

    # 1. Validaciones de seguridad
    # Usamos 'df' que viene de tu celda de carga
    mo.stop(
        df is None, 
        mo.md(" **Detenido:** El dataframe no se ha cargado correctamente.")
    )

    missing_cols = [c for c in ['created_at', 'quantity_available'] if c not in df.columns]
    mo.stop(
        len(missing_cols) > 0,
        mo.md(f" **Error:** Faltan columnas requeridas: {missing_cols}")
    )

    # 2. Preparación de datos (Usamos el 'pd' global de tu celda 3)
    df_time = df.copy()
    df_time['created_at'] = pd.to_datetime(df_time['created_at'], errors='coerce')
    df_time = df_time.dropna(subset=['created_at', 'quantity_available']).sort_values('created_at')

    mo.stop(
        df_time.empty,
        mo.md(" **Aviso:** No quedan datos válidos tras limpiar fechas.")
    )

    # 3. Resampleo Diario
    daily = df_time.set_index('created_at').resample('D')['quantity_available'].mean().dropna()

    mo.stop(
        daily.empty,
        mo.md(" **Aviso:** La serie diaria está vacía.")
    )

    # 4. Gráfico de Estabilidad
    rolling_window = 30
    roll_mean = daily.rolling(window=rolling_window, min_periods=7).mean()
    roll_std = daily.rolling(window=rolling_window, min_periods=7).std()

    # Usamos 'plt' global de tu celda 3
    fig_rolling, ax = plt.subplots(figsize=(14, 6))
    ax.plot(daily.index, daily.values, label='Media Diaria', color='C0', alpha=0.6)
    ax.plot(roll_mean.index, roll_mean.values, label=f'Rolling Mean ({rolling_window}d)', color='C1')
    ax.fill_between(roll_mean.index,
                    (roll_mean - roll_std).values,
                    (roll_mean + roll_std).values,
                    color='C1', alpha=0.2, label='Rolling ±1 std')
    ax.set_title('Estabilidad Temporal')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    # 5. Autocorrelación
    fig_ac = None
    if autocorrelation_plot is not None:
        try:
            fig_ac = plt.figure(figsize=(10, 4))
            autocorrelation_plot(daily)
            plt.title('Autocorrelación')
            plt.tight_layout()
        except Exception:
            fig_ac = None

    # 6. Primera Diferencia
    diffs = daily.diff().dropna()
    fig_diff = plt.figure(figsize=(12, 4))
    # Usamos 'sns' global de tu celda 3
    sns.histplot(diffs, bins=50, kde=True, color='tab:purple')
    plt.title('Distribución de Cambios Diarios')

    # 7. Salida visual
    mo.vstack([
        fig_rolling,
        fig_ac if fig_ac else mo.md(" *Gráfico de autocorrelación no disponible*"),
        fig_diff
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 14. Conclusiones y Hallazgos Principales

    Resumen ejecutivo de los principales insights obtenidos durante el análisis exploratorio.
    """)
    return


@app.cell
def _(categoricas, df, numericas):
    # Generar reporte de conclusiones
    print('RESUMEN EJECUTIVO - ANÁLISIS EXPLORATORIO DE DATOS')
    print('\n1. CARACTERÍSTICAS DEL DATASET')
    print(f'   - Total de registros: {df.shape[0]:,}')
    print(f'   - Total de variables: {df.shape[1]}')
    print(f'   - Variables numéricas: {len(numericas)}')
    print(f'   - Variables categóricas: {len(categoricas)}')
    if df.isnull().sum().sum() > 0:
        print(f'   - Registros con valores nulos: {df.isnull().any(axis=1).sum():,}')
    else:
        print('   - No se detectaron valores nulos en el dataset')
    print('\n2. ANÁLISIS DE INVENTARIO')
    if 'quantity_on_hand' in df.columns:
        print(f"   - Stock total en mano: {df['quantity_on_hand'].sum():,.0f} unidades")
        print(f"   - Stock promedio por registro: {df['quantity_on_hand'].mean():.2f} unidades")
    if 'quantity_available' in df.columns:
        print(f"   - Stock disponible total: {df['quantity_available'].sum():,.0f} unidades")
    if 'quantity_reserved' in df.columns:
        print(f"   - Stock reservado total: {df['quantity_reserved'].sum():,.0f} unidades")
    print('\n3. ANÁLISIS FINANCIERO')
    if 'total_value' in df.columns:
        print(f"   - Valor total del inventario: ${df['total_value'].sum():,.2f}")
        print(f"   - Valor promedio por registro: ${df['total_value'].mean():,.2f}")
    if 'unit_cost' in df.columns:
        print(f"   - Costo unitario promedio: ${df['unit_cost'].mean():.2f}")
        print(f"   - Rango de costos: ${df['unit_cost'].min():.2f} - ${df['unit_cost'].max():.2f}")
    print('\n4. DIVERSIDAD DE PRODUCTOS Y PROVEEDORES')
    if 'product_name' in df.columns:
        print(f"   - Productos únicos: {df['product_name'].nunique()}")
    if 'categoria_producto' in df.columns:
        print(f"   - Categorías de productos: {df['categoria_producto'].nunique()}")
        print(f"   - Categoría más frecuente: {df['categoria_producto'].mode()[0]}")
    if 'supplier_name' in df.columns:
        print(f"   - Proveedores únicos: {df['supplier_name'].nunique()}")
    print('\n5. ESTADO DEL STOCK')
    if 'stock_status' in df.columns:
        print('   - Distribución por estado:')
        for status, count in df['stock_status'].value_counts().items():
            print(f'     * {status}: {count:,} ({count / len(df) * 100:.1f}%)')
    if 'quantity_on_hand' in df.columns and 'reorder_point' in df.columns:
        _below_reorder = (df['quantity_on_hand'] < df['reorder_point']).sum()
        print(f'   - Productos por debajo del punto de reorden: {_below_reorder:,} ({_below_reorder / len(df) * 100:.1f}%)')
    print('\n6. FACTORES ESTACIONALES')
    if 'temporada_alta' in df.columns:
        high_season = (df['temporada_alta'] == True).sum()
        print(f'   - Registros en temporada alta: {high_season:,} ({high_season / len(df) * 100:.1f}%)')
    if 'es_feriado' in df.columns:
        holidays = (df['es_feriado'] == True).sum()
        print(f'   - Registros en días feriados: {holidays:,} ({holidays / len(df) * 100:.1f}%)')

    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
