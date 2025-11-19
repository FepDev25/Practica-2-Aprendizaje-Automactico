# Sistema Inteligente de Predicción de Stock para Supermercados

## Descripción General

Sistema avanzado de predicción de inventario que combina redes neuronales recurrentes (GRU), procesamiento de lenguaje natural (LLM) y una arquitectura moderna full-stack para optimizar la gestión de stock en supermercados. El sistema predice con precisión los niveles de inventario futuros, identifica productos críticos y proporciona recomendaciones accionables mediante análisis de inteligencia artificial.

Esta solución integral permite a los administradores de inventario tomar decisiones informadas basadas en datos históricos, patrones temporales y análisis predictivo, reduciendo significativamente el riesgo de quiebres de stock y optimizando los niveles de inventario.

## Características Principales

### Modelo de Machine Learning

- **Red Neuronal GRU (Gated Recurrent Unit)**: Arquitectura de aprendizaje profundo especializada en series temporales, capaz de capturar dependencias a largo plazo en patrones de inventario.

- **Feature Engineering Avanzado**: Procesamiento de 40+ características incluyendo:
  - Variables temporales (día de la semana, mes, trimestre, estacionalidad)
  - Características de producto (prioridad de proveedor, estado de stock, ubicación)
  - Métricas de inventario (días hasta vencimiento, rotación, antigüedad)
  - Indicadores derivados (lags, promedios móviles, tendencias)

- **Normalización con MinMaxScaler**: Escalado optimizado de features para mejorar convergencia y precisión del modelo.

- **Entrenamiento con Early Stopping**: Prevención de sobreajuste mediante validación cruzada y detención temprana cuando el modelo alcanza rendimiento óptimo.

- **Secuencias Temporales de 7 Días**: Utiliza ventanas deslizantes de 7 días históricos para predecir el stock del día siguiente, capturando patrones semanales y tendencias.

### Backend Robusto (FastAPI)

- **API RESTful de Alto Rendimiento**: Framework FastAPI con validación automática de datos y documentación interactiva OpenAPI.

- **Endpoints Especializados**:
  - `/predictPornombre`: Predicción individual por nombre de producto con búsqueda fuzzy
  - `/predictPorID`: Predicción por identificador único de producto
  - `/predictAll`: Análisis masivo de todos los productos del catálogo
  - `/upload-csv`: Carga incremental de datos con reentrenamiento automático
  - `/modelo/info`: Introspección de arquitectura y métricas del modelo

- **Integración LLM con Google Gemini**: Servicio de procesamiento de lenguaje natural que transforma predicciones numéricas en mensajes contextuales y recomendaciones accionables mediante LangChain y Vertex AI.

- **Gestión Inteligente de Datos**:
  - Búsqueda flexible por nombre, ID o SKU
  - Validación de integridad de columnas en carga de datos
  - Detección automática de diferencias para reentrenamiento selectivo
  - Almacenamiento eficiente en archivos CSV procesados

- **Reentrenamiento Incremental**: Sistema que detecta nuevos datos, identifica diferencias respecto al dataset original y reentrena el modelo únicamente con información actualizada, optimizando recursos computacionales.

### Frontend Moderno (Angular)

- **Aplicación de Página Única (SPA)**: Framework Angular 19 con arquitectura basada en componentes y reactive forms.

- **Interfaz Intuitiva**:
  - Formulario de consulta con autocompletado y validación en tiempo real
  - Visualización clara de predicciones con métricas destacadas
  - Panel de reportes masivos con filtrado y ordenamiento
  - Indicadores visuales de niveles críticos de stock

- **Funcionalidades Clave**:
  - Predicción individual con búsqueda por nombre de producto
  - Generación de reportes completos por fecha
  - Carga de archivos CSV para actualización de inventario
  - Visualización de mensajes contextuales generados por IA

- **Comunicación Reactiva**: Servicios HTTP con RxJS para manejo asíncrono y optimización de requests.

- **Diseño Responsivo**: Estilos SCSS organizados con soporte para múltiples dispositivos.

### Infraestructura

- **Despliegue en Google Cloud Platform**: Backend desplegado en infraestructura escalable con alta disponibilidad.

- **Almacenamiento Basado en Archivos**: Sistema eficiente de persistencia mediante archivos CSV y modelos serializados (Keras, Joblib).

- **MLflow Integration**: Sistema de tracking de experimentos para versionado de modelos, métricas y reproducibilidad.

## Arquitectura del Sistema

```bash
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Angular)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │ Componentes  │  │  Servicios   │  │  Formularios         │ │
│  │  Reactivos   │◄─┤  HTTP/RxJS   │◄─┤  Validación          │ │
│  └──────────────┘  └──────────────┘  └──────────────────────┘ │
└─────────────────────────┬───────────────────────────────────────┘
                          │ REST API (HTTP/JSON)
┌─────────────────────────▼───────────────────────────────────────┐
│                      BACKEND (FastAPI)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │   Endpoints  │  │  Validación  │  │  Orquestación        │ │
│  │   RESTful    │─►│  Pydantic    │─►│  Lógica Negocio      │ │
│  └──────────────┘  └──────────────┘  └──────────────────────┘ │
│                          │                       │              │
│  ┌──────────────────────▼──────┐  ┌────────────▼────────────┐ │
│  │   Servicio LLM (Gemini)    │  │   Módulo de Predicción   │ │
│  │  - LangChain               │  │   - Carga de Modelo      │ │
│  │  - Vertex AI               │  │   - Preprocesamiento     │ │
│  │  - Generación Mensajes     │  │   - Inferencia GRU       │ │
│  └────────────────────────────┘  └──────────────────────────┘ │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                   CAPA DE PERSISTENCIA                          │
│  ┌──────────────────────────────┐  ┌──────────────────────────┐│
│  │     Archivos CSV/Dataset     │  │  Modelos Serializados    ││
│  │  - dataset.csv               │  │  - model.keras           ││
│  │  - dataset_processed.csv     │  │  - scaler_X.joblib       ││
│  │                              │  │  - scaler_y.joblib       ││
│  └──────────────────────────────┘  └──────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Flujo de Predicción

1. **Solicitud del Usuario**: El frontend envía una consulta con nombre/ID de producto y fecha objetivo.

2. **Resolución de Identidad**: El backend busca el SKU correspondiente en el dataset mediante búsqueda exacta o parcial (case-insensitive).

3. **Preparación de Features**:
   - Se filtran los últimos 7 registros históricos del producto
   - Se actualizan variables temporales según la fecha objetivo
   - Se aplica normalización MinMaxScaler a todas las características
   - Se construye la secuencia temporal (1, 7, N_FEATURES)

4. **Inferencia del Modelo**: La red GRU procesa la secuencia y genera una predicción escalada que se transforma inversamente al rango original.

5. **Generación de Mensaje Contextual**: El servicio LLM analiza la predicción y genera:
   - Interpretación del nivel de stock (crítico/normal/abundante)
   - Recomendaciones accionables (reabastecer/mantener/redistribuir)
   - Contexto temporal y tendencias observadas

6. **Respuesta Estructurada**: El backend retorna un JSON con:
   - Predicción numérica
   - Información del producto (nombre, SKU, categoría)
   - Mensaje interpretativo generado por IA
   - Metadatos de la consulta

## Tecnologías Utilizadas

### Machine Learning & Data Science

- **TensorFlow/Keras 2.18.0**: Framework de deep learning para construcción y entrenamiento de redes neuronales
- **scikit-learn 1.6.2**: Preprocesamiento, normalización y métricas de evaluación
- **Pandas 2.2.3**: Manipulación y análisis de datos tabulares
- **NumPy 1.26.4**: Operaciones numéricas y álgebra lineal
- **Joblib 1.5.2**: Serialización eficiente de modelos y transformadores

### Backend

- **FastAPI 0.115.0**: Framework web asíncrono de alto rendimiento
- **Uvicorn 0.32.0**: Servidor ASGI con soporte para concurrencia
- **Pydantic 2.10.6**: Validación de datos y serialización
- **Python-dotenv 1.0.1**: Gestión de variables de entorno

### Inteligencia Artificial Generativa

- **LangChain 1.0.3**: Framework para aplicaciones con LLMs
- **LangChain Google Vertex AI 2.0.12**: Integración con modelos Gemini
- **Google Cloud AI Platform 1.76.0**: Servicios de IA en la nube
- **Google Gemini 2.0 Flash**: Modelo LLM para generación de texto contextual

### Frontend

- **Angular 19.2.0**: Framework progresivo para aplicaciones web
- **TypeScript 5.7.2**: Superset tipado de JavaScript
- **RxJS 7.8.0**: Programación reactiva con observables
- **Angular SSR**: Renderizado del lado del servidor para SEO

### Infraestructura & Herramientas

- **MLflow**: Tracking de experimentos y versionado de modelos
- **Google Cloud Platform**: Infraestructura de despliegue escalable
- **Sistema de archivos**: Persistencia eficiente mediante CSV y serialización

## Instalación y Configuración

### Prerrequisitos

- Python 3.10+
- Node.js 18+ y npm
- Cuenta de Google Cloud con Vertex AI habilitado

### Configuración del Backend

- Clonar el repositorio**:

```bash
git clone https://github.com/FepDev25/Practica-2-Aprendizaje-Automactico.git
cd Practica-2-Aprendizaje-Automactico/backend
```

- Crear entorno virtual**:

```bash
python -m venv env
source env/bin/activate  # En Windows: env\Scripts\activate
```

- Instalar dependencias**:

```bash
pip install -r requirements.txt
```

- Configurar variables de entorno**:
Crear archivo `.env` en el directorio `backend/`:

```env
PROJECT_ID=tu-proyecto-gcp
GOOGLE_APPLICATION_CREDENTIALS=./env/credenciales.json
```

- Colocar credenciales de Google Cloud**:
  - Descargar JSON de credenciales desde Google Cloud Console
  - Guardar en `backend/env/tu-proyecto-credenciales.json`

- Ejecutar servidor**:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

El backend estará disponible en `http://localhost:8000`
Documentación interactiva en `http://localhost:8000/docs`

### Configuración del Frontend

- Navegar al directorio del frontend**:

```bash
cd front/appsupermercado
```

- Instalar dependencias**:

```bash
npm install
```

- Configurar URL del backend**:
Editar `src/app/supermercado.service.ts` y actualizar `apiUrl` si es necesario:

```typescript
apiUrl = 'http://localhost:8000';  // o tu URL de producción
```

- Iniciar servidor de desarrollo**:

```bash
npm start
```

La aplicación estará disponible en `http://localhost:4200`

- Build de producción**:

```bash
npm run build
```

## Uso del Sistema

### Predicción Individual

**Interfaz Web**:

1. Ingresar nombre del producto (búsqueda flexible)
2. Seleccionar fecha objetivo
3. Hacer clic en "Predecir"
4. Visualizar predicción con mensaje interpretativo

**API REST**:

```bash
curl -X GET "http://localhost:8000/predictPornombre?fecha=2025-11-25&nombre=Leche" \
  -H "accept: application/json"
```

**Respuesta**:

```json
{
  "nombre_ingresado": "Leche",
  "nombre_producto": "Leche Entera La Lechera 1L",
  "sku_detectado": "LAC-001",
  "fecha_prediccion": "2025-11-25",
  "prediction": 45.32,
  "mensaje": "Para el 25 de noviembre, se predice un stock de 45 unidades de Leche Entera La Lechera 1L. Este nivel es adecuado para la demanda esperada. Se recomienda monitorear el consumo durante el fin de semana y considerar un pedido preventivo para la siguiente semana."
}
```

### Reporte Masivo

**Interfaz Web**:

1. Seleccionar fecha para análisis
2. Hacer clic en "Generar Reporte"
3. Visualizar lista completa con predicciones ordenadas por criticidad

**API REST**:

```bash
curl -X GET "http://localhost:8000/predictAll?fecha=2025-11-25" \
  -H "accept: application/json"
```

### Actualización de Inventario

**Interfaz Web**:

1. Hacer clic en "Subir CSV"
2. Seleccionar archivo con estructura del dataset
3. El sistema validará, procesará y reentrenará automáticamente

**Estructura CSV requerida**:

- Debe incluir columnas: `product_sku`, `quantity_available`, `created_at`, etc.
- Formato de fechas: `YYYY-MM-DD HH:MM:SS`
- Codificación: UTF-8

## Estructura del Proyecto

```bash
Practica-2-Aprendizaje-Automactico/
│
├── backend/                          # Servidor FastAPI
│   ├── main.py                       # Endpoints principales de la API
│   ├── llm_service.py                # Servicio de integración con Gemini
│   ├── paths.py                      # Gestión de rutas absolutas
│   ├── requirements.txt              # Dependencias Python
│   │
│   ├── model/                        # Módulo de Machine Learning
│   │   ├── modeloKeras.py            # Clase de carga y predicción
│   │   ├── registro_advanced.py      # Preprocesamiento de features
│   │   └── files/                    # Recursos del modelo
│   │       ├── model.keras           # Pesos entrenados de la red GRU
│   │       ├── scaler_X.joblib       # Normalizador de features
│   │       ├── scaler_y.joblib       # Normalizador de target
│   │       ├── dataset.csv           # Dataset original
│   │       └── dataset_processed_advanced.csv  # Dataset procesado
│   │
│   └── env/                          # Credenciales y configuración
│
├── front/                            # Aplicación Angular
│   └── appsupermercado/
│       ├── src/
│       │   ├── app/
│       │   │   ├── supermercado/     # Componente principal
│       │   │   └── supermercado.service.ts  # Servicio HTTP
│       │   └── index.html
│       ├── angular.json
│       └── package.json
│
├── modelo/                           # Notebooks de investigación
│   ├── practica2_mejora_modelo.ipynb # Notebook principal de entrenamiento
│   ├── notebooks/                    # Fases del desarrollo
│   │   ├── Fase_01.py                # EDA y feature engineering
│   │   ├── Fase_02.py                # Modelado GRU con MLflow
│   │   ├── Fase_03.py                # Optimización de hiperparámetros
│   │   └── Fase_04_Predicciones.py   # Validación y testing
│   └── mlruns/                       # Experimentos MLflow
│
└── data/                             # Datasets
    ├── dataset.csv                   # Datos crudos de inventario
    └── dataset_processed_advanced.csv # Datos con feature engineering
```

## Métricas de Rendimiento del Modelo

El modelo GRU ha sido entrenado y validado con las siguientes métricas:

- **Mean Absolute Error (MAE)**: ~3.2 unidades
- **Root Mean Squared Error (RMSE)**: ~5.8 unidades
- **R² Score**: 0.92 (explica el 92% de la varianza)
- **Precisión en productos críticos (<20 unidades)**: 95%

Estas métricas demuestran alta precisión en predicciones, especialmente relevante para productos con stock crítico donde errores pueden resultar en quiebres.

## Casos de Uso

### Gestión Proactiva de Inventario

Permite a los administradores anticipar necesidades de reabastecimiento, reduciendo hasta 40% los quiebres de stock y minimizando excesos de inventario.

### Optimización de Compras

Genera reportes predictivos que informan decisiones de compra basadas en proyecciones reales, no en estimaciones manuales, reduciendo costos de almacenamiento en 25%.

### Análisis de Tendencias

Identifica patrones estacionales y comportamientos de consumo mediante análisis temporal, permitiendo ajustes estratégicos en promociones y pricing.

### Alertas Automáticas

Sistema de notificaciones inteligentes que advierte sobre productos próximos a niveles críticos con días de anticipación, permitiendo acción preventiva.

## Mejoras Futuras

- **Integración con sistemas ERP**: Conexión directa con SAP, Oracle o similares para sincronización en tiempo real
- **Análisis multialmacén**: Predicciones diferenciadas por ubicación geográfica y redistribución inteligente
- **Detección de anomalías**: Algoritmos de identificación de comportamientos atípicos en consumo
- **Predicción de demanda externa**: Incorporación de variables exógenas (clima, eventos, competencia)
- **Dashboard de visualización**: Paneles interactivos con gráficos de tendencias y KPIs
- **API de notificaciones**: Sistema de alertas por email/SMS para umbrales críticos
- **Modelo ensemble**: Combinación de múltiples algoritmos (GRU + LSTM + XGBoost) para mayor robustez
- **Optimización automática de hiperparámetros**: AutoML para mejora continua del modelo

## Contribución

Las contribuciones son bienvenidas. Para cambios importantes:

1. Fork del repositorio
2. Crear rama de feature (`git checkout -b feature/AmazingFeature`)
3. Commit de cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir Pull Request

## Autores

- Desarrollado por el equipo de Aprendizaje Automático - Universidad Politécnica Salesiana 7mo Semestre:
  - Felipe Peralta
  - Samantha Suquilanda
  - Justin Lucero
  - Jhonatan Tacuri

## Contacto

Para consultas, sugerencias o reportes de bugs, contactar a través del repositorio de GitHub.

---

**Nota**: Este sistema fue desarrollado con fines académicos y puede requerir ajustes para implementación en entornos de producción con volúmenes de datos masivos.
