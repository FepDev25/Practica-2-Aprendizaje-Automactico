"""
Información de la empresa UPS Tuti
Autor: Samantha Suquilanda
"""

COMPANY_INFO = {
    "nombre": "UPS Tuti",
    "nicho": "Distribución mayorista de snacks saludables",
    "slogan": "Tu aliado en nutrición inteligente",
    
    # Historia
    "fundacion": "2020",
    "sede": "Cuenca, Ecuador",
    "mision": "Proporcionar snacks saludables de calidad a tiendas y supermercados, optimizando la gestión de inventario mediante inteligencia artificial para garantizar disponibilidad y frescura.",
    "vision": "Ser el distribuidor líder en Ecuador en gestión inteligente de inventarios, expandiéndonos a nivel regional para 2027.",
    
    # Productos (del dataset)
    "productos": {
        "barras_cereal": [
            {"sku": "CRT-103", "nombre": "Barra Cereal Choco", "categoria": "Barras de Cereal"},
            {"sku": "CPF-107", "nombre": "Barra Proteica Frutos", "categoria": "Barras de Cereal"},
            {"sku": "BCA-115", "nombre": "Barra Chocolate Avellana", "categoria": "Barras de Cereal"}
        ],
        "galletas": [
            {"sku": "GCC-110", "nombre": "Galletas Chocolate Chip", "categoria": "Galletas"},
            {"sku": "GIA-114", "nombre": "Galletas Integral Avena", "categoria": "Galletas"},
            {"sku": "GLD-102", "nombre": "Galletas Dulces Familia", "categoria": "Galletas"},
            {"sku": "GLS-106", "nombre": "Galletas Saladas Light", "categoria": "Galletas"}
        ],
        "chips": [
            {"sku": "CHQ-101", "nombre": "Chips Sabor Queso", "categoria": "Chips"},
            {"sku": "CHV-113", "nombre": "Chips Verde Lima", "categoria": "Chips"}
        ],
        "otros_snacks": [
            {"sku": "SMX-115", "nombre": "Snack Mix Saludable", "categoria": "Mix"},
            {"sku": "PLC-104", "nombre": "Palomitas Caramelo", "categoria": "Palomitas"},
            {"sku": "PLQ-108", "nombre": "Palomitas Queso", "categoria": "Palomitas"}
        ]
    },
    
    # Servicios
    "servicios": [
        "Predicción de stock disponible con IA (modelo GRU)",
        "Alertas automáticas de reabastecimiento",
        "Análisis ejecutivo de inventario con recomendaciones",
        "Reportes priorizados por criticidad de stock",
        "Dashboard en tiempo real",
        "API REST para integración con sistemas ERP"
    ],
    
    # Proveedores
    "proveedores": {
        "Proveedor A": {"prioridad": 1, "productos": ["Barras de Cereal", "Galletas"]},
        "Proveedor B": {"prioridad": 2, "productos": ["Chips", "Palomitas"]},
        "Proveedor C": {"prioridad": 3, "productos": ["Mix", "Snacks especiales"]}
    },
    
    # Operación
    "horario": {
        "atencion_cliente": "Lunes a Viernes: 8:00 AM - 6:00 PM | Sábados: 9:00 AM - 2:00 PM",
        "operaciones_almacen": "24/7 (sistema automático)",
        "soporte_tecnico": "Lunes a Domingo: 7:00 AM - 10:00 PM"
    },
    
    "contacto": {
        "email_ventas": "ventas@upstuti.com",
        "email_soporte": "soporte@upstuti.com",
        "telefono": "+593 7 234 5678",
        "whatsapp": "+593 98 765 4321",
        "direccion": "C. Vieja 12-301, Cuenca, Ecuador"
    },
    
    # Logros
    "hitos": [
        "2020: Fundación de UPS Tuti con 5 productos iniciales",
        "2021: Expansión a 15 productos y 3 proveedores",
        "2023: Implementación de modelo predictivo con Redes Neuronales GRU",
        "2024: Alianza con 50+ tiendas y supermercados en Ecuador",
        "2025: Lanzamiento de API REST para clientes corporativos"
    ],
    
    # Políticas y garantías
    "politicas": {
        "devoluciones": "Aceptamos devoluciones dentro de los primeros 7 días si el producto presenta defectos de fabricación.",
        "garantia_frescura": "Garantizamos que todos los productos tienen al menos 60 días de vida útil al momento de la entrega.",
        "entregas": "Entregas gratuitas para pedidos mayores a $500. Entregas en 24-48 horas en Cuenca y 3-5 días a nivel nacional.",
        "pagos": "Aceptamos transferencias bancarias, tarjetas de crédito/débito y pagos contra entrega (para clientes corporativos)."
    },
    
    # Ventajas competitivas
    "diferenciadores": [
        "Único distribuidor en Ecuador con sistema de IA para predicción de stock",
        "Plataforma web con dashboard en tiempo real para clientes",
        "Alertas automáticas de reabastecimiento por WhatsApp y email",
        "Productos 100% certificados y con garantía de frescura",
        "Soporte técnico 7 días a la semana"
    ]
}

# FAQs MEJORADAS (Es para la persona 2)

FAQS = [
    {
        "pregunta": "¿Qué es UPS Tuti?",
        "respuesta": f"UPS Tuti es un distribuidor mayorista de snacks saludables fundado en {COMPANY_INFO['fundacion']} en {COMPANY_INFO['sede']}. Nuestra misión es proporcionar snacks de calidad optimizando la gestión de inventario mediante inteligencia artificial."
    },
    {
        "pregunta": "¿Qué productos manejan?",
        "respuesta": "Gestionamos 15 productos de snacks saludables divididos en 4 categorías: Barras de Cereal (Choco, Proteica, Avellana), Galletas (Chocolate Chip, Integral Avena, Dulces Familia, Saladas Light), Chips (Queso, Verde Lima) y otros snacks como Palomitas (Caramelo, Queso) y Snack Mix Saludable."
    },
    {
        "pregunta": "¿Cómo funciona la predicción de stock con IA?",
        "respuesta": "Nuestro sistema usa un modelo de redes neuronales GRU (Gated Recurrent Unit) entrenado con datos históricos de ventas. El modelo analiza patrones temporales, estacionalidad, días festivos y tendencias para predecir cuánto stock necesitarás en los próximos días. Las predicciones se generan en menos de 2 segundos."
    },
    {
        "pregunta": "¿Cuál es el horario de atención?",
        "respuesta": f"Atención al cliente: {COMPANY_INFO['horario']['atencion_cliente']}. Soporte técnico: {COMPANY_INFO['horario']['soporte_tecnico']}. Nuestro sistema de predicción y alertas funciona {COMPANY_INFO['horario']['operaciones_almacen']}."
    },
    {
        "pregunta": "¿Hacen envíos a otras ciudades fuera de Cuenca?",
        "respuesta": f"Sí, realizamos envíos a nivel nacional. Entregas gratuitas para pedidos mayores a $500. Tiempos de entrega: 24-48 horas en Cuenca y 3-5 días al resto del país. Contáctanos al {COMPANY_INFO['contacto']['telefono']} para coordinar tu pedido."
    },
    {
        "pregunta": "¿Cómo puedo convertirme en cliente de UPS Tuti?",
        "respuesta": f"Es muy fácil: (1) Envía un email a {COMPANY_INFO['contacto']['email_ventas']} o llámanos al {COMPANY_INFO['contacto']['telefono']}, (2) Nuestro equipo te contactará para conocer tus necesidades, (3) Te configuramos una cuenta en nuestro dashboard, (4) ¡Listo! Empiezas a recibir predicciones automáticas de tus productos."
    },
    {
        "pregunta": "¿Qué tan confiable es el modelo de predicción?",
        "respuesta": "Nuestro modelo fue entrenado con más de 7,000 registros históricos y alcanza un R² de 0.966, lo que significa que explica el 96.6% de la variabilidad del stock. El error promedio es de apenas 7% (57 unidades sobre un inventario promedio de 800), lo cual es excelente para la industria de distribución."
    },
    {
        "pregunta": "¿Puedo recibir reportes automáticos por email?",
        "respuesta": "¡Sí! Nuestro sistema puede enviarte reportes automáticos de predicción de stock, alertas de productos críticos y análisis ejecutivo directamente a tu correo. También puedes configurar notificaciones por WhatsApp. Solo debes solicitarlo al configurar tu cuenta."
    },
    {
        "pregunta": "¿Qué pasa si un producto llega en mal estado?",
        "respuesta": f"Aplicamos nuestra {COMPANY_INFO['politicas']['garantia_frescura']} Además, {COMPANY_INFO['politicas']['devoluciones']} Contáctanos inmediatamente al {COMPANY_INFO['contacto']['email_soporte']} y resolveremos el problema."
    },
    {
        "pregunta": "¿Cuáles son los métodos de pago aceptados?",
        "respuesta": f"{COMPANY_INFO['politicas']['pagos']} Para clientes nuevos, solicitamos pago anticipado hasta establecer historial comercial."
    },
    {
        "pregunta": "¿Qué hace diferente a UPS Tuti de otros distribuidores?",
        "respuesta": f"Somos el único distribuidor en Ecuador con sistema de IA para predicción de stock. Además, ofrecemos: {', '.join(COMPANY_INFO['diferenciadores'][:3])}. Esto te permite reducir pérdidas por falta de stock o productos vencidos."
    },
    {
        "pregunta": "¿Cómo sé cuándo debo pedir más producto?",
        "respuesta": "¡Nuestro sistema lo hace por ti! Recibirás alertas automáticas por WhatsApp y email cuando el stock de un producto esté cerca del nivel mínimo. Además, el dashboard muestra en tiempo real qué productos necesitan reposición con colores (verde=OK, amarillo=precaución, rojo=crítico)."
    },
    {
        "pregunta": "¿Puedo ver el historial de mis pedidos?",
        "respuesta": "Sí, tu cuenta en el dashboard incluye un historial completo de pedidos, predicciones anteriores y un análisis de precisión del modelo para tus productos. Esto te ayuda a tomar mejores decisiones de compra."
    },
    {
        "pregunta": "¿Ofrecen capacitación sobre cómo usar el sistema?",
        "respuesta": "¡Por supuesto! Al registrarte como cliente, nuestro equipo te brinda una capacitación gratuita de 1 hora sobre cómo usar el dashboard, interpretar predicciones y configurar alertas. También tenemos tutoriales en video disponibles 24/7."
    },
    {
        "pregunta": "¿Trabajan con marcas específicas o tienen productos propios?",
        "respuesta": "Trabajamos con 3 proveedores certificados (Proveedor A, B y C) que nos suministran productos de marcas reconocidas en el mercado ecuatoriano. No tenemos marca propia, pero todos nuestros productos cumplen estándares de calidad y están certificados."
    }
]

# Utilidades (uso en otros módulos)

def obtener_productos_por_categoria(categoria: str):
    """Retorna lista de productos de una categoría específica"""
    return COMPANY_INFO["productos"].get(categoria, [])

def obtener_todos_los_skus():
    """Retorna lista de todos los SKUs disponibles"""
    skus = []
    for categoria in COMPANY_INFO["productos"].values():
        skus.extend([p["sku"] for p in categoria])
    return skus

def buscar_producto_por_sku(sku: str):
    """Busca información de un producto por su SKU"""
    for categoria in COMPANY_INFO["productos"].values():
        for producto in categoria:
            if producto["sku"] == sku:
                return producto
    return None