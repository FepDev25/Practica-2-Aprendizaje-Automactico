"""
Módulo de detección de saludos y despedidas usando Regex
Miembro 3: Samantha Suquilanda
Empresa: UPS Tuti
"""
import re
import random

# PATRONES DE REGEX

GREETING_PATTERNS = [
    r'\b(hola|buenos días|buenas tardes|buenas noches|hey|hi|hello|qué tal)\b',
    r'\b(cómo estás|cómo va|saludos|buen día|buenas)\b'
]

FAREWELL_PATTERNS = [
    r'\b(adiós|adios|chao|hasta luego|nos vemos|bye|hasta pronto|me voy)\b',
    r'\b(gracias.*(?:adiós|chao|bye))\b'
]

THANKS_PATTERNS = [
    r'\b(gracias|te agradezco|muchas gracias|mil gracias|gracias totales)\b'
]


# RESPUESTAS (Personalizadas para UPS Tuti)

GREETING_RESPONSES = [
    "¡Hola! Soy el asistente virtual de UPS Tuti. ¿En qué puedo ayudarte con tu inventario de snacks hoy?",
    "¡Buenos días!  Estoy aquí para ayudarte con predicciones de stock, alertas y reportes. ¿Qué necesitas?",
    "¡Hola! Soy tu asistente de UPS Tuti. Puedo ayudarte con predicciones, días de stock restante y mucho más. ¿Qué consulta tienes?"
]

FAREWELL_RESPONSES = [
    "¡Hasta luego! Si necesitas más ayuda con tu inventario, aquí estaré 24/7.",
    "¡Adiós! Fue un placer ayudarte. ¡Que tengas excelentes ventas!",
    "¡Nos vemos! Recuerda revisar tus alertas de stock. ¡Hasta pronto!"
]

THANKS_RESPONSES = [
    "¡De nada! Es un placer ayudarte con UPS Tuti. ¿Necesitas algo más sobre tu inventario?",
    "¡Con gusto! Si tienes más preguntas sobre predicciones o stock, estoy aquí.",
    "¡Encantado de ayudarte! ¿Hay algo más en lo que pueda asistirte?"
]

# FUNCIONES DE DETECCIÓN

def es_saludo(texto: str) -> bool:
    """
    Detecta si el mensaje es un saludo
    
    Args:
        texto: Mensaje del usuario
    
    Returns:
        True si es saludo, False si no
    
    Ejemplo:
        >>> es_saludo("Hola, buenos días")
        True
        >>> es_saludo("Predice stock de leche")
        False
    """
    texto_lower = texto.lower()
    return any(re.search(pattern, texto_lower, re.IGNORECASE) for pattern in GREETING_PATTERNS)

def es_despedida(texto: str) -> bool:
    """Detecta si el mensaje es una despedida"""
    texto_lower = texto.lower()
    return any(re.search(pattern, texto_lower, re.IGNORECASE) for pattern in FAREWELL_PATTERNS)

def es_agradecimiento(texto: str) -> bool:
    """Detecta si el mensaje es un agradecimiento"""
    texto_lower = texto.lower()
    return any(re.search(pattern, texto_lower, re.IGNORECASE) for pattern in THANKS_PATTERNS)

def obtener_respuesta_saludo() -> str:
    """Retorna una respuesta de saludo aleatoria"""
    return random.choice(GREETING_RESPONSES)

def obtener_respuesta_despedida() -> str:
    """Retorna una respuesta de despedida aleatoria"""
    return random.choice(FAREWELL_RESPONSES)

def obtener_respuesta_agradecimiento() -> str:
    """Retorna una respuesta de agradecimiento aleatoria"""
    return random.choice(THANKS_RESPONSES)

# FUNCIÓN PRINCIPAL (Para el orquestador - Miembro 1)

def procesar_mensaje_simple(texto: str) -> dict:
    """
    Procesa un mensaje y retorna respuesta si es saludo/despedida/gracias
    
    Args:
        texto: Mensaje del usuario
    
    Returns:
        Dict con la respuesta o None si no es ninguno
    
    Ejemplo:
        >>> procesar_mensaje_simple("Hola!")
        {"tipo": "saludo", "respuesta": "¡Hola! Soy...", "requiere_llm": False}
    """
    if es_saludo(texto):
        return {
            "tipo": "saludo",
            "respuesta": obtener_respuesta_saludo(),
            "requiere_llm": False
        }
    
    if es_despedida(texto):
        return {
            "tipo": "despedida",
            "respuesta": obtener_respuesta_despedida(),
            "requiere_llm": False
        }
    
    if es_agradecimiento(texto):
        return {
            "tipo": "agradecimiento",
            "respuesta": obtener_respuesta_agradecimiento(),
            "requiere_llm": False
        }
    
    return None  # No es ninguno -> pasa al LLM (Miembro 1 lo manejará)

# PRUEBAS AUTOMÁTICAS


if __name__ == "__main__":
    print("===  PRUEBAS AUTOMÁTICAS DE REGEX - UPS TUTI ===\n")
    
    casos_prueba = [
        ("Hola", "saludo"),
        ("Buenos días, ¿cómo estás?", "saludo"),
        ("Adiós", "despedida"),
        ("Gracias por todo, chao", "despedida"),
        ("Muchas gracias", "agradecimiento"),
        ("Predice el stock de Galletas Chocolate", None),  # No debe matchear
        ("¿Cuántos días me quedan de stock?", None),
        ("Hey, qué tal", "saludo"),
        ("Hasta luego, nos vemos", "despedida"),
    ]
    
    print("Casos de Prueba:")
    print("-" * 70)
    
    for caso, tipo_esperado in casos_prueba:
        resultado = procesar_mensaje_simple(caso)
        
        if resultado:
            tipo_detectado = resultado['tipo']
            emoji = "bien" if tipo_detectado == tipo_esperado else "mal"
            print(f"{emoji} '{caso}'")
            print(f"   Tipo: {tipo_detectado}")
            print(f"   Respuesta: {resultado['respuesta'][:60]}...\n")
        else:
            emoji = "bien" if tipo_esperado is None else "mal"
            print(f"{emoji} '{caso}'")
            print(f"   No es saludo/despedida/gracias → Pasa al LLM\n")
    
    print("\n ¡Pruebas completadas! Este módulo está listo para integrarse.")