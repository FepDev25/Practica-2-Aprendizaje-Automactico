# test_whatsapp_debug.py
"""Script de diagnóstico para WhatsApp"""

import requests
import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("🔍 DIAGNÓSTICO WHATSAPP - UPS TUTI")
print("=" * 60)
print()

# 1. Verificar variables de entorno
print("1️⃣ Verificando variables de entorno...")
phone_id = os.getenv("WHATSAPP_PHONE_ID")
token = os.getenv("WHATSAPP_TOKEN")
version = os.getenv("WHATSAPP_VERSION", "v22.0")

print(f"   WHATSAPP_PHONE_ID: {phone_id}")
print(f"   WHATSAPP_TOKEN: {token[:50]}... (primeros 50 chars)")
print(f"   WHATSAPP_VERSION: {version}")
print()

if not phone_id or not token:
    print("❌ ERROR: Faltan credenciales en .env")
    exit(1)

# 2. Construir URL de la API
api_url = f"https://graph.facebook.com/{version}/{phone_id}/messages"
print(f"2️⃣ URL de la API: {api_url}")
print()

# 3. Preparar mensaje de prueba SIMPLE
numero_destino = "593967056288"  # 🔴 CAMBIAR POR TU NÚMERO

payload = {
    "messaging_product": "whatsapp",
    "to": numero_destino,
    "type": "text",
    "text": {
        "body": "Hola, este es un mensaje de prueba desde UPS Tuti 🚀"
    }
}

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

print(f"3️⃣ Enviando mensaje de prueba a: +{numero_destino}")
print()

# 4. Hacer la petición
try:
    response = requests.post(api_url, json=payload, headers=headers, timeout=10)
    
    print(f"4️⃣ Respuesta del servidor:")
    print(f"   Status Code: {response.status_code}")
    print()
    
    if response.status_code == 200:
        print("✅ ÉXITO! Mensaje enviado correctamente")
        print(f"   Respuesta: {response.json()}")
        print()
        print("📱 Revisa tu WhatsApp, debería llegar en unos segundos.")
    else:
        print("❌ ERROR en la API:")
        print(f"   Código: {response.status_code}")
        print(f"   Respuesta completa: {response.text}")
        print()
        
        # Interpretar errores comunes
        if response.status_code == 400:
            print("💡 Posibles causas (Error 400):")
            print("   - Phone ID incorrecto")
            print("   - Número de destino no está en la lista de prueba")
            print("   - Formato de número incorrecto")
        elif response.status_code == 401:
            print("💡 Posibles causas (Error 401):")
            print("   - Token expirado o inválido")
            print("   - Permisos insuficientes")
        elif response.status_code == 403:
            print("💡 Posibles causas (Error 403):")
            print("   - App no tiene acceso a WhatsApp API")
            print("   - Número de teléfono no verificado")

except Exception as e:
    print(f"❌ Excepción: {e}")

print()
print("=" * 60)