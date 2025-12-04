# test_whatsapp_template.py
"""Prueba con plantilla aprobada de Meta"""

import requests
import os
from dotenv import load_dotenv

load_dotenv()

phone_id = os.getenv("WHATSAPP_PHONE_ID")
token = os.getenv("WHATSAPP_TOKEN")
version = os.getenv("WHATSAPP_VERSION", "v22.0")

api_url = f"https://graph.facebook.com/{version}/{phone_id}/messages"

# USAR PLANTILLA APROBADA (hello_world viene por defecto)
payload = {
    "messaging_product": "whatsapp",
    "to": "593967056288",  # TU NÚMERO
    "type": "template",
    "template": {
        "name": "hello_world",  # Plantilla que viene por defecto
        "language": {
            "code": "en_US"
        }
    }
}

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

print("📤 Enviando plantilla 'hello_world'...")
response = requests.post(api_url, json=payload, headers=headers)

print(f"Status: {response.status_code}")
print(f"Respuesta: {response.json()}")

if response.status_code == 200:
    print("\n✅ Revisa tu WhatsApp AHORA. Debe llegar un mensaje 'Hello World'")
else:
    print(f"\n❌ Error: {response.text}")