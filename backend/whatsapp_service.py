"""
Servicio de notificaciones por WhatsApp Business API
Miembro 3: Samantha Suquilanda
Empresa: UPS Tuti

Funcionalidad:
- Enviar alertas de stock crítico por WhatsApp
- Enviar reportes semanales resumidos
- Notificar a clientes y equipo interno

IMPORTANTE: En modo desarrollo de Meta, solo funcionan plantillas aprobadas.
Para texto libre, necesitas pasar a producción.

Requisitos:
1. Cuenta Meta Business (https://business.facebook.com)
2. WhatsApp Business API activada
3. Variables en .env:
   - WHATSAPP_PHONE_ID
   - WHATSAPP_TOKEN
   - WHATSAPP_VERSION (opcional, default: v22.0)

Instalación:
pip install requests
"""

import requests
import os
from typing import Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class WhatsAppService:
    def __init__(self):
        """Inicializa el servicio de WhatsApp Business API"""
        self.phone_id = os.getenv("WHATSAPP_PHONE_ID")
        self.token = os.getenv("WHATSAPP_TOKEN")
        self.version = os.getenv("WHATSAPP_VERSION", "v22.0")
        
        # Validar credenciales
        if not self.phone_id or not self.token:
            raise ValueError(
                "❌ ERROR: No se encontraron WHATSAPP_PHONE_ID y WHATSAPP_TOKEN.\n"
                "Solución:\n"
                "1. Crea una cuenta en Meta Business (https://business.facebook.com)\n"
                "2. Activa WhatsApp Business API\n"
                "3. Agrega al archivo .env:\n"
                "   WHATSAPP_PHONE_ID=tu_phone_id\n"
                "   WHATSAPP_TOKEN=tu_token_permanente\n"
                "\nGuía completa: https://developers.facebook.com/docs/whatsapp/cloud-api/get-started"
            )
        
        self.api_url = f"https://graph.facebook.com/{self.version}/{self.phone_id}/messages"
        print(f"✅ WhatsApp Service configurado (Phone ID: {self.phone_id[:10]}...)")
    
    # ============================================
    # MÉTODO PRINCIPAL: USA PLANTILLA (FUNCIONA EN DESARROLLO)
    # ============================================
    
    def enviar_notificacion_hello_world(self, numero: str) -> Dict:
        """
        Envía mensaje usando plantilla 'hello_world' de Meta
        
        Esta plantilla viene pre-aprobada y funciona inmediatamente.
        Úsala para demostrar que WhatsApp funciona.
        
        Args:
            numero: Número de WhatsApp (ej: +593987654321)
        
        Returns:
            Dict con resultado del envío
        """
        try:
            numero_limpio = self._limpiar_numero(numero)
            
            payload = {
                "messaging_product": "whatsapp",
                "to": numero_limpio,
                "type": "template",
                "template": {
                    "name": "hello_world",
                    "language": {
                        "code": "en_US"
                    }
                }
            }
            
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Plantilla 'hello_world' enviada a {numero}")
                return {
                    "exito": True,
                    "mensaje": f"WhatsApp enviado a {numero}",
                    "whatsapp_message_id": data.get("messages", [{}])[0].get("id"),
                    "response": data
                }
            else:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", response.text)
                print(f"❌ Error API WhatsApp ({response.status_code}): {error_msg}")
                return {
                    "exito": False,
                    "error": f"API Error {response.status_code}: {error_msg}",
                    "response": error_data
                }
        
        except Exception as e:
            print(f"❌ Error al enviar plantilla: {e}")
            return {
                "exito": False,
                "error": str(e)
            }
    
    # ============================================
    # MÉTODOS CON TEXTO LIBRE (REQUIERE PRODUCCIÓN)
    # ============================================
    
    def enviar_alerta_stock_critico(
        self, 
        numero: str,
        producto: str,
        dias_restantes: float,
        stock_actual: int,
        sku: str = None
    ) -> Dict:
        """
        Envía alerta de stock crítico por WhatsApp con texto libre
        
        ⚠️ IMPORTANTE: Solo funciona en modo PRODUCCIÓN de Meta.
        En modo desarrollo, usa enviar_notificacion_hello_world()
        
        Args:
            numero: Número de WhatsApp con código país (ej: +593987654321 o 593987654321)
            producto: Nombre del producto
            dias_restantes: Días que durará el stock actual
            stock_actual: Unidades disponibles
            sku: Código SKU del producto (opcional)
        
        Returns:
            Dict con resultado del envío
        """
        try:
            numero_limpio = self._limpiar_numero(numero)
            
            # Generar mensaje formateado
            urgencia_emoji = "🚨" if dias_restantes < 7 else "⚠️"
            sku_texto = f"\n📋 SKU: {sku}" if sku else ""
            
            mensaje = f"""{urgencia_emoji} *ALERTA DE STOCK - UPS TUTI*

📦 Producto: *{producto}*{sku_texto}
⏱️ Días restantes: *{dias_restantes:.1f} días*
📊 Stock actual: *{stock_actual} unidades*

{self._generar_recomendacion(dias_restantes)}

_Mensaje automático del Sistema UPS Tuti_
_Contacto: +593 7 234 5678_
_Fecha: {datetime.now().strftime("%d/%m/%Y %H:%M")}_"""

            resultado = self._enviar_mensaje_texto(numero_limpio, mensaje)
            
            if resultado["exito"]:
                print(f"✅ Alerta crítica enviada a {numero} para producto '{producto}'")
            
            return resultado
        
        except Exception as e:
            print(f"❌ Error al enviar alerta crítica: {e}")
            return {
                "exito": False,
                "error": str(e)
            }
    
    def enviar_reporte_semanal(
        self,
        numero: str,
        productos_criticos: List[Dict],
        productos_ok: int,
        total_productos: int
    ) -> Dict:
        """
        Envía reporte semanal resumido de inventario
        
        ⚠️ Solo funciona en PRODUCCIÓN
        
        Args:
            numero: Número de WhatsApp del destinatario
            productos_criticos: Lista de productos críticos
            productos_ok: Cantidad de productos en buen estado
            total_productos: Total de productos en inventario
        """
        try:
            numero_limpio = self._limpiar_numero(numero)
            
            # Calcular porcentajes
            pct_ok = (productos_ok / total_productos * 100) if total_productos > 0 else 0
            pct_criticos = (len(productos_criticos) / total_productos * 100) if total_productos > 0 else 0
            
            # Generar lista de productos críticos (máximo 5)
            criticos_texto = ""
            for i, prod in enumerate(productos_criticos[:5], 1):
                emoji = "🔴" if prod.get("dias_restantes", 0) < 5 else "🟡"
                criticos_texto += f"{emoji} {prod['nombre']}: {prod.get('dias_restantes', 0):.1f} días ({prod.get('stock', 0)} unid.)\n"
            
            if len(productos_criticos) > 5:
                criticos_texto += f"... y {len(productos_criticos) - 5} productos más\n"
            
            # Mensaje completo
            mensaje = f"""📊 *REPORTE SEMANAL - UPS TUTI*
_Semana del {datetime.now().strftime("%d/%m/%Y")}_

*ESTADO GENERAL DEL INVENTARIO:*
📦 Total productos: {total_productos}
🟢 Productos OK: {productos_ok} ({pct_ok:.0f}%)
🔴 Productos críticos: {len(productos_criticos)} ({pct_criticos:.0f}%)

*PRODUCTOS QUE REQUIEREN ATENCIÓN:*
{criticos_texto if criticos_texto else "✅ Ninguno - Todo en orden"}

📈 *Acciones recomendadas:*
{self._generar_acciones_reporte(productos_criticos)}

🌐 Ver dashboard completo: http://34.10.83.87/

_Sistema Automatizado UPS Tuti_
_"Tu aliado en nutrición inteligente"_"""

            resultado = self._enviar_mensaje_texto(numero_limpio, mensaje)
            
            if resultado["exito"]:
                print(f"✅ Reporte semanal enviado a {numero}")
            
            return resultado
        
        except Exception as e:
            print(f"❌ Error al enviar reporte semanal: {e}")
            return {
                "exito": False,
                "error": str(e)
            }
    
    def enviar_prediccion_personalizada(
        self,
        numero: str,
        producto: str,
        prediccion: float,
        fecha_prediccion: str,
        nivel_minimo: int,
        recomendacion_ia: str = None
    ) -> Dict:
        """
        Envía predicción de stock personalizada (generada por la IA)
        
        ⚠️ Solo funciona en PRODUCCIÓN
        
        Args:
            numero: Número de WhatsApp
            producto: Nombre del producto
            prediccion: Stock predicho (unidades)
            fecha_prediccion: Fecha de la predicción
            nivel_minimo: Nivel mínimo de stock
            recomendacion_ia: Recomendación generada por el LLM (opcional)
        """
        try:
            numero_limpio = self._limpiar_numero(numero)
            
            # Determinar estado
            if prediccion < nivel_minimo:
                estado = "🔴 CRÍTICO"
            elif prediccion < nivel_minimo * 1.5:
                estado = "🟡 PRECAUCIÓN"
            else:
                estado = "🟢 ADECUADO"
            
            mensaje = f"""🔮 *PREDICCIÓN DE STOCK - UPS TUTI*

📦 *{producto}*
📅 Fecha: {fecha_prediccion}

*PREDICCIÓN:*
Stock estimado: *{prediccion:.0f} unidades*
Nivel mínimo: {nivel_minimo} unidades
Estado: {estado}

{f'💡 *Análisis IA:*\n{recomendacion_ia}\n' if recomendacion_ia else ''}
_Predicción generada por modelo GRU de UPS Tuti_
_Precisión promedio: 94%_"""

            resultado = self._enviar_mensaje_texto(numero_limpio, mensaje)
            
            if resultado["exito"]:
                print(f"✅ Predicción enviada a {numero} para '{producto}'")
            
            return resultado
        
        except Exception as e:
            print(f"❌ Error al enviar predicción: {e}")
            return {
                "exito": False,
                "error": str(e)
            }
    
    # ============================================
    # MÉTODOS INTERNOS
    # ============================================
    
    def _enviar_mensaje_texto(self, numero: str, mensaje: str) -> Dict:
        """Método interno para enviar mensaje de texto por WhatsApp API"""
        try:
            payload = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": numero,
                "type": "text",
                "text": {
                    "preview_url": False,
                    "body": mensaje
                }
            }
            
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "exito": True,
                    "mensaje": f"WhatsApp enviado a {numero}",
                    "whatsapp_message_id": data.get("messages", [{}])[0].get("id"),
                    "response": data
                }
            else:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", response.text)
                print(f"❌ Error API WhatsApp ({response.status_code}): {error_msg}")
                return {
                    "exito": False,
                    "error": f"API Error {response.status_code}: {error_msg}",
                    "response": error_data
                }
        
        except requests.exceptions.Timeout:
            return {
                "exito": False,
                "error": "Timeout: WhatsApp API no respondió a tiempo"
            }
        except requests.exceptions.RequestException as e:
            return {
                "exito": False,
                "error": f"Error de conexión: {str(e)}"
            }
    
    def _limpiar_numero(self, numero: str) -> str:
        """Limpia y valida formato de número de WhatsApp"""
        # Remover espacios, guiones, paréntesis
        numero_limpio = ''.join(filter(str.isdigit, numero.replace('+', '')))
        
        # Validar que tenga al menos 10 dígitos
        if len(numero_limpio) < 10:
            raise ValueError(f"Número inválido: {numero}. Debe incluir código de país (ej: 593987654321)")
        
        return numero_limpio
    
    def _generar_recomendacion(self, dias_restantes: float) -> str:
        """Genera recomendación según días restantes"""
        if dias_restantes < 3:
            return "⚡ *ACCIÓN URGENTE:* Generar orden de compra HOY. Stock crítico."
        elif dias_restantes < 7:
            return "⚠️ *ACCIÓN REQUERIDA:* Programar orden de compra esta semana."
        elif dias_restantes < 14:
            return "📋 *PRECAUCIÓN:* Considerar orden de compra próxima semana."
        else:
            return "✅ *Stock adecuado.* Monitorear evolución."
    
    def _generar_acciones_reporte(self, productos_criticos: List[Dict]) -> str:
        """Genera acciones recomendadas para el reporte semanal"""
        if not productos_criticos:
            return "✅ Mantener monitoreo rutinario. Sistema funcionando correctamente."
        
        muy_criticos = [p for p in productos_criticos if p.get("dias_restantes", 0) < 5]
        
        if muy_criticos:
            return f"🚨 URGENTE: {len(muy_criticos)} producto(s) requieren orden de compra inmediata.\n" \
                   f"📞 Contactar proveedores HOY."
        else:
            return f"⚠️ Programar órdenes de compra para {len(productos_criticos)} productos.\n" \
                   f"📅 Coordinar con proveedores esta semana."


# ============================================
# PRUEBAS Y EJEMPLOS
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 PRUEBA DEL SERVICIO DE WHATSAPP - UPS TUTI")
    print("=" * 60)
    print()
    
    try:
        servicio = WhatsAppService()
        print()
        
        # 🔴 CAMBIAR ESTOS NÚMEROS POR LOS DE PRUEBA
        NUMEROS_PRUEBA = [
            "+593939019136",  # Tu número
            "+593967056288",  # Número de compañero
        ]
        
        print("📱 Números de prueba configurados:")
        for num in NUMEROS_PRUEBA:
            print(f"   - {num}")
        print()
        
        print("Selecciona el tipo de prueba:")
        print("1️⃣  Plantilla Hello World (FUNCIONA en desarrollo)")
        print("2️⃣  Alerta de stock crítico (Solo producción)")
        print("3️⃣  Reporte semanal (Solo producción)")
        print("4️⃣  Predicción personalizada (Solo producción)")
        print("5️⃣  Enviar Hello World a TODOS los números")
        print()
        
        opcion = input("Opción (1-5): ").strip()
        print()
        
        if opcion == "1":
            numero = input(f"Número de destino (Enter para {NUMEROS_PRUEBA[0]}): ").strip()
            numero = numero if numero else NUMEROS_PRUEBA[0]
            
            print(f"📤 Enviando plantilla 'Hello World' a {numero}...")
            resultado = servicio.enviar_notificacion_hello_world(numero)
            
            if resultado["exito"]:
                print(f"✅ ÉXITO! Mensaje enviado")
                print(f"   Message ID: {resultado.get('whatsapp_message_id')}")
                print(f"\n📱 Revisa el WhatsApp {numero}")
            else:
                print(f"❌ ERROR: {resultado.get('error')}")
        
        elif opcion == "2":
            numero = input(f"Número de destino (Enter para {NUMEROS_PRUEBA[0]}): ").strip()
            numero = numero if numero else NUMEROS_PRUEBA[0]
            
            print(f"📤 Enviando alerta de stock crítico a {numero}...")
            print("⚠️ NOTA: Esta función solo funciona en modo PRODUCCIÓN de Meta")
            print()
            
            resultado = servicio.enviar_alerta_stock_critico(
                numero=numero,
                producto="Galletas Chocolate Chip",
                dias_restantes=3.5,
                stock_actual=70,
                sku="GCC-110"
            )
            
            if resultado["exito"]:
                print(f"✅ ÉXITO! Alerta enviada")
            else:
                print(f"❌ ERROR: {resultado.get('error')}")
                print("\n💡 Si dice que el mensaje se envió pero no llega,")
                print("   es porque estás en modo DESARROLLO (solo plantillas).")
        
        elif opcion == "3":
            numero = input(f"Número de destino (Enter para {NUMEROS_PRUEBA[0]}): ").strip()
            numero = numero if numero else NUMEROS_PRUEBA[0]
            
            print(f"📤 Enviando reporte semanal a {numero}...")
            print("⚠️ NOTA: Solo funciona en modo PRODUCCIÓN")
            print()
            
            criticos_ejemplo = [
                {"nombre": "Galletas Chocolate Chip", "dias_restantes": 4.2, "stock": 85},
                {"nombre": "Chips Verde Lima", "dias_restantes": 5.8, "stock": 116},
                {"nombre": "Barra Cereal Choco", "dias_restantes": 6.3, "stock": 127}
            ]
            resultado = servicio.enviar_reporte_semanal(
                numero=numero,
                productos_criticos=criticos_ejemplo,
                productos_ok=10,
                total_productos=13
            )
            
            if resultado["exito"]:
                print(f"✅ ÉXITO! Reporte enviado")
            else:
                print(f"❌ ERROR: {resultado.get('error')}")
        
        elif opcion == "4":
            numero = input(f"Número de destino (Enter para {NUMEROS_PRUEBA[0]}): ").strip()
            numero = numero if numero else NUMEROS_PRUEBA[0]
            
            print(f"📤 Enviando predicción personalizada a {numero}...")
            print("⚠️ NOTA: Solo funciona en modo PRODUCCIÓN")
            print()
            
            resultado = servicio.enviar_prediccion_personalizada(
                numero=numero,
                producto="Chips Sabor Queso",
                prediccion=185.5,
                fecha_prediccion="2025-12-15",
                nivel_minimo=150,
                recomendacion_ia="El stock predicho está 23% por encima del nivel mínimo. "
                                 "Situación favorable, pero monitorear tendencia de ventas."
            )
            
            if resultado["exito"]:
                print(f"✅ ÉXITO! Predicción enviada")
            else:
                print(f"❌ ERROR: {resultado.get('error')}")
        
        elif opcion == "5":
            print("📤 Enviando 'Hello World' a TODOS los números de prueba...")
            print()
            
            for numero in NUMEROS_PRUEBA:
                print(f"   Enviando a {numero}...")
                resultado = servicio.enviar_notificacion_hello_world(numero)
                
                if resultado["exito"]:
                    print(f"   ✅ Enviado")
                else:
                    print(f"   ❌ Error: {resultado.get('error')}")
                print()
            
            print("✅ Envío masivo completado")
            print("📱 Revisa los WhatsApp de todos los números")
        
        else:
            print("❌ Opción inválida")
        
        print()
        print("=" * 60)
        print(" Pruebas completadas")
        print("=" * 60)
        print()
        print("📝 NOTAS IMPORTANTES:")
        print("   ✅ Plantilla 'hello_world' - Funciona en DESARROLLO")
        print("   ⚠️ Texto libre (alertas, reportes) - Solo PRODUCCIÓN")
        print()
        print("   Para pasar a producción:")
        print("   1. Ve a Meta for Developers → App Review")
        print("   2. Solicita permiso: whatsapp_business_messaging")
        print("   3. Espera aprobación (2-3 días)")
        
    except Exception as e:
        print(f"❌ Error en las pruebas: {e}")
        print()
        print("💡 Verifica:")
        print("   1. Variables en .env (WHATSAPP_PHONE_ID, WHATSAPP_TOKEN)")
        print("   2. Números en la lista de prueba de Meta")
        print("   3. Token no expirado (dura 24h)")