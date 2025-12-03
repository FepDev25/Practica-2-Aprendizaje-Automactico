"""
Servicio de envío de reportes por email
Miembro 3: Samantha Suquilanda
Empresa: UPS Tuti
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from datetime import datetime

# Cargamos variables desde .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print(" Advertencia: python-dotenv no instalado. Instala con: pip install python-dotenv")

class EmailService:
    def __init__(self):
        """Configura credenciales de Gmail"""
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.email_from = os.getenv("EMAIL_FROM", "sistema@upstuti.com")
        self.email_password = os.getenv("EMAIL_PASSWORD")  # Gmail App Password

        #  Validar credenciales
        if not self.email_from or not self.email_password:
            raise ValueError(
                " ERROR: No se encontraron EMAIL_FROM y EMAIL_PASSWORD.\n"
                "Solución:\n"
                "1. Crea un archivo .env en la carpeta backend/\n"
                "2. Agrega estas líneas:\n"
                "   EMAIL_FROM=tu_email@gmail.com\n"
                "   EMAIL_PASSWORD=tu_contraseña_app"
            )
        
        print(f" Email configurado: {self.email_from}")
    
    def enviar_reporte_prediccion(
        self, 
        destinatario: str, 
        fecha: str, 
        predicciones: list,
        resumen: str = None
    ) -> dict:
        """
        Envía un reporte de predicción de stock por email
        
        Args:
            destinatario: Email del destinatario
            fecha: Fecha del reporte
            predicciones: Lista de predicciones (ej: [{"nombre": "...", "stock": 150}, ...])
            resumen: Resumen ejecutivo opcional
        
        Returns:
            Dict con resultado del envío
        """
        try:
            # Crear mensaje HTML bonito
            html_predicciones = self._crear_tabla_html(predicciones)
            
            mensaje = MIMEMultipart('alternative')
            mensaje['From'] = self.email_from
            mensaje['To'] = destinatario
            mensaje['Subject'] = f" Reporte de Stock UPS Tuti - {fecha}"
            
            # Cuerpo del email
            html = f"""
            <html>
              <head>
                <style>
                  body {{ font-family: Arial, sans-serif; }}
                  .header {{ background-color: #4CAF50; color: white; padding: 20px; text-align: center; }}
                  .content {{ padding: 20px; }}
                  table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                  th {{ background-color: #4CAF50; color: white; padding: 12px; text-align: left; }}
                  td {{ border: 1px solid #ddd; padding: 10px; }}
                  .critico {{ background-color: #ffcccc; }}
                  .ok {{ background-color: #ccffcc; }}
                  .footer {{ color: gray; font-size: 12px; margin-top: 30px; }}
                </style>
              </head>
              <body>
                <div class="header">
                  <h1> UPS Tuti - Reporte de Inventario</h1>
                  <p>Tu aliado en nutrición inteligente</p>
                </div>
                
                <div class="content">
                  <h2>Predicción de Stock - {fecha}</h2>
                  <p><strong>Generado:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                  
                  {f"<p><strong>Resumen Ejecutivo:</strong></p><p>{resumen}</p>" if resumen else ""}
                  
                  <h3>Detalle de Productos</h3>
                  {html_predicciones}
                  
                  <hr>
                  <p class="footer">
                    Este es un email automático generado por el sistema de UPS Tuti.<br>
                    Contacto: ventas@upstuti.com | +593 7 234 5678<br>
                    C. Vieja 12-301, Cuenca, Ecuador
                  </p>
                </div>
              </body>
            </html>
            """
            
            mensaje.attach(MIMEText(html, 'html'))
            
            # Enviar
            servidor = smtplib.SMTP(self.smtp_server, self.smtp_port)
            servidor.starttls()
            servidor.login(self.email_from, self.email_password)
            servidor.send_message(mensaje)
            servidor.quit()
            
            print(f" Email enviado exitosamente a {destinatario}")
            return {
                "exito": True,
                "mensaje": f"Reporte enviado a {destinatario}",
                "fecha_envio": datetime.now().isoformat()
            }
        
        except Exception as e:
            print(f" Error al enviar email: {e}")
            return {
                "exito": False,
                "error": str(e)
            }
    
    def _crear_tabla_html(self, predicciones: list) -> str:
        """Crea tabla HTML con las predicciones"""
        if not predicciones:
            return "<p>No hay predicciones disponibles.</p>"
        
        filas = ""
        for pred in predicciones:
            nombre = pred.get('nombre', 'Desconocido')
            stock = pred.get('stock_predicho', 0)
            estado = pred.get('estado', 'OK')
            
            # Color según estado
            clase_css = "critico" if "CRÍTICO" in estado or "ADVERTENCIA" in estado else "ok"
            
            filas += f"""
            <tr class="{clase_css}">
                <td>{nombre}</td>
                <td>{stock:.0f} unidades</td>
                <td>{estado}</td>
            </tr>
            """
        
        tabla = f"""
        <table>
            <thead>
                <tr>
                    <th>Producto</th>
                    <th>Stock Predicho</th>
                    <th>Estado</th>
                </tr>
            </thead>
            <tbody>
                {filas}
            </tbody>
        </table>
        """
        
        return tabla

# PRUEBA (CON DATOS DE EJEMPLO)


if __name__ == "__main__":
    print("=== PRUEBA DE ENVÍO DE EMAIL ===\n")
    
    # Datos de ejemplo
    predicciones_ejemplo = [
        {"nombre": "Barra Cereal Choco", "stock_predicho": 187, "estado": "ADECUADO"},
        {"nombre": "Galletas Chocolate Chip", "stock_predicho": 124, "estado": "CRÍTICO"},
        {"nombre": "Chips Verde Lima", "stock_predicho": 188, "estado": "ADECUADO"},
    ]
    
    resumen_ejemplo = """
    Atención: Se detectó 1 producto en estado CRÍTICO (Galletas Chocolate Chip).
    Recomendación: Generar orden de compra para Proveedor B antes del viernes.
    """
    
    servicio = EmailService()
    
    # IMPORTANTE: Cambia esto por TU email para probar
    email_prueba = "ssuquilanda200412@gmail.com"  #  CAMBIAR AQUÍ
    
    print(f"Enviando reporte de prueba a: {email_prueba}")
    print("(Asegúrate de haber configurado EMAIL_FROM y EMAIL_PASSWORD en las variables de entorno)\n")
    
    resultado = servicio.enviar_reporte_prediccion(
        destinatario=email_prueba,
        fecha="2025-12-10",
        predicciones=predicciones_ejemplo,
        resumen=resumen_ejemplo
    )
    
    if resultado['exito']:
        print(f" {resultado['mensaje']}")
    else:
        print(f" Error: {resultado['error']}")
        print("\n Recuerda configurar:")
        print("   export EMAIL_FROM='tu_email@gmail.com'")
        print("   export EMAIL_PASSWORD='tu_contraseña_app_gmail'")