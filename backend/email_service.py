"""
Servicio de envío de reportes por email
Miembro 3: Samantha Suquilanda
Empresa: UPS Tuti
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
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

        # Validar credenciales
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
            # 🔧 FIX: Asegurar que resumen esté en UTF-8
            if resumen and isinstance(resumen, bytes):
                resumen = resumen.decode('utf-8')
            elif resumen:
                resumen = str(resumen)  # Asegurar que sea string
            
            # Crear mensaje con MIMEMultipart
            mensaje = MIMEMultipart('alternative')
            mensaje['From'] = self.email_from
            mensaje['To'] = destinatario
            
            # 🔧 FIX: Usar Header con UTF-8 para el asunto
            asunto = f" Reporte de Stock UPS Tuti - {fecha}"
            mensaje['Subject'] = Header(asunto, 'utf-8')
            
            # Crear tabla HTML bonita
            html_predicciones = self._crear_tabla_html(predicciones)
            
            #  FIX: Construir HTML con meta charset UTF-8
            html = f"""
<!DOCTYPE html>
<html lang="es">
  <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
      body {{ 
        font-family: Arial, sans-serif; 
        margin: 0;
        padding: 0;
        background-color: #f4f4f4;
      }}
      .container {{
        max-width: 800px;
        margin: 20px auto;
        background: white;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
      }}
      .header {{ 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; 
        padding: 30px 20px; 
        text-align: center; 
      }}
      .header h1 {{
        margin: 0;
        font-size: 28px;
      }}
      .header p {{
        margin: 10px 0 0 0;
        font-size: 14px;
        opacity: 0.9;
      }}
      .content {{ 
        padding: 30px 20px; 
      }}
      .info-box {{
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        border-left: 4px solid #667eea;
      }}
      .resumen {{
        background: #fff3cd;
        padding: 20px;
        border-radius: 8px;
        margin: 20px 0;
        border-left: 4px solid #ffc107;
        white-space: pre-wrap;
        line-height: 1.6;
      }}
      table {{ 
        border-collapse: collapse; 
        width: 100%; 
        margin-top: 20px; 
      }}
      th {{ 
        background-color: #667eea; 
        color: white; 
        padding: 12px; 
        text-align: left;
        font-weight: 600;
      }}
      td {{ 
        border: 1px solid #ddd; 
        padding: 12px; 
      }}
      tr:nth-child(even) {{
        background-color: #f8f9fa;
      }}
      .critico {{ 
        background-color: #fee !important; 
        border-left: 4px solid #dc3545;
      }}
      .alerta {{
        background-color: #fff3cd !important;
        border-left: 4px solid #ffc107;
      }}
      .ok {{ 
        background-color: #d4edda !important; 
        border-left: 4px solid #28a745;
      }}
      .footer {{ 
        background: #f8f9fa;
        color: #666; 
        font-size: 12px; 
        padding: 20px;
        text-align: center;
        border-top: 1px solid #ddd;
      }}
      .footer a {{
        color: #667eea;
        text-decoration: none;
      }}
    </style>
  </head>
  <body>
    <div class="container">
      <div class="header">
        <h1> UPS Tuti - Reporte de Inventario</h1>
        <p>Tu aliado en nutrición inteligente</p>
      </div>
      
      <div class="content">
        <div class="info-box">
          <p style="margin: 0;"><strong> Fecha del reporte:</strong> {fecha}</p>
          <p style="margin: 5px 0 0 0;"><strong> Generado:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
        
        {f'<div class="resumen"><h3 style="margin-top: 0;"> Resumen Ejecutivo</h3><p style="margin: 0;">{resumen}</p></div>' if resumen else ''}
        
        <h3> Detalle de Productos</h3>
        {html_predicciones}
      </div>
      
      <div class="footer">
        <p style="margin: 5px 0;">Este es un email automático generado por el sistema de UPS Tuti.</p>
        <p style="margin: 5px 0;">
           <a href="mailto:ventas@upstuti.com">ventas@upstuti.com</a> | 
           +593 7 234 5678
        </p>
        <p style="margin: 5px 0;"> C. Vieja 12-301, Cuenca, Ecuador</p>
      </div>
    </div>
  </body>
</html>
            """
            
            #  FIX: Adjuntar HTML con charset UTF-8 explícito
            html_part = MIMEText(html, 'html', 'utf-8')
            mensaje.attach(html_part)
            
            # Enviar
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as servidor:
                servidor.starttls()
                servidor.login(self.email_from, self.email_password)
                #  FIX: send_message maneja UTF-8 automáticamente
                servidor.send_message(mensaje)
            
            print(f" Email enviado exitosamente a {destinatario}")
            return {
                "exito": True,
                "mensaje": f"Reporte enviado a {destinatario}",
                "fecha_envio": datetime.now().isoformat()
            }
        
        except Exception as e:
            print(f" Error al enviar email: {e}")
            import traceback
            traceback.print_exc()
            return {
                "exito": False,
                "error": str(e)
              }
          
    def enviar_reporte_con_pdf(
          self,
          destinatario: str,
        fecha: str,
        pdf_bytes: bytes,
        nombre_archivo: str,
        resumen: str = None
      ) -> dict:
        """
        Envía email con PDF adjunto
        
        Args:
            destinatario: Email del destinatario
            fecha: Fecha del reporte
            pdf_bytes: Contenido del PDF en bytes
            nombre_archivo: Nombre del archivo PDF
            resumen: Resumen del reporte (opcional)
        
        Returns:
            dict: Resultado del envío
        """
        try:
            from email.mime.base import MIMEBase
            from email import encoders
            
            mensaje = MIMEMultipart()
            mensaje['From'] = self.email_from
            mensaje['To'] = destinatario
            mensaje['Subject'] = f" Reporte de Stock UPS Tuti - {fecha}"
            
            # Cuerpo del email
            html = f"""
            <html>
              <head>
                <style>
                  body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                  .header {{ background-color: #1e3a8a; color: white; padding: 20px; text-align: center; }}
                  .content {{ padding: 20px; }}
                  .footer {{ background-color: #f3f4f6; padding: 15px; text-align: center; font-size: 12px; color: #666; }}
                </style>
              </head>
              <body>
                <div class="header">
                  <h1> UPS Tuti - Reporte de Inventario</h1>
                </div>
                <div class="content">
                  <p>Estimado cliente,</p>
                  <p>Adjunto encontrarás el <b>Reporte de Predicción de Stock</b> generado por nuestro sistema inteligente para la fecha <b>{fecha}</b>.</p>
                  
                  {f'<p><b>Resumen:</b><br>{resumen}</p>' if resumen else ''}
                  
                  <p>El PDF incluye:</p>
                  <ul>
                    <li>✅ Predicciones detalladas por producto</li>
                    <li>📊 Análisis ejecutivo generado por IA</li>
                    <li>🎯 Productos críticos priorizados</li>
                    <li>📈 Resumen estadístico completo</li>
                  </ul>
                  
                  <p>Si tienes alguna pregunta, no dudes en contactarnos.</p>
                  <p><b>Equipo UPS Tuti</b><br>
                  📧 soporte@upstuti.com<br>
                  📞 +593 7 234 5678</p>
                </div>
                <div class="footer">
                  <p>Sistema Inteligente de Inventarios UPS Tuti | Powered by ML + LLM</p>
                  <p>Este es un correo automático generado por IA - {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
                </div>
              </body>
            </html>
            """
            
            mensaje.attach(MIMEText(html, 'html'))
            
            # Adjuntar PDF
            adjunto = MIMEBase('application', 'pdf')
            adjunto.set_payload(pdf_bytes)
            encoders.encode_base64(adjunto)
            adjunto.add_header(
                'Content-Disposition',
                f'attachment; filename={nombre_archivo}'
            )
            mensaje.attach(adjunto)
            
            # Enviar
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_from, self.email_password)
                server.send_message(mensaje)
            
            return {
                "exito": True,
                "mensaje": f"✅ Reporte PDF enviado a {destinatario}",
                "destinatario": destinatario,
                "archivo": nombre_archivo
            }
        
        except Exception as e:
            return {
                "exito": False,
                "error": f"❌ Error al enviar email: {str(e)}"
            }


    def _crear_tabla_html(self, predicciones: list) -> str:
        """Crea tabla HTML con las predicciones"""
        if not predicciones:
            return "<p>No hay predicciones disponibles.</p>"
        
        filas = ""
        for pred in predicciones:
            # 🔧 FIX: Asegurar que todos los strings sean UTF-8
            nombre = str(pred.get('nombre', 'Desconocido'))
            stock = pred.get('stock_predicho', 0)
            estado = str(pred.get('estado', 'OK'))
            
            # Color según estado (más específico)
            if "CRÍTICO" in estado.upper() or "CRITICO" in estado.upper():
                clase_css = "critico"
            elif "ALERTA" in estado.upper() or "ADVERTENCIA" in estado.upper():
                clase_css = "alerta"
            else:
                clase_css = "ok"
            
            # 🔧 FIX: Usar .2f para decimales consistentes
            stock_formateado = f"{float(stock):.2f}" if stock else "0.00"
            
            filas += f"""
            <tr class="{clase_css}">
                <td>{nombre}</td>
                <td>{stock_formateado} unidades</td>
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
