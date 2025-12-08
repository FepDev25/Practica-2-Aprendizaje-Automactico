"""
Servicio de Exportación de Reportes a PDF

"""

import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# PDF
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT


class ExportService:
    def __init__(self):
        """Inicializa servicio de exportación"""
        self.reportes_dir = Path(__file__).parent / "reportes"
        self.reportes_dir.mkdir(exist_ok=True)
        print(f" Export Service configurado (Reportes: {self.reportes_dir})")
    
    def generar_pdf_reporte(
        self,
        fecha: str,
        predicciones: List[Dict],
        mensaje_llm: str = None,
        tipo_reporte: str = "completo"  # "completo" o "individual"
    ) -> str:
        """
        Genera reporte PDF profesional con predicciones
        
        Args:
            fecha: Fecha de las predicciones
            predicciones: Lista de dict con {nombre, sku, prediccion, minimum_stock}
            mensaje_llm: Mensaje generado por el LLM
            tipo_reporte: "completo" (todos) o "individual" (un producto)
        
        Returns:
            str: Ruta del archivo PDF generado
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Reporte_Stock_{fecha.replace('-', '')}_{timestamp}.pdf"
        filepath = self.reportes_dir / filename
        
        # Crear documento PDF
        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=50
        )
        
        # Contenedor de elementos
        story = []
        styles = getSampleStyleSheet()
        
        # ===== ENCABEZADO =====
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1e3a8a'),
            spaceAfter=20,
            alignment=TA_CENTER
        )
        
        story.append(Paragraph("📊 REPORTE DE PREDICCIÓN DE STOCK", title_style))
        
        subtitle_style = ParagraphStyle(
            'Subtitle',
            parent=styles['Normal'],
            fontSize=12,
            textColor=colors.HexColor('#4b5563'),
            alignment=TA_CENTER,
            spaceAfter=30
        )
        story.append(Paragraph("<b>UPS Tuti</b> - Sistema Inteligente de Inventarios", subtitle_style))
        
        # ===== INFORMACIÓN GENERAL =====
        info_data = [
            [" Fecha de predicción:", fecha],
            [" Total de productos:", str(len(predicciones))],
            [" Generado:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["  Generado por:", "IA Gemini 2.0 + Modelo GRU"]
        ]
        
        info_table = Table(info_data, colWidths=[2.5*inch, 3.5*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e0e7ff')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        
        story.append(info_table)
        story.append(Spacer(1, 0.3*inch))
        
        # ===== ANÁLISIS LLM =====
        if mensaje_llm:
            story.append(Paragraph("<b>Análisis Ejecutivo:</b>", styles['Heading2']))
            story.append(Spacer(1, 0.15*inch))
            
            # Convertir mensaje markdown a HTML para PDF
            mensaje_html = self._markdown_to_html(mensaje_llm)
            
            for linea in mensaje_html.split('\n'):
                if linea.strip():
                    story.append(Paragraph(linea, styles['Normal']))
                    story.append(Spacer(1, 0.05*inch))
            
            story.append(Spacer(1, 0.3*inch))
        
        # ===== TABLA DE PREDICCIONES =====
        story.append(Paragraph("<b> Detalle de Predicciones:</b>", styles['Heading2']))
        story.append(Spacer(1, 0.15*inch))
        
        # Ordenar por stock predicho (menor a mayor)
        predicciones_sorted = sorted(predicciones, key=lambda x: x.get('prediccion', 0))
        
        # Crear tabla
        table_data = [
            ["#", "Producto", "SKU", "Predicción", "Mín. Stock", "Estado"]
        ]
        
        for idx, p in enumerate(predicciones_sorted, start=1):
            nombre = p.get('nombre', 'N/A')
            if len(nombre) > 35:
                nombre = nombre[:32] + "..."
            
            sku = p.get('sku', 'N/A')
            pred = p.get('prediccion', 0)
            min_stock = p.get('minimum_stock', 20)
            
            # Determinar estado
            if pred < min_stock:
                estado = "🔴 CRÍTICO"
            elif pred < min_stock * 1.5:
                estado = "🟡 ALERTA"
            else:
                estado = "🟢 OK"
            
            table_data.append([
                str(idx),
                nombre,
                sku,
                f"{pred:.0f}",
                f"{min_stock:.0f}",
                estado
            ])
        
        # Crear tabla con estilos
        pred_table = Table(
            table_data, 
            colWidths=[0.4*inch, 2*inch, 0.9*inch, 0.9*inch, 1*inch, 1.3*inch]
        )
        
        # Estilo de tabla
        table_style_list = [
            # Encabezado
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 0), (-1, 0), 12),
            
            # Contenido
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9fafb')]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        ]
        
        # Colorear filas críticas
        for i, p in enumerate(predicciones_sorted, start=1):
            if p.get('prediccion', 0) < p.get('minimum_stock', 20):
                table_style_list.append(
                    ('BACKGROUND', (0, i), (-1, i), colors.HexColor('#fee2e2'))
                )
        
        pred_table.setStyle(TableStyle(table_style_list))
        story.append(pred_table)
        
        # ===== RESUMEN ESTADÍSTICO =====
        if len(predicciones) > 1:
            story.append(Spacer(1, 0.4*inch))
            story.append(Paragraph("<b> Resumen Estadístico:</b>", styles['Heading2']))
            story.append(Spacer(1, 0.15*inch))
            
            predicciones_valores = [p.get('prediccion', 0) for p in predicciones]
            criticos = [p for p in predicciones if p.get('prediccion', 0) < p.get('minimum_stock', 20)]
            alertas = [p for p in predicciones if p.get('minimum_stock', 20) <= p.get('prediccion', 0) < p.get('minimum_stock', 20) * 1.5]
            ok = [p for p in predicciones if p.get('prediccion', 0) >= p.get('minimum_stock', 20) * 1.5]
            
            stats_data = [
                ["Stock promedio predicho:", f"{sum(predicciones_valores)/len(predicciones_valores):.1f} unidades"],
                ["Stock mínimo:", f"{min(predicciones_valores):.0f} unidades"],
                ["Stock máximo:", f"{max(predicciones_valores):.0f} unidades"],
                ["Productos CRÍTICOS:", f"{len(criticos)} ({len(criticos)/len(predicciones)*100:.1f}%)"],
                ["Productos en ALERTA:", f"{len(alertas)} ({len(alertas)/len(predicciones)*100:.1f}%)"],
                ["Productos OK:", f"{len(ok)} ({len(ok)/len(predicciones)*100:.1f}%)"]
            ]
            
            stats_table = Table(stats_data, colWidths=[3*inch, 3*inch])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f3f4f6')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                ('TOPPADDING', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey)
            ]))
            
            story.append(stats_table)
        
        # ===== PIE DE PÁGINA =====
        story.append(Spacer(1, 0.5*inch))
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER
        )
        story.append(Paragraph(
            "Sistema Inteligente de Inventarios UPS Tuti | ML + LLM (Gemini 2.0) | "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')}",
            footer_style
        ))
        
        # Construir PDF
        doc.build(story)
        print(f" PDF generado: {filepath}")
        
        return str(filepath)
    
    def generar_pdf_en_memoria(
        self,
        fecha: str,
        predicciones: List[Dict],
        mensaje_llm: str = None,
        tipo_reporte: str = "completo"
    ) -> bytes:
        """
        Genera PDF en memoria (sin guardar en disco) y devuelve bytes
        
        Returns:
            bytes: Contenido del PDF en bytes
        """
        from io import BytesIO
        
        # Crear buffer en memoria
        buffer = BytesIO()
        
        # Crear documento PDF en el buffer
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=50
        )
        
        # Contenedor de elementos (misma lógica que generar_pdf_reporte)
        story = []
        styles = getSampleStyleSheet()
        
        # ===== ENCABEZADO =====
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1e3a8a'),
            spaceAfter=20,
            alignment=TA_CENTER
        )
        
        story.append(Paragraph("📊 REPORTE DE PREDICCIÓN DE STOCK", title_style))
        
        subtitle_style = ParagraphStyle(
            'Subtitle',
            parent=styles['Normal'],
            fontSize=12,
            textColor=colors.HexColor('#4b5563'),
            alignment=TA_CENTER,
            spaceAfter=30
        )
        
        story.append(Paragraph(f"UPS Tuti - Distribución de Snacks Saludables", subtitle_style))
        story.append(Paragraph(f"Fecha de Análisis: {fecha}", subtitle_style))
        story.append(Spacer(1, 0.3*inch))
        
        # ===== MENSAJE LLM =====
        if mensaje_llm:
            llm_style = ParagraphStyle(
                'LLMStyle',
                parent=styles['Normal'],
                fontSize=10,
                textColor=colors.HexColor('#1f2937'),
                spaceAfter=20,
                leading=14,
                leftIndent=20,
                rightIndent=20
            )
            
            story.append(Paragraph("<b>📋 Resumen Ejecutivo:</b>", styles['Heading2']))
            story.append(Spacer(1, 0.1*inch))
            
            # Procesar mensaje LLM línea por línea
            for linea in mensaje_llm.split('\n'):
                if linea.strip():
                    story.append(Paragraph(linea.strip(), llm_style))
        
        story.append(Spacer(1, 0.3*inch))
    
        # ===== ESTADÍSTICAS RÁPIDAS =====
        criticos = [p for p in predicciones if p.get('stock_predicho', p.get('prediccion', 0)) < p.get('minimum_stock', 20)]
        alertas = [p for p in predicciones if p.get('minimum_stock', 20) <= p.get('stock_predicho', p.get('prediccion', 0)) < p.get('minimum_stock', 20) * 1.5]
        ok = len(predicciones) - len(criticos) - len(alertas)
        
        stats_data = [
            ['Estado', 'Cantidad', 'Porcentaje'],
            ['🔴 CRÍTICO', str(len(criticos)), f"{(len(criticos)/len(predicciones)*100):.1f}%"],
            ['🟡 ALERTA', str(len(alertas)), f"{(len(alertas)/len(predicciones)*100):.1f}%"],
            ['🟢 OK', str(ok), f"{(ok/len(predicciones)*100):.1f}%"],
        ]
        
        stats_table = Table(stats_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        ]))
        
        story.append(Paragraph("<b>📊 Resumen General:</b>", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))
        story.append(stats_table)
        story.append(Spacer(1, 0.3*inch))
        
        # ===== TABLA DE PREDICCIONES =====
        story.append(Paragraph("<b>📦 Detalle de Predicciones:</b>", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))
        
        # Preparar datos
        table_data = [['#', 'Producto', 'SKU', 'Stock Predicho', 'Stock Mínimo', 'Estado']]
        
        for idx, pred in enumerate(predicciones, 1):
            stock_pred = pred.get('stock_predicho', pred.get('prediccion', 0))
            min_stock = pred.get('minimum_stock', 20)
            
            if stock_pred < min_stock:
                estado = '🔴 CRÍTICO'
                color_fondo = colors.HexColor('#fee2e2')
            elif stock_pred < min_stock * 1.5:
                estado = '🟡 ALERTA'
                color_fondo = colors.HexColor('#fef3c7')
            else:
                estado = '🟢 OK'
                color_fondo = colors.HexColor('#d1fae5')
            
            nombre = pred.get('nombre', 'Desconocido')[:30]
            
            table_data.append([
                str(idx),
                nombre,
                pred.get('sku', 'N/A'),
                f"{stock_pred:.1f}",
                f"{min_stock:.1f}",
                estado
            ])
        
        # Crear tabla
        col_widths = [0.4*inch, 2.2*inch, 1*inch, 1.2*inch, 1.2*inch, 1*inch]
        predictions_table = Table(table_data, colWidths=col_widths, repeatRows=1)
        
        # Estilos base
        table_style = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]
        
        # Colorear filas según estado
        for idx, pred in enumerate(predicciones, 1):
            stock_pred = pred.get('stock_predicho', pred.get('prediccion', 0))
            min_stock = pred.get('minimum_stock', 20)
            
            if stock_pred < min_stock:
                bg_color = colors.HexColor('#fee2e2')
            elif stock_pred < min_stock * 1.5:
                bg_color = colors.HexColor('#fef3c7')
            else:
                bg_color = colors.HexColor('#d1fae5')
            
            table_style.append(('BACKGROUND', (0, idx), (-1, idx), bg_color))
        
        predictions_table.setStyle(TableStyle(table_style))
        story.append(predictions_table)
        
        # ===== PIE DE PÁGINA =====
        story.append(Spacer(1, 0.5*inch))
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER
        )
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        story.append(Paragraph(
            f"Generado automáticamente por UPS Tuti Stock Predictor | {timestamp}",
            footer_style
        ))
        
        # Construir PDF en el buffer
        doc.build(story)
        
        # Obtener bytes del buffer
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return pdf_bytes
    
    def _markdown_to_html(self, texto: str) -> str:
        """Convierte markdown básico a HTML para reportlab"""
        # Reemplazar ** por <b>
        texto = texto.replace('**', '<b>', 1)
        texto = texto.replace('**', '</b>', 1)
        
        # Reemplazar listas con viñetas
        lineas = []
        for linea in texto.split('\n'):
            if linea.strip().startswith('- '):
                linea = '  • ' + linea.strip()[2:]
            elif linea.strip().startswith('* '):
                linea = '  • ' + linea.strip()[2:]
            lineas.append(linea)
        
        return '\n'.join(lineas)
    
    def leer_pdf_como_bytes(self, filepath: str) -> bytes:
        """Lee el PDF generado como bytes para envío por email"""
        with open(filepath, 'rb') as f:
            return f.read()
    
    def limpiar_reportes_antiguos(self, dias: int = 7):
        """Elimina reportes PDF más antiguos que X días"""
        import time
        ahora = time.time()
        
        for archivo in self.reportes_dir.glob("*.pdf"):
            edad = ahora - archivo.stat().st_mtime
            if edad > dias * 86400:  # 86400 segundos = 1 día
                archivo.unlink()
                print(f" Reporte antiguo eliminado: {archivo.name}")


# Singleton para reutilizar instancia
_export_service = None

def get_export_service() -> ExportService:
    global _export_service
    if _export_service is None:
        _export_service = ExportService()
    return _export_service


# ============================================
# PRUEBAS
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print(" PRUEBAS EXPORT SERVICE - UPS Tuti")
    print("=" * 60)
    
    service = ExportService()
    
    # Datos de prueba
    predicciones_prueba = [
        {"nombre": "Chips Verde Lima", "sku": "CHV-113", "prediccion": 15, "minimum_stock": 50},
        {"nombre": "Barra Cereal Choco", "sku": "CRT-103", "prediccion": 80, "minimum_stock": 70},
        {"nombre": "Palomitas Caramelo", "sku": "PLC-104", "prediccion": 120, "minimum_stock": 70},
        {"nombre": "Snack Mix Saludable", "sku": "SMX-115", "prediccion": 35, "minimum_stock": 50}
    ]
    
    mensaje_llm = """**Análisis Ejecutivo:**

- **Situación General:** MODERADA - 2 de 4 productos requieren atención inmediata
- **Prioridad 1:** Reponer Chips Verde Lima (15 unidades, crítico)
- **Prioridad 2:** Monitorear Snack Mix Saludable (35 unidades, alerta)
- **Productos estables:** Barra Cereal Choco, Palomitas Caramelo
    """
    
    print("\n Generando PDF de prueba...")
    pdf_path = service.generar_pdf_reporte(
        fecha="2025-12-08",
        predicciones=predicciones_prueba,
        mensaje_llm=mensaje_llm,
        tipo_reporte="completo"
    )
    print(f"PDF generado exitosamente: {pdf_path}")
    print(f"\nAbre el archivo para verificar: {pdf_path}")
    
    print("\n Generando PDF en memoria (prueba)...")
    pdf_bytes = service.generar_pdf_en_memoria(
        fecha="2025-12-08",
        predicciones=predicciones_prueba,
        mensaje_llm=mensaje_llm,
        tipo_reporte="completo"
    )
    
    # Guardar PDF en disco para verificar (opcional)
    with open("reporte_prueba_memoria.pdf", "wb") as f:
        f.write(pdf_bytes)
        print("PDF en memoria guardado como 'reporte_prueba_memoria.pdf'")