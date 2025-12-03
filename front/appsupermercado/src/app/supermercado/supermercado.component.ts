import { Component, OnInit } from '@angular/core';
import { SupermercadoService } from '../supermercado.service';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MarkdownModule } from 'ngx-markdown';
import { ChatComponent } from '../chat/chat.component';

@Component({
  selector: 'app-supermercado',
  imports: [CommonModule, FormsModule, MarkdownModule, ChatComponent],
  templateUrl: './supermercado.component.html',
  styleUrls: ['./supermercado.component.scss']
})
export class SupermercadoComponent implements OnInit {
  chatAbierto: boolean = false;
  mensajesSinLeer: number = 0;

  prediccionResultado: any = null;
  fecha: string = '';
  nombre: string = '';
  cargandoPrediccion: boolean = false;

  fechaReporte: string = '';
  reporte: any[] = [];
  cargandoReporte: boolean = false;

  mostrarAnalisis: boolean = false;
  mensajeResumen: string = '';

  constructor(private service: SupermercadoService) { }

  ngOnInit(): void {
    const hoy = new Date();
    this.fecha = this.formatDate(hoy);
    this.fechaReporte = this.formatDate(hoy);

    setTimeout(() => {
      this.mensajesSinLeer = 1;
    }, 2000);
  }

  
  toggleChat(): void {
    this.chatAbierto = !this.chatAbierto;
    
    if (this.chatAbierto) {
      this.mensajesSinLeer = 0;
    }

    console.log('Chat abierto:', this.chatAbierto);
  }

  recibirNotificacionChat(): void {
    if (!this.chatAbierto) {
      this.mensajesSinLeer++;
    }
  }
  
  getPedict(fecha: string, nombre: string): void {
    if (!fecha || !nombre) {
      alert('Debe ingresar el producto y la fecha');
      return;
    }

    this.cargandoPrediccion = true;
    this.prediccionResultado = null;
    
    this.service.getPedriccion(fecha, nombre).subscribe({
      next: (data) => {
        console.log(' Predicción recibida:', data);
        this.prediccionResultado = data;
        this.cargandoPrediccion = false;

        setTimeout(() => {
          const resultElement = document.querySelector('.result-container');
          if (resultElement) {
            resultElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
          }
        }, 100);

        console.log(' Se podría notificar al chat sobre la predicción exitosa');
      },
      error: (error) => {
        console.error(' Error en predicción:', error);
        alert('Error al obtener la predicción. Por favor intente nuevamente.');
        this.cargandoPrediccion = false;
      }
    });
  }
  
  generarReporte(): void {
    if (!this.fechaReporte) {
      alert('Debe seleccionar una fecha para el reporte');
      return;
    }

    this.cargandoReporte = true;
    this.mostrarAnalisis = false;
    this.reporte = [];
    this.mensajeResumen = '';
    
    console.log(' Generando reporte para fecha:', this.fechaReporte);

    this.service.getReportePorFecha(this.fechaReporte).subscribe({
      next: (data) => {
        console.log(' Reporte recibido:', data);
        const resp: any = data;
        
        this.reporte = resp.predictions ?? [];
        this.mensajeResumen = resp.mensaje_resumen ?? '';
        
        console.log(' Total de productos:', this.reporte.length);
        console.log(' Mensaje resumen (longitud):', this.mensajeResumen.length);
        console.log(' Mensaje resumen (primeros 200 chars):', this.mensajeResumen.slice(0, 200));
        console.log(' Mensaje resumen (últimos 200 chars):', this.mensajeResumen.slice(-200));
        
        if (this.mensajeResumen && this.mensajeResumen.trim().length > 0) {
          this.mostrarAnalisis = true;
          
          setTimeout(() => {
            const analysisElement = document.querySelector('.analysis-section');
            if (analysisElement) {
              analysisElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
          }, 500);
        } else {
          console.warn(' No se recibió mensaje de análisis');
        }
        
        this.cargandoReporte = false;

        if (this.reporte.length > 0) {
          setTimeout(() => {
            const tableElement = document.querySelector('.table-section');
            if (tableElement) {
              tableElement.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
          }, 300);
        }
      },
      error: (error) => {
        console.error(' Error al generar reporte:', error);
        alert('Error al generar el reporte. Por favor intente nuevamente.');
        this.cargandoReporte = false;
        this.mostrarAnalisis = false;
      }
    });
  }

  cerrarAnalisis(): void {
    this.mostrarAnalisis = false;
    console.log('🔒 Panel de análisis cerrado');
  }
  
  subirCsv(): void {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.csv';

    input.onchange = () => {
      const file = input.files?.[0];
      
      if (file) {
        console.log(' Archivo seleccionado:', file.name, '(', file.size, 'bytes)');
        
        if (!file.name.toLowerCase().endsWith('.csv')) {
          alert('Por favor seleccione un archivo CSV válido');
          return;
        }

        const confirmar = confirm(
          `¿Desea subir el archivo "${file.name}" y reentrenar el modelo?\n\n` +
          'Este proceso puede tomar varios minutos.'
        );

        if (!confirmar) {
          console.log(' Usuario canceló la subida');
          return;
        }

        console.log(' Subiendo CSV...');

        this.service.subirCsv(file).subscribe({
          next: (resp) => {
            console.log(' CSV subido exitosamente:', resp);
            alert('CSV subido correctamente. Iniciando reentrenamiento del modelo...');
            
            this.reentrenarModelo();
          },
          error: (error) => {
            console.error(' Error al subir CSV:', error);
            alert('Error al subir el archivo CSV. Por favor intente nuevamente.\n\nDetalle: ' + 
                  (error.error?.message || error.message || 'Error desconocido'));
          }
        });
      }
    };

    input.click();
  }

  private reentrenarModelo(): void {
    console.log(' Iniciando reentrenamiento del modelo...');

    this.service.reentrenarModelo().subscribe({
      next: (res) => {
        console.log(' Modelo reentrenado exitosamente:', res);
        alert(
          '¡Modelo reentrenado correctamente! ✓\n\n' +
          'Las nuevas predicciones ya están disponibles con los datos actualizados.'
        );
        
        console.log(' El modelo ha sido actualizado con los nuevos datos');
      },
      error: (error) => {
        console.error(' Error al reentrenar el modelo:', error);
        alert(
          'Error al reentrenar el modelo. Por favor intente nuevamente.\n\n' +
          'Detalle: ' + (error.error?.message || error.message || 'Error desconocido')
        );
      }
    });
  }
  
  private formatDate(date: Date): string {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
  }

  getResumenReporte(): string {
    if (this.reporte.length === 0) {
      return 'No hay datos de reporte disponibles';
    }

    const total = this.reporte.length;
    const promedio = this.reporte.reduce((sum, item) => sum + (item.prediction || 0), 0) / total;
    const maximo = Math.max(...this.reporte.map(item => item.prediction || 0));
    const minimo = Math.min(...this.reporte.map(item => item.prediction || 0));

    return `Reporte de ${total} productos. Promedio: ${promedio.toFixed(2)}, Máximo: ${maximo.toFixed(2)}, Mínimo: ${minimo.toFixed(2)} unidades.`;
  }

  hayAlertasCriticas(): boolean {
    if (!this.reporte || this.reporte.length === 0) {
      return false;
    }

    const stockBajo = this.reporte.filter(item => item.prediction < 50);
    
    const stockAlto = this.reporte.filter(item => item.prediction > 500);

    if (stockBajo.length > 0 || stockAlto.length > 0) {
      console.log(` Alertas: ${stockBajo.length} productos con stock bajo, ${stockAlto.length} con stock alto`);
      return true;
    }

    return false;
  }
}