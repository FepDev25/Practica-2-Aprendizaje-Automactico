import { Component, OnInit } from '@angular/core';
import { SupermercadoService } from '../supermercado.service';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MarkdownModule } from 'ngx-markdown';

@Component({
  selector: 'app-supermercado',
  imports: [CommonModule, FormsModule, MarkdownModule],
  templateUrl: './supermercado.component.html',
  styleUrls: ['./supermercado.component.scss']
})
export class SupermercadoComponent implements OnInit {
  prediccionResultado: any = null;
  mensajeResumen: string = '';
  fecha: string = '';
  nombre: string = '';
  fechaReporte: string = '';
  reporte: any[] = [];
  
  cargandoPrediccion: boolean = false;
  cargandoReporte: boolean = false;
  mostrarAnalisis: boolean = false;

  constructor(private service: SupermercadoService) { }

  ngOnInit(): void { }

  getPedict(fecha: string, nombre: string) {
    if (!fecha || !nombre) {
      alert('Debe ingresar el producto y la fecha');
      return;
    }

    this.cargandoPrediccion = true;
    
    this.service.getPedriccion(fecha, nombre).subscribe({
      next: (data) => {
        console.log('Predicción recibida:', data);
        this.prediccionResultado = data;
        this.cargandoPrediccion = false;
      },
      error: (error) => {
        console.error('Error en predicción:', error);
        alert('Error al obtener la predicción. Por favor intente nuevamente.');
        this.cargandoPrediccion = false;
      }
    });
  }

  generarReporte() {
    if (!this.fechaReporte) {
      alert('Debe seleccionar una fecha para el reporte');
      return;
    }

    this.cargandoReporte = true;
    this.mostrarAnalisis = false;
    
    this.service.getReportePorFecha(this.fechaReporte).subscribe({
      next: (data) => {
        console.log('Reporte recibido:', data);
        const resp: any = data;
        this.reporte = resp.predictions ?? [];
        this.mensajeResumen = resp.mensaje_resumen;
        
        // Debug: Verificar contenido completo
        console.log('📊 Mensaje resumen (longitud):', this.mensajeResumen.length);
        console.log('📊 Mensaje resumen (completo):', this.mensajeResumen);
        console.log('📊 Últimos 200 caracteres:', this.mensajeResumen.slice(-200));
        
        this.mostrarAnalisis = true;
        this.cargandoReporte = false;
      },
      error: (error) => {
        console.error('Error al generar reporte:', error);
        alert('Error al generar el reporte. Por favor intente nuevamente.');
        this.cargandoReporte = false;
      }
    });
  }

  cerrarAnalisis() {
    this.mostrarAnalisis = false;
    this.mensajeResumen = '';
  }

  subirCsv() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.csv';

    input.onchange = () => {
      const file = input.files?.[0];
      if (file) {
        this.service.subirCsv(file).subscribe({
          next: (resp) => {
            console.log('CSV subido:', resp);
            alert('CSV subido correctamente');
            this.service.reentrenarModelo().subscribe({
              next: (res) => {
                console.log('Modelo reentrenado:', res);
                alert('Modelo reentrenado correctamente');
              },
              error: (error) => {
                console.error('Error al reentrenar el modelo:', error);
                alert('Error al reentrenar el modelo. Por favor intente nuevamente.');
              }
            });
          },
          error: (error) => {
            console.error('Error al subir CSV:', error);
            alert('Error al subir el archivo CSV. Por favor intente nuevamente.');
          }
        });
      }
    };

    input.click();
  }
}