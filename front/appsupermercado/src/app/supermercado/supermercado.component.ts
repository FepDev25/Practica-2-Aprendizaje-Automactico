import { Component, OnInit } from '@angular/core';
import { SupermercadoService } from '../supermercado.service';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-supermercado',
  imports: [CommonModule, FormsModule],
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

  constructor(private service: SupermercadoService) { }

  ngOnInit(): void { }

  getPedict(fecha: string, nombre: string) {
    if (!fecha || !nombre) {
      alert('Debe ingresar el producto y la fecha');
      return;
    }

    this.service.getPedriccion(fecha, nombre).subscribe(data => {
      console.log('Predicción recibida:', data);
      this.prediccionResultado = data;
    });
  }

  generarReporte() {
    this.service.getReportePorFecha(this.fechaReporte).subscribe(data => {
      console.log('Reporte recibido:', data);

      const resp: any = data;
      this.reporte = resp.predictions ?? [];
      this.mensajeResumen = resp.mensaje_resumen;
    });
  }

  subirCsv() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.csv';

    input.onchange = () => {
      const file = input.files?.[0];
      if (file) {
        this.service.subirCsv(file).subscribe(resp => {
          console.log('CSV subido:', resp);
          alert('CSV subido correctamente');
        });
      }
    };

    input.click();
  }

}
