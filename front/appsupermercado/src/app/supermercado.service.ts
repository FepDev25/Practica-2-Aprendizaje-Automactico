import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class SupermercadoService {

  constructor(private http: HttpClient) { }

  apiUrl = 'http://34.10.46.216:8000';
  getPedriccion(fecha: string, nombre: string) {
    return this.http.get<any[]>(
      `${this.apiUrl}/predictPornombre/`,
      { params: { fecha, nombre } }
    );
  }

  getReportePorFecha(fecha: string) {
    return this.http.get<any[]>(
      `${this.apiUrl}/predictAll/`,
      { params: { fecha } }
    );
  }

  subirCsv(file: File) {
    const formData = new FormData();
    formData.append('file', file);

    return this.http.post(`${this.apiUrl}/upload-csv/`, formData);
  }

  reentrenarModelo() {
    return this.http.get(`${this.apiUrl}/reentrenarModelo/`, {});
  }
}
