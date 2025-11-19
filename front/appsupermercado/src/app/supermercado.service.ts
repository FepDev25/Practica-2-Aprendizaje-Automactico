import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class SupermercadoService {

  constructor(private http: HttpClient) { }

  apiUrl = 'http://34.10.83.87:8000';
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

    return this.http.post("TU_URL/upload_csv", formData);
  }

}
