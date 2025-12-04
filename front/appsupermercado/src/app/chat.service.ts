import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class ChatService {

  url = 'http://34.10.46.216:8000';
  constructor(private http: HttpClient) { }

  postChatMessage(msg: string) {
    const body = { mensaje: msg };
    return this.http.post(this.url + '/chat', body);
  }

  responder_faq(pregunta: string) {
    const params = { pregunta: pregunta };
    return this.http.get(this.url + '/rag/faq', { params });
  }

  consultar_empresa(pregunta: string) {
    const params = { pregunta: pregunta };
    return this.http.get(this.url + '/rag/empresa', { params });
  }

  responder_pregunta(pregunta: string) {
    const params = { pregunta: pregunta };
    return this.http.get(this.url + '/rag/pregunta', { params });
  }
}
