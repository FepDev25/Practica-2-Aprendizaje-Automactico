import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class ChatService {

  url = 'http://localhost:8000';
  constructor(private http: HttpClient) { }

  postChatMessage(msg: string) {
    const params = { query: msg };
    return this.http.post(this.url + '/chat', {}, { params });
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
