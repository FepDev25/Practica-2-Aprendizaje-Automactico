import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class ChatService {

  url = 'http://localhost:8000/chat';
  constructor(private http: HttpClient) { }

  postChatMessage(msg: string) {
    const params = { query: msg };
    return this.http.post('http://localhost:8000/chat', {}, { params });
  }

}
