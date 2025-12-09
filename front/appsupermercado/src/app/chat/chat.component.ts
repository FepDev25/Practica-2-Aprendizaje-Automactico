import { Component, OnInit, ViewChild, ElementRef, AfterViewChecked } from '@angular/core';
import { ChatService } from '../chat.service';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [FormsModule, CommonModule],
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.scss']
})
export class ChatComponent implements OnInit, AfterViewChecked {

  @ViewChild('messagesScroll') private messagesScroll!: ElementRef;

  message: string = '';
  messages: any[] = [];
  private shouldScroll = false;

  constructor(private chatservice: ChatService) { }

  ngOnInit(): void { }

  ngAfterViewChecked(): void {
    if (this.shouldScroll) {
      this.scrollToBottom();
      this.shouldScroll = false;
    }
  }

  chat(msg: string) {
    if (!msg.trim()) return;

    this.messages.push({
      from: 'user',
      text: msg,
      time: new Date()
    });

    this.shouldScroll = true;

    this.chatservice.postChatMessage(msg).subscribe(
      (response: any) => {
        console.log('Respuesta del servidor:', response);

        let textoBot = 'No recibí una respuesta válida.';

        // Caso A: respuesta conversacional de tu IA
        if (response?.resultado?.message) {
          textoBot = response.resultado.message;
        }

        // Caso B: respuesta clásica {respuesta: "..."}
        else if (response?.respuesta) {
          textoBot = response.respuesta;
        }

        this.messages.push({
          from: 'bot',
          text: textoBot,
          metodo: response?.metodo || response?.metadata?.metodo,
          tipo: response?.tipo || response?.metadata?.tipo,
          confianza: response?.confianza || response?.metadata?.confianza,
          raw: response,
          time: new Date()
        });

        this.shouldScroll = true;
      },


      (error) => {
        console.error('Error al enviar el mensaje:', error);

        this.messages.push({
          from: 'bot',
          text: ' Error conectando con el servidor.',
          time: new Date()
        });

        this.shouldScroll = true;
      }
    );

    this.message = '';
  }

  scrollToBottom(): void {
    try {
      this.messagesScroll.nativeElement.scrollTop =
        this.messagesScroll.nativeElement.scrollHeight;
    } catch (err) {
      console.error('Error al hacer scroll:', err);
    }
  }

  clearChat(): void {
    this.messages = [];
  }
}
