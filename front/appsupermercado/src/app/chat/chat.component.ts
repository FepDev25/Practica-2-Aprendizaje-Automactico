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
    if (!msg.trim()) return; // evita enviar vacíos

    // 1. Guardar mensaje del usuario
    this.messages.push({
      from: 'user',
      text: msg,
      time: new Date()
    });

    this.shouldScroll = true;

    // 2. Enviar al backend
    this.chatservice.postChatMessage(msg).subscribe(
      (response: any) => {
        console.log('Respuesta del servidor:', response);

        let textoBot = '';

        if (response?.tipo === 'desconocido') {
          textoBot = response.mensaje || 'No entendí eso.';
        }

        else if (response?.tipo === 'accion') {
          const funcion = response.funcion || 'acción';
          const descripcion = response.meta?.docstring || '';
          textoBot = `${funcion}: ${descripcion}`;
        }

        else {
          textoBot = 'No pude interpretar la respuesta del servidor.';
        }

        // Agregar mensaje del bot
        this.messages.push({
          from: 'bot',
          text: textoBot,
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

    // 3. Limpiar input
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
