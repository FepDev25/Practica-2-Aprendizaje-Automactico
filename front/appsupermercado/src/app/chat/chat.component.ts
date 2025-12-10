import { Component, OnInit, ViewChild, ElementRef, AfterViewChecked } from '@angular/core';
import { ChatService } from '../chat.service';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { MarkdownModule } from 'ngx-markdown';

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [FormsModule, CommonModule, MarkdownModule],
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
        let downloadUrl: string | null = null;
        let downloadTitle: string | null = null;

        // Caso A: respuesta conversacional de tu IA
        if (response?.resultado?.message) {
          textoBot = response.resultado.message;
        }

        // Caso B: respuesta clásica {respuesta: "..."}
        else if (response?.respuesta) {
          textoBot = response.respuesta;
        }

        // Detectar si hay URL de descarga
        if (response?.resultado?.data?.url_descarga) {
          downloadUrl = response.resultado.data.url_descarga;

          // Extraer nombre corto del archivo
          const nombreArchivo = response.resultado.data.archivo_generado || 'archivo.pdf';
          // Extraer solo el tipo de reporte (ejemplo: "Reporte_Stock" de "Reporte_Stock_20251209_20251209_193654.pdf")
          const partes = nombreArchivo.split('_');
          downloadTitle = partes.length > 1 ? `${partes[0]} ${partes[1]}` : nombreArchivo;

          // Limpiar el mensaje para no mostrar la URL duplicada
          textoBot = textoBot.replace(/Descárgalo:.*$/i, '').trim();
        }

        this.messages.push({
          from: 'bot',
          text: textoBot,
          downloadUrl: downloadUrl,
          downloadTitle: downloadTitle,
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

  getFullDownloadUrl(url: string): string {
    if (url.startsWith('http://') || url.startsWith('https://')) {
      return url;
    }

    // Usar la misma URL base del ChatService
    const baseUrl = this.chatservice.url;
    const cleanUrl = url.startsWith('/') ? url : `/${url}`;

    return `${baseUrl}${cleanUrl}`;
  }
}
