import { Component, ElementRef, ViewChild, AfterViewInit, AfterViewChecked, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ButtonModule } from 'primeng/button';
import { InputTextModule } from 'primeng/inputtext';
import { DividerModule } from 'primeng/divider';
import { HttpClientModule } from '@angular/common/http';
import { firstValueFrom } from 'rxjs';
import { ChatService } from '../../services/chat.service';

@Component({
  selector: 'app-chat-thread',
  imports: [CommonModule, FormsModule, HttpClientModule, DividerModule, ButtonModule, InputTextModule],
  templateUrl: './chat-thread.html',
  styleUrl: './chat-thread.scss',
  standalone: true
})
export class ChatThread implements AfterViewInit, AfterViewChecked {
  @ViewChild('chatContainer') chatContainer?: ElementRef<HTMLElement>;
  private chatService = inject(ChatService);
  input = '';
  messages: { id: string; role: 'user' | 'assistant'; content: string; loading?: boolean; verbose?: string }[] = [];
  private chatId: string | null = null;
  isBusy = false;

  // SSE (stream) controller
  private sse?: EventSource;


  async send() {
    const text = this.input.trim();
    if (!text || this.isBusy) return;
    this.isBusy = true;
    const userMsg = { id: String(Date.now()), role: 'user' as const, content: text };
    this.messages.push(userMsg);
    this.input = '';
    this.scrollToBottom();

    // Ensure chat exists
    if (!this.chatId) {
      const res = await firstValueFrom(this.chatService.createChat());
      this.chatId = res.chat_id;
    }

    // Add assistant typing placeholder
    const placeholderId = `pending-${Date.now()}`;
    const placeholder = { id: placeholderId, role: 'assistant' as const, content: '', loading: true, verbose: '' };
    this.messages.push(placeholder);
    this.scrollToBottom();

    // Start SSE stream to get real-time verbose and tokens
    let assembled = '';
    this.sse = this.chatService.streamMessage(this.chatId!, userMsg.content);
    this.sse.addEventListener('verbose', (e: MessageEvent) => {
      const idx = this.messages.findIndex((m) => m.id === placeholderId);
      if (idx !== -1 && this.messages[idx].loading) {
        this.messages[idx].verbose = e.data;
      }
    });
    this.sse.addEventListener('delta', (e: MessageEvent) => {
      assembled += e.data;
      const idx = this.messages.findIndex((m) => m.id === placeholderId);
      if (idx !== -1 && this.messages[idx].loading) {
        this.messages[idx].content = assembled;
      }
      this.scrollToBottom();
    });
    this.sse.addEventListener('done', (e: MessageEvent) => {
      const idx = this.messages.findIndex((m) => m.id === placeholderId);
      if (idx !== -1) {
        this.messages[idx] = { ...this.messages[idx], loading: false, verbose: undefined };
      }
      this.cleanupStream();
      this.isBusy = false;
      this.scrollToBottom();
    });

    // Fallback: if stream errors, close and re-enable
    this.sse.onerror = () => {
      this.cleanupStream();
      this.isBusy = false;
      this.scrollToBottom();
    };
  }

  async loadChat(id: string) {
    if (this.isBusy) return;
    this.chatId = id;
    const detail = await firstValueFrom(this.chatService.getChat(id));
    this.messages = detail.messages.map((m, idx) => ({ id: `${idx}-${m.ts ?? idx}`, role: m.role, content: m.content }));
    this.scrollToBottom();
  }

  async startNewChat() {
    if (this.isBusy) return;
    this.chatId = null;
    this.messages = [];
    this.input = '';
    this.scrollToBottom();
  }

  private scrollToBottom(): void {
    const el = this.chatContainer?.nativeElement;
    if (!el) return;
    // scroll to bottom smoothly
    el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
  }

  ngAfterViewInit(): void {
    // Nothing yet; ChatLayout will instruct which chat to load. Keep scroll at bottom.
    this.scrollToBottom();
  }

  ngAfterViewChecked(): void {
    // Ensure we stay pinned to bottom when new messages render
    this.scrollToBottom();
  }

  private cleanupStream() {
    if (this.sse) {
      this.sse.close();
      this.sse = undefined;
    }
  }
}
