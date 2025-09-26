import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';

export type Role = 'user' | 'assistant';
export interface ChatMessage { role: Role; content: string; ts?: string }
export interface ChatDetail { id: string; title: string; messages: ChatMessage[] }
export interface ChatListItem { id: string; title: string; updated_at?: string }
export interface ChatListResponse { items: ChatListItem[] }
export interface Settings { temperature: number }

@Injectable({ providedIn: 'root' })
export class ChatService {
  private http = inject(HttpClient);
  private baseUrl = 'http://localhost:8000';

  createChat() {
    return this.http.post<{ chat_id: string }>(`${this.baseUrl}/chats`, {});
  }

  getChat(chatId: string) {
    return this.http.get<ChatDetail>(`${this.baseUrl}/chats/${chatId}`);
  }

  listChats(limit = 20) {
    return this.http.get<ChatListResponse>(`${this.baseUrl}/chats`, { params: { limit } as any });
  }

  sendMessage(chatId: string, content: string) {
    return this.http.post<{ reply: string }>(`${this.baseUrl}/chats/${chatId}/messages`, { content });
  }

  streamMessage(chatId: string, content: string) {
    const url = new URL(`${this.baseUrl}/chats/${chatId}/stream`);
    url.searchParams.set('content', content);
    return new EventSource(url.toString());
  }

  getSettings() {
    return this.http.get<Settings>(`${this.baseUrl}/settings`);
  }

  updateSettings(patch: Partial<Settings>) {
    return this.http.post<Settings>(`${this.baseUrl}/settings`, patch);
  }
}
