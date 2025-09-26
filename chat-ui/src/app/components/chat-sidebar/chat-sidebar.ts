import { Component, EventEmitter, Output, OnInit, inject } from '@angular/core';
import { CommonModule, DecimalPipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ButtonModule } from 'primeng/button';
import { DividerModule } from 'primeng/divider';
import { InputTextModule } from 'primeng/inputtext';
import { ListboxModule } from 'primeng/listbox';
import { ChatService, ChatListItem, Settings } from '../../services/chat.service';

type ChatItem = {
  id: string;
  title: string;
  date: string;
  pinned?: boolean;
  section?: 'recent' | 'last7';
};

@Component({
  selector: 'app-chat-sidebar',
  imports: [CommonModule, FormsModule, ButtonModule, InputTextModule, ListboxModule, DividerModule, DecimalPipe],
  templateUrl: './chat-sidebar.html',
  styleUrl: './chat-sidebar.scss',
  standalone: true,
})
export class ChatSidebar implements OnInit {
  private api = inject(ChatService);

  @Output() selectChat = new EventEmitter<string>();
  @Output() newChat = new EventEmitter<void>();

  chats: ChatListItem[] = [];
  activeId: string | null = null;

  // Settings
  temperature = 0.4;
  savingTemp = false;

  async ngOnInit() {
    // Load chats
    this.api.listChats(50).subscribe((res) => {
      this.chats = res.items;
      if (!this.activeId && this.chats.length) {
        this.activeId = this.chats[0].id;
        this.selectChat.emit(this.activeId);
      }
    });
    // Load settings
    this.api.getSettings().subscribe((s) => (this.temperature = s.temperature));
  }

  onSelect(id: string) {
    this.activeId = id;
    this.selectChat.emit(id);
  }

  onNewChat() {
    this.newChat.emit();
  }

  onTempChangeCommit() {
    this.savingTemp = true;
    this.api.updateSettings({ temperature: this.temperature }).subscribe({
      next: (s) => (this.temperature = s.temperature),
      error: () => {},
      complete: () => (this.savingTemp = false),
    });
  }
}
