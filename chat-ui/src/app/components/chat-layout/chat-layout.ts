import { Component, ViewChild } from '@angular/core';
import { ChatSidebar } from '../chat-sidebar/chat-sidebar';
import { ChatThread } from '../chat-thread/chat-thread';

@Component({
  selector: 'app-chat-layout',
  imports: [ChatSidebar, ChatThread],
  templateUrl: './chat-layout.html',
  styleUrl: './chat-layout.scss',
  standalone: true
})
export class ChatLayout {
  @ViewChild(ChatThread) thread?: ChatThread;

  onSelectChat(id: string) {
    this.thread?.loadChat(id);
  }

  onNewChat() {
    this.thread?.startNewChat();
  }
}
