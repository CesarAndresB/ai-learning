import { Component, signal } from '@angular/core';
import { ChatLayout } from './components/chat-layout/chat-layout';

@Component({
  selector: 'app-root',
  imports: [ChatLayout],
  templateUrl: './app.html',
  styleUrl: './app.scss'
})
export class App {
  protected readonly title = signal('chat-ui');
}
