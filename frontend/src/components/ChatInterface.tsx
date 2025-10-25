import { useState, useRef, useEffect } from 'react';
import { Send, Folder, File, Loader2 } from 'lucide-react';
import { useChat } from '@/hooks/useChat';
import { useFolders } from '@/hooks/useFolders';
import { useFiles } from '@/hooks/useFiles';
import type { ChatMessage } from '@/types';
import { SourceCard } from './SourceCard';

export const ChatInterface = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [selectedFolderId, setSelectedFolderId] = useState<string | undefined>();
  const [selectedFileId, setSelectedFileId] = useState<string | undefined>();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const chatMutation = useChat();
  const { data: folders } = useFolders();
  const { data: files } = useFiles();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || chatMutation.isPending) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');

    try {
      const response = await chatMutation.mutateAsync({
        message: input,
        conversationHistory: messages,
        folderId: selectedFolderId,
        fileId: selectedFileId,
      });

      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.message,
        sources: response.sources,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your request.',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Context Selector */}
      <div className="border-b border-black p-4 bg-white">
        <div className="flex gap-4 items-center">
          <div className="flex items-center gap-2">
            <Folder className="w-4 h-4 text-black" />
            <select
              value={selectedFolderId || ''}
              onChange={(e) => {
                setSelectedFolderId(e.target.value || undefined);
                setSelectedFileId(undefined);
              }}
              className="border border-black rounded-md px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-black"
            >
              <option value="">All folders</option>
              {folders?.map((folder) => (
                <option key={folder.id} value={folder.id}>
                  {folder.name} ({folder.fileCount} files)
                </option>
              ))}
            </select>
          </div>

          {selectedFolderId && (
            <div className="flex items-center gap-2">
              <File className="w-4 h-4 text-black" />
              <select
                value={selectedFileId || ''}
                onChange={(e) => setSelectedFileId(e.target.value || undefined)}
                className="border border-black rounded-md px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-black"
              >
                <option value="">All files in folder</option>
                {files
                  ?.filter((file) => file.path.includes(selectedFolderId))
                  .map((file) => (
                    <option key={file.id} value={file.id}>
                      {file.name}
                    </option>
                  ))}
              </select>
            </div>
          )}
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-white">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-600">
            <div className="text-center">
              <p className="text-lg font-medium text-black">Ask me anything about your Google Drive</p>
              <p className="text-sm mt-2">
                I can search for documents, find specific information, and answer questions
              </p>
            </div>
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-3xl rounded-lg px-4 py-2 ${
                  message.role === 'user'
                    ? 'bg-black text-white'
                    : 'bg-white border border-black'
                }`}
              >
                <p className="whitespace-pre-wrap">{message.content}</p>
                {message.sources && message.sources.length > 0 && (
                  <div className="mt-4 space-y-2">
                    <p className="text-sm font-medium text-black">Sources:</p>
                    {message.sources.map((source, idx) => (
                      <SourceCard key={idx} source={source} />
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))
        )}
        {chatMutation.isPending && (
          <div className="flex justify-start">
            <div className="bg-white border border-black rounded-lg px-4 py-2 flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span className="text-sm text-black">Thinking...</span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="border-t border-black p-4 bg-white">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about your files..."
            className="flex-1 border border-black rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-black"
            disabled={chatMutation.isPending}
          />
          <button
            type="submit"
            disabled={!input.trim() || chatMutation.isPending}
            className="bg-black text-white rounded-lg px-4 py-2 hover:bg-gray-800 disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center gap-2 transition-colors"
          >
            <Send className="w-4 h-4" />
            Send
          </button>
        </div>
      </form>
    </div>
  );
};
