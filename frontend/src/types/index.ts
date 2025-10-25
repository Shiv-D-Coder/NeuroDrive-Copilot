export interface DriveFile {
  id: string;
  name: string;
  mimeType: string;
  path: string;
  modifiedTime: string;
  size?: number;
  webViewLink?: string;
}

export interface DriveFolder {
  id: string;
  name: string;
  path: string;
  fileCount: number;
}

export interface SearchResult {
  file: DriveFile;
  snippet: string;
  relevanceScore: number;
  highlights: string[];
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: SearchResult[];
  timestamp: Date;
}

export interface IngestionStatus {
  isIngesting: boolean;
  totalFiles: number;
  processedFiles: number;
  currentFile?: string;
  error?: string;
}

export interface SearchRequest {
  query: string;
  folderId?: string;
  fileId?: string;
  limit?: number;
}

export interface ChatRequest {
  message: string;
  conversationHistory?: ChatMessage[];
  folderId?: string;
  fileId?: string;
}

export interface ChatResponse {
  message: string;
  sources: SearchResult[];
}
