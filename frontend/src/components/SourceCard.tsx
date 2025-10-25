import { ExternalLink, FileText } from 'lucide-react';
import type { SearchResult } from '@/types';

interface SourceCardProps {
  source: SearchResult;
}

export const SourceCard = ({ source }: SourceCardProps) => {
  return (
    <div className="border border-black rounded-lg p-3 bg-white">
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-start gap-2 flex-1 min-w-0">
          <FileText className="w-4 h-4 text-black mt-0.5 flex-shrink-0" />
          <div className="flex-1 min-w-0">
            <p className="font-medium text-sm text-black truncate">
              {source.file.name}
            </p>
            <p className="text-xs text-gray-600 truncate">{source.file.path}</p>
          </div>
        </div>
        {source.file.webViewLink && (
          <a
            href={source.file.webViewLink}
            target="_blank"
            rel="noopener noreferrer"
            className="text-black hover:text-gray-700 flex-shrink-0"
          >
            <ExternalLink className="w-4 h-4" />
          </a>
        )}
      </div>

      {source.snippet && (
        <div className="mt-2">
          <p className="text-sm text-gray-800 line-clamp-3">{source.snippet}</p>
        </div>
      )}

      {source.highlights && source.highlights.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {source.highlights.map((highlight, idx) => (
            <span
              key={idx}
              className="inline-block bg-gray-200 text-black text-xs px-2 py-0.5 rounded border border-black"
            >
              {highlight}
            </span>
          ))}
        </div>
      )}

      <div className="mt-2 flex items-center justify-between">
        <span className="text-xs text-gray-600">
          Relevance: {(source.relevanceScore * 100).toFixed(0)}%
        </span>
        <span className="text-xs text-gray-600">
          {new Date(source.file.modifiedTime).toLocaleDateString()}
        </span>
      </div>
    </div>
  );
};
