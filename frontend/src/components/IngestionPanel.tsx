import { Play, Loader2, CheckCircle, AlertCircle } from 'lucide-react';
import { useIngestionStatus } from '@/hooks/useIngestionStatus';
import { useStartIngestion } from '@/hooks/useStartIngestion';

export const IngestionPanel = () => {
  const { data: status } = useIngestionStatus();
  const startIngestion = useStartIngestion();

  const handleStartIngestion = () => {
    startIngestion.mutate();
  };

  const progress =
    status?.totalFiles && status.totalFiles > 0
      ? (status.processedFiles / status.totalFiles) * 100
      : 0;

  return (
    <div className="border border-black rounded-lg p-4 bg-white">
      <h3 className="text-lg font-semibold mb-4 text-black">Drive Ingestion</h3>

      {!status?.isIngesting && !status?.processedFiles && (
        <div className="space-y-3">
          <p className="text-sm text-gray-700">
            Start ingesting your Google Drive files into the vector database for semantic search.
          </p>
          <button
            onClick={handleStartIngestion}
            disabled={startIngestion.isPending}
            className="w-full bg-black text-white rounded-lg px-4 py-2 hover:bg-gray-800 disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition-colors"
          >
            {startIngestion.isPending ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Starting...
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Start Ingestion
              </>
            )}
          </button>
        </div>
      )}

      {status?.isIngesting && (
        <div className="space-y-3">
          <div className="flex items-center gap-2 text-black">
            <Loader2 className="w-5 h-5 animate-spin" />
            <span className="font-medium">Ingesting files...</span>
          </div>

          <div className="space-y-2">
            <div className="flex justify-between text-sm text-gray-700">
              <span>
                {status.processedFiles} / {status.totalFiles} files
              </span>
              <span>{progress.toFixed(0)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-black h-2 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>

          {status.currentFile && (
            <p className="text-sm text-gray-700 truncate">
              Processing: <span className="font-medium">{status.currentFile}</span>
            </p>
          )}
        </div>
      )}

      {!status?.isIngesting && status?.processedFiles && status.processedFiles > 0 && (
        <div className="space-y-3">
          <div className="flex items-center gap-2 text-black">
            <CheckCircle className="w-5 h-5" />
            <span className="font-medium">Ingestion complete</span>
          </div>
          <p className="text-sm text-gray-700">
            Successfully processed {status.processedFiles} files
          </p>
          <button
            onClick={handleStartIngestion}
            disabled={startIngestion.isPending}
            className="w-full bg-white text-black border border-black rounded-lg px-4 py-2 hover:bg-gray-100 disabled:bg-gray-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition-colors"
          >
            <Play className="w-4 h-4" />
            Re-ingest Drive
          </button>
        </div>
      )}

      {status?.error && (
        <div className="mt-3 p-3 bg-gray-100 border border-black rounded-lg">
          <div className="flex items-start gap-2">
            <AlertCircle className="w-5 h-5 text-black flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-medium text-black">Ingestion Error</p>
              <p className="text-sm text-gray-700 mt-1">{status.error}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
