import { LogIn, Loader2 } from 'lucide-react';
import { useAuthUrl } from '@/hooks/useAuthUrl';

export const AuthButton = () => {
  const { data, isLoading } = useAuthUrl();

  const handleAuth = () => {
    if (data?.url) {
      window.location.href = data.url;
    }
  };

  return (
    <button
      onClick={handleAuth}
      disabled={isLoading || !data?.url}
      className="bg-black text-white rounded-lg px-6 py-3 hover:bg-gray-800 disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center gap-2 text-lg font-medium transition-colors"
    >
      {isLoading ? (
        <>
          <Loader2 className="w-5 h-5 animate-spin" />
          Loading...
        </>
      ) : (
        <>
          <LogIn className="w-5 h-5" />
          Connect Google Drive
        </>
      )}
    </button>
  );
};
