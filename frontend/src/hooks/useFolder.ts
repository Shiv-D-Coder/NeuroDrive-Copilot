import { useQuery } from '@tanstack/react-query';
import { driveApi } from '@/lib/api';
import { useAuthStatus } from './useAuthStatus';

export const useFolder = (folderId: string) => {
  const { data: authStatus } = useAuthStatus();
  const isAuthenticated = authStatus?.isAuthenticated ?? false;

  return useQuery({
    queryKey: ['folder', folderId],
    queryFn: () => driveApi.getFolder(folderId),
    enabled: !!folderId && isAuthenticated, // Only fetch when folderId exists and user is authenticated
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
};
