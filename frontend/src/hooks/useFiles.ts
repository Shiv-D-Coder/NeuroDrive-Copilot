import { useQuery } from '@tanstack/react-query';
import { driveApi } from '@/lib/api';
import { useAuthStatus } from './useAuthStatus';

export const useFiles = () => {
  const { data: authStatus } = useAuthStatus();
  const isAuthenticated = authStatus?.isAuthenticated ?? false;

  return useQuery({
    queryKey: ['files'],
    queryFn: driveApi.listFiles,
    enabled: isAuthenticated, // Only fetch when authenticated
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
};
