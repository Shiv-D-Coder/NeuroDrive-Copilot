import { useQuery } from '@tanstack/react-query';
import { driveApi } from '@/lib/api';
import { useAuthStatus } from './useAuthStatus';

export const useIngestionStatus = () => {
  const { data: authStatus } = useAuthStatus();
  const isAuthenticated = authStatus?.isAuthenticated ?? false;

  return useQuery({
    queryKey: ['ingestionStatus'],
    queryFn: driveApi.getIngestionStatus,
    enabled: isAuthenticated, // Only fetch when authenticated
    refetchInterval: (data) => {
      // Poll every 2 seconds while ingesting, otherwise don't poll
      return data?.isIngesting ? 2000 : false;
    },
  });
};
