import { useQuery } from '@tanstack/react-query';
import { driveApi } from '@/lib/api';

export const useAuthStatus = () => {
  return useQuery({
    queryKey: ['authStatus'],
    queryFn: driveApi.getAuthStatus,
    refetchInterval: 5000, // Check every 5 seconds
    staleTime: 0, // Always consider data stale to ensure we check regularly
  });
};
