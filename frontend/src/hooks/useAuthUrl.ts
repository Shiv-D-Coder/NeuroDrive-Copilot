import { useQuery } from '@tanstack/react-query';
import { driveApi } from '@/lib/api';

export const useAuthUrl = () => {
  return useQuery({
    queryKey: ['authUrl'],
    queryFn: driveApi.getAuthUrl,
    staleTime: Infinity, // Auth URL never changes
  });
};
