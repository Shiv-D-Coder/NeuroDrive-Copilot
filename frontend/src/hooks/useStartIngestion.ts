import { useMutation, useQueryClient } from '@tanstack/react-query';
import { driveApi } from '@/lib/api';

export const useStartIngestion = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: driveApi.startIngestion,
    onSuccess: () => {
      // Invalidate and refetch ingestion status
      queryClient.invalidateQueries({ queryKey: ['ingestionStatus'] });
    },
  });
};
