import { useMutation } from '@tanstack/react-query';
import { driveApi } from '@/lib/api';
import type { SearchRequest } from '@/types';

export const useSearch = () => {
  return useMutation({
    mutationFn: (request: SearchRequest) => driveApi.searchDocuments(request),
  });
};
