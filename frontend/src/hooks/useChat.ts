import { useMutation } from '@tanstack/react-query';
import { driveApi } from '@/lib/api';
import type { ChatRequest } from '@/types';

export const useChat = () => {
  return useMutation({
    mutationFn: (request: ChatRequest) => driveApi.chat(request),
  });
};
