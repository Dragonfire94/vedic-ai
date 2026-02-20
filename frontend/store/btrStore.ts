import { create } from 'zustand'
import { BTRAnalyzeResponse } from '@/lib/api'

interface BTRStore {
  result: BTRAnalyzeResponse | null
  setResult: (result: BTRAnalyzeResponse) => void
  clear: () => void
}

export const useBTRStore = create<BTRStore>((set) => ({
  result: null,
  setResult: (result) => set({ result }),
  clear: () => set({ result: null }),
}))
