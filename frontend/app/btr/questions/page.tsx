import { Suspense } from 'react'
import { ErrorBoundary } from '@/components/ErrorBoundary'
import QuestionsClient from './QuestionsClient'

export const dynamic = 'force-dynamic'

export default function BTRQuestionsPage() {
  return (
    <ErrorBoundary>
      <Suspense fallback={<div>Loading...</div>}>
        <QuestionsClient />
      </Suspense>
    </ErrorBoundary>
  )
}
