import { Suspense } from 'react'
import { ErrorBoundary } from '@/components/ErrorBoundary'
import ResultsClient from './ResultsClient'

export const dynamic = 'force-dynamic'

export default function BTRResultsPage() {
  return (
    <ErrorBoundary>
      <Suspense fallback={<div>Loading...</div>}>
        <ResultsClient />
      </Suspense>
    </ErrorBoundary>
  )
}
