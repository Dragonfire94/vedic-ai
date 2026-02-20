import { Suspense } from 'react'
import { ErrorBoundary } from '@/components/ErrorBoundary'
import ChartClient from './ChartClient'

export const dynamic = 'force-dynamic'

export default function ChartPage() {
  return (
    <ErrorBoundary>
      <Suspense fallback={<div>Loading...</div>}>
        <ChartClient />
      </Suspense>
    </ErrorBoundary>
  )
}
