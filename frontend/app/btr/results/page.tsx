import { Suspense } from 'react'
import ResultsClient from './ResultsClient'

export const dynamic = 'force-dynamic'

export default function BTRResultsPage() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <ResultsClient />
    </Suspense>
  )
}