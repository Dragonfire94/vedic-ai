import { Suspense } from 'react'
import ChartClient from './ChartClient'

export const dynamic = 'force-dynamic'

export default function ChartPage() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <ChartClient />
    </Suspense>
  )
}