import { Suspense } from 'react'
import QuestionsClient from './QuestionsClient'

export const dynamic = 'force-dynamic'

export default function BTRQuestionsPage() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <QuestionsClient />
    </Suspense>
  )
}