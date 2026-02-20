# Dev Change Log 
- 2026-02-19: Refactored App Router pages `/chart`, `/btr/questions`, and `/btr/results` to keep `page.tsx` as Server Components with `export const dynamic = "force-dynamic"`, moved `useSearchParams` logic into new client files (`ChartClient.tsx`, `QuestionsClient.tsx`, `ResultsClient.tsx`), and wrapped each page render in `Suspense` with `Loading...` fallback.
- Updated api types and chart param builder; refactored chart/AI/PDF/BTR responses.
- Added BTR zustand store and routed BTR results via store instead of URL params.
- Added personality answers and optional hour to BTR analyze payload/type.
- Removed defensive reading fallback and simplified BTR mid-hour parsing.
- Updated BTR types, removed any in client pages, and centralized toNum utility.
- Restricted CORS origins, secured BTR admin endpoint, and added backend env templates.
