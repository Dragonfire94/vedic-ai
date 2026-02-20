import { test, expect } from '@playwright/test'

test.beforeEach(async ({ page }) => {
  // Use regex patterns -- more reliable than glob for localhost:port URLs
  await page.route(/127\.0\.0\.1:8000\/btr\/questions/, (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        questions: [
          {
            id: 'career_change',
            text: 'Career change?',
            text_ko: '직업 변화가 있었나요?',
            type: 'yesno_date',
            event_type: 'career_change',
          },
        ],
      }),
    })
  )

  await page.route(/127\.0\.0\.1:8000\/btr\/analyze/, (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        status: 'ok',
        birth_date: '1994-12-18',
        candidates: [
          {
            mid_hour: 14.5,
            start_hour: 14.0,
            end_hour: 15.0,
            confidence: 0.82,
            event_matches: 3,
            event_weights_sum: 2.5,
            time_range: '14:00 - 15:00',
            ascendant: 'Aries',
          },
        ],
      }),
    })
  )

  await page.route(/127\.0\.0\.1:8000\/chart/, (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        planets: {
          Sun: {
            house: 1,
            rasi: { name: 'Aries', name_kr: '양자리' },
            nakshatra: { name: 'Ashwini', pada: 1 },
          },
        },
        houses: {
          ascendant: { rasi: { name: 'Aries', name_kr: '양자리' } },
        },
      }),
    })
  )
})

test('BTR questions page loads with birth params', async ({ page }) => {
  // Wait for the questions API response instead of networkidle
  const questionsResponse = page.waitForResponse(/btr\/questions/)
  await page.goto('/btr/questions?year=1994&month=12&day=18&lat=37.5665&lon=126.978&timezone=9')
  await questionsResponse

  // "Birth Time Check" label is always visible (not behind loading state)
  await expect(page.getByText('Birth Time Check')).toBeVisible({ timeout: 10000 })
  // Progress bar appears after questions load
  await expect(page.locator('.h-2').first()).toBeVisible({ timeout: 10000 })
})

test('BTR results page redirects to questions when store is empty', async ({ page }) => {
  await page.goto('/btr/results')
  await expect(page).toHaveURL(/\/btr\/questions/, { timeout: 10000 })
})

test('Chart page loads with URL params', async ({ page }) => {
  const chartResponse = page.waitForResponse(/127\.0\.0\.1:8000\/chart/)
  await page.goto('/chart?year=1994&month=12&day=18&hour=14.5&lat=37.5665&lon=126.978&timezone=9&gender=female&house_system=W')
  await chartResponse

  await expect(page.getByText('Vedic Signature')).toBeVisible({ timeout: 10000 })
})

test('Chart page shows error state gracefully when API fails', async ({ page }) => {
  // Use page.on (not waitForEvent) -- React StrictMode runs useEffect twice
  // in development, causing two alert() dialogs. page.on handles all of them.
  page.on('dialog', async (dialog) => {
    await dialog.dismiss()
  })

  await page.unroute(/127\.0\.0\.1:8000\/chart/)
  await page.route(/127\.0\.0\.1:8000\/chart/, (route) =>
    route.fulfill({ status: 500, body: 'Internal Server Error' })
  )

  await page.goto(
    '/chart?year=1994&month=12&day=18&hour=14.5&lat=37.5665&lon=126.978&timezone=9'
  )

  await expect(page.getByText('차트를 불러오지 못했습니다')).toBeVisible({ timeout: 15000 })
})
