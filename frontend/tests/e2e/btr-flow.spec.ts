import { test, expect } from '@playwright/test'

test.beforeEach(async ({ page }) => {
  await page.route(/nominatim\.openstreetmap\.org\/search/, (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify([
        {
          place_id: 'seoul-1',
          display_name: 'Seoul, South Korea',
          lat: '37.5665',
          lon: '126.9780',
          type: 'city',
          class: 'place',
        },
      ]),
    })
  )

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

  await page.route(/127\.0\.0\.1:8000\/ai_reading/, (route) => {
    const requestUrl = new URL(route.request().url())
    const isLifeCycle = requestUrl.searchParams.get('product_type') === 'life_cycle'
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(
        isLifeCycle
          ? {
              polished_reading: '## 인생 구조 한 장 요약\n민서님은 기본 life_cycle 리포트를 보고 있습니다.',
              product_type: 'life_cycle',
              summary: {
                structured_summary: {
                  product_type: 'life_cycle',
                  onboarding_goal: requestUrl.searchParams.get('onboarding_goal') || 'life_direction',
                  subject_name: requestUrl.searchParams.get('subject_name') || '민서',
                },
              },
              meta: {
                as_of_local: '2026-03-11T09:00:00+09:00',
                valid_until: '2029-03-11',
                current_mahadasha_planet: 'Moon',
                next_mahadasha_date: '2031-03-11',
                product_type: 'life_cycle',
                contract_version: 'v1.4.0',
                render_profile: 'life_cycle_lite_v1',
                onboarding_goal: requestUrl.searchParams.get('onboarding_goal') || 'life_direction',
              },
            }
          : {
              polished_reading: '기본 AI 해석입니다.',
            }
      ),
    })
  })
})

test('BTR questions page loads with birth params', async ({ page }) => {
  const questionsResponse = page.waitForResponse(/btr\/questions/)
  await page.goto('/btr/questions?year=1994&month=12&day=18&lat=37.5665&lon=126.978&timezone=9')
  await questionsResponse

  await expect(page.getByText('Birth Time Check')).toBeVisible({ timeout: 10000 })
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

test('Home page builds a life_cycle chart URL from exact birth input', async ({ page }) => {
  await page.goto('/')

  await page.locator('#report-life-cycle').click()
  await expect(page.getByLabel('이름')).toBeVisible({ timeout: 10000 })
  await page.getByLabel('출생 도시').fill('Seoul')
  await expect(page.getByRole('button', { name: 'Seoul' })).toBeVisible({ timeout: 10000 })
  await page.getByRole('button', { name: 'Seoul' }).click()

  await page.getByLabel('이름').fill('민서')
  await page.getByLabel('현재 맥락').fill('브랜드 전략 업무')
  await page.getByLabel('관계 상태').fill('싱글')
  await page.getByLabel('집중 토큰').fill('우선순위, 전환')
  await page.getByLabel('걱정 토큰').fill('이직 타이밍, 수입 안정')

  await page.getByRole('button', { name: '다음' }).click()
  await page.getByLabel('정확히 기억함').click()

  await page.getByRole('button', { name: '리포트 보기' }).click()
  await expect(page).toHaveURL(/\/chart\?/, { timeout: 10000 })

  const currentUrl = new URL(page.url())
  expect(currentUrl.searchParams.get('product_type')).toBe('life_cycle')
  expect(currentUrl.searchParams.get('subject_name')).toBe('민서')
  expect(currentUrl.searchParams.get('onboarding_goal')).toBe('life_direction')
  expect(currentUrl.searchParams.get('focus_tokens')).toBe('우선순위,전환')
  expect(currentUrl.searchParams.get('concern_tokens')).toBe('이직 타이밍,수입 안정')
  expect(currentUrl.searchParams.get('occupation_context')).toBe('브랜드 전략 업무')
  expect(currentUrl.searchParams.get('relationship_status')).toBe('싱글')
})

test('Chart page forwards life_cycle params to ai_reading and renders contract metadata', async ({ page }) => {
  let aiReadingUrl = ''
  await page.route(/127\.0\.0\.1:8000\/ai_reading/, (route) => {
    aiReadingUrl = route.request().url()
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        polished_reading: '## 인생 구조 한 장 요약\n민서님은 지금 기준을 좁혀야 하는 시즌입니다.',
        product_type: 'life_cycle',
        summary: {
          structured_summary: {
            product_type: 'life_cycle',
            onboarding_goal: 'career_money',
            subject_name: '민서',
          },
        },
        meta: {
          as_of_local: '2026-03-11T09:00:00+09:00',
          valid_until: '2029-03-11',
          current_mahadasha_planet: 'Moon',
          next_mahadasha_date: '2031-03-11',
          product_type: 'life_cycle',
          contract_version: 'v1.4.0',
          render_profile: 'life_cycle_lite_v1',
          onboarding_goal: 'career_money',
        },
      }),
    })
  })

  const chartResponse = page.waitForResponse(/127\.0\.0\.1:8000\/chart/)
  const aiReadingResponse = page.waitForResponse(/127\.0\.0\.1:8000\/ai_reading/)
  await page.goto(
    '/chart?year=1994&month=12&day=18&hour=14.5&lat=37.5665&lon=126.978&timezone=9&gender=female&house_system=W&product_type=life_cycle&subject_name=민서&onboarding_goal=career_money&focus_tokens=우선순위,전환&concern_tokens=이직 타이밍,수입 안정&occupation_context=브랜드 전략 업무&relationship_status=싱글'
  )
  await chartResponse
  await aiReadingResponse

  await expect(page.getByRole('heading', { name: 'Vedic Life Cycle Report' })).toBeVisible({ timeout: 10000 })

  const requestUrl = new URL(aiReadingUrl)
  expect(requestUrl.searchParams.get('product_type')).toBe('life_cycle')
  expect(requestUrl.searchParams.get('subject_name')).toBe('민서')
  expect(requestUrl.searchParams.get('onboarding_goal')).toBe('career_money')
  expect(requestUrl.searchParams.get('focus_tokens')).toBe('우선순위,전환')
  expect(requestUrl.searchParams.get('concern_tokens')).toBe('이직 타이밍,수입 안정')
  expect(requestUrl.searchParams.get('occupation_context')).toBe('브랜드 전략 업무')
  expect(requestUrl.searchParams.get('relationship_status')).toBe('싱글')

  await expect(page.getByText('리포트 설정')).toBeVisible({ timeout: 10000 })
  await expect(page.getByText('contract_version')).toBeVisible({ timeout: 10000 })
  await expect(page.getByText('민서님은 지금 기준을 좁혀야 하는 시즌입니다.')).toBeVisible({ timeout: 10000 })
})

test('Chart page reuses session cached life_cycle reading on revisit', async ({ page }) => {
  let aiReadingHits = 0
  await page.unroute(/127\.0\.0\.1:8000\/ai_reading/)
  await page.route(/127\.0\.0\.1:8000\/ai_reading/, (route) => {
    aiReadingHits += 1
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        polished_reading: '## 인생 구조 한 장 요약\n세션 캐시 테스트 리포트입니다.',
        product_type: 'life_cycle',
        summary: {
          structured_summary: {
            product_type: 'life_cycle',
            onboarding_goal: 'life_direction',
            subject_name: '민서',
          },
        },
        meta: {
          as_of_local: '2026-03-11T09:00:00+09:00',
          valid_until: '2029-03-11',
          current_mahadasha_planet: 'Moon',
          next_mahadasha_date: '2031-03-11',
          product_type: 'life_cycle',
          contract_version: 'v1.4.0',
          render_profile: 'life_cycle_lite_v1',
          onboarding_goal: 'life_direction',
        },
      }),
    })
  })

  const chartUrl =
    '/chart?year=1994&month=12&day=18&hour=14.5&lat=37.5665&lon=126.978&timezone=9&gender=female&house_system=W&product_type=life_cycle&subject_name=민서&onboarding_goal=life_direction&focus_tokens=우선순위,전환&concern_tokens=이직 타이밍,수입 안정&occupation_context=브랜드 전략 업무&relationship_status=싱글'

  const firstAiReading = page.waitForResponse(/127\.0\.0\.1:8000\/ai_reading/)
  await page.goto(chartUrl)
  await firstAiReading
  await expect(page.getByText('세션 캐시 테스트 리포트입니다.')).toBeVisible({ timeout: 10000 })

  await page.goto('/')
  await page.goto(chartUrl)
  await expect(page.getByText('세션 캐시 테스트 리포트입니다.')).toBeVisible({ timeout: 10000 })
  expect(aiReadingHits).toBe(1)
})
