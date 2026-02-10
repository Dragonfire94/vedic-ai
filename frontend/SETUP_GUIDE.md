# Next.js 프론트엔드 설치 가이드

## 📦 Step 1: 폴더 생성

```bash
# vedic-ai 프로젝트 루트에서
cd C:\dev\vedic-ai

# 기존 frontend를 백업
ren frontend frontend-old

# 새 frontend 폴더 만들기
mkdir frontend
cd frontend
```

---

## 📥 Step 2: 파일 복사

다운받은 파일들을 다음과 같이 복사:

```
C:\dev\vedic-ai\frontend\
├── package.json
├── next.config.js
├── tsconfig.json
├── tailwind.config.js
├── postcss.config.js
├── app/
│   ├── globals.css
│   ├── layout.tsx
│   └── page.tsx
├── lib/
│   ├── api.ts
│   └── utils.ts
└── components/
    └── ui/
        └── button.tsx
```

---

## 🔧 Step 3: 패키지 설치

```bash
cd C:\dev\vedic-ai\frontend

# npm 패키지 설치
npm install

# 또는 yarn 사용 시
yarn install
```

**설치 시간**: 2-3분

---

## ⚙️ Step 4: 환경변수 설정

`frontend/.env.local` 파일 만들기:

```env
# Railway 백엔드 URL
NEXT_PUBLIC_API_URL=https://vedic-ai-production.up.railway.app

# 로컬 테스트 시
# NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

---

## 🚀 Step 5: 로컬 실행

```bash
npm run dev
```

브라우저 자동으로 열림: http://localhost:3000

---

## 🧩 Step 6: 나머지 UI 컴포넌트 설치

shadcn/ui CLI로 빠르게 추가:

```bash
# shadcn/ui 초기화
npx shadcn-ui@latest init

# 질문 나오면:
# ✔ Would you like to use TypeScript? … yes
# ✔ Which style would you like to use? › Default
# ✔ Which color would you like to use as base color? › Slate
# ✔ Where is your global CSS file? › app/globals.css
# ✔ Would you like to use CSS variables for colors? … yes
# ✔ Where is your tailwind.config.js located? › tailwind.config.js
# ✔ Configure the import alias for components: › @/components
# ✔ Configure the import alias for utils: › @/lib/utils

# 필요한 컴포넌트 추가
npx shadcn-ui@latest add card
npx shadcn-ui@latest add input
npx shadcn-ui@latest add label
npx shadcn-ui@latest add select
npx shadcn-ui@latest add radio-group
npx shadcn-ui@latest add dialog
npx shadcn-ui@latest add badge
npx shadcn-ui@latest add progress
```

**자동으로 `components/ui/` 폴더에 추가됨!**

---

## ✅ Step 7: 테스트

1. http://localhost:3000 접속
2. 생년월일 입력
3. 시간 선택
4. 작동 확인!

---

## 🚢 Step 8: Vercel 배포

### 8-1. GitHub 푸시

```bash
# GitHub Desktop에서:
# 1. frontend 폴더 전체 선택
# 2. Summary: "Add Next.js frontend"
# 3. Commit to main
# 4. Push origin
```

### 8-2. Vercel 배포

1. https://vercel.com 접속
2. "Import Project" 클릭
3. GitHub에서 `vedic-ai` 선택
4. 설정:
   - **Framework Preset**: Next.js
   - **Root Directory**: `frontend`
   - **Environment Variables**:
     - `NEXT_PUBLIC_API_URL` = `https://vedic-ai-production.up.railway.app`
5. "Deploy" 클릭

**3-5분 후** 배포 완료!

---

## 📝 다음 작업 (우선순위)

### 1. 도시 검색 추가 (Google Places)
```bash
npm install @googlemaps/js-api-loader
```

`components/CitySearch.tsx` 만들기

### 2. BTR 페이지 완성
```
app/btr/page.tsx           # BTR 시작
app/btr/questions/page.tsx # 질문 폼
app/btr/results/page.tsx   # 후보 상승궁
```

### 3. 차트 결과 페이지
```
app/chart/page.tsx
```

### 4. 결제 연동
```bash
npm install @tosspayments/payment-sdk
```

---

## 🐛 문제 해결

### 에러: "Module not found"
```bash
# 패키지 재설치
rm -rf node_modules package-lock.json
npm install
```

### 에러: "tailwindcss-animate not found"
```bash
npm install tailwindcss-animate
```

### 포트 충돌 (3000 이미 사용 중)
```bash
# 다른 포트로 실행
npm run dev -- -p 3001
```

---

## 📞 도움이 필요하면

1. 에러 메시지 스크린샷
2. `npm run dev` 터미널 로그
3. 어느 단계에서 막혔는지

보내주세요! 🚀
