# 🎮 런팟 AI 백엔드 - 보드게임 추천 & 룰 설명

> **🚀 런팟에서 원클릭 실행 가능한 AI 백엔드 서비스**

## ⚡ 빠른 시작 (런팟 SSH)

```bash
# 1. 저장소 클론
git clone <your-repo-url>
cd runpod_ai_backend

# 2. 패키지 설치 (이것만 하면 끝!)
pip install -r requirements.txt

# 3. 환경변수 설정
cp .env.example .env
nano .env  # OPENAI_API_KEY 설정

# 4. 서버 실행
python main.py
```

## 🌟 또는 자동화 스크립트 사용

```bash
chmod +x setup_and_run.sh
./setup_and_run.sh
```

## 📋 필수 준비사항

### 1. 런팟 환경
- **GPU**: RTX 4090 이상 권장
- **RAM**: 16GB 이상
- **Storage**: 50GB 이상

### 2. 환경변수 설정
`.env` 파일에서 다음 값들을 설정:
```env
OPENAI_API_KEY=your-openai-api-key-here
FINETUNING_MODEL_ID=ft:gpt-3.5-turbo-0125:tset::BX2RnWfq
```

### 3. 데이터 파일 업로드
다음 파일들을 `data/` 폴더에 업로드:
- `game.json` - 게임 전체 룰 데이터
- `game_index.faiss` - 게임 추천용 벡터 인덱스  
- `texts.json` - 게임 설명 텍스트
- `game_names.json` - 게임 이름 목록
- `game_data/game_data/` - 개별 게임별 룰 청크 파일들

## 🔗 API 엔드포인트

서버 실행 후 다음 URL에서 사용 가능:

- **서버 주소**: `http://localhost:8000`
- **API 문서**: `http://localhost:8000/docs`
- **헬스체크**: `http://localhost:8000/health`

### 주요 API
- `POST /recommend` - 게임 추천
- `POST /explain-rules` - 룰 설명
- `POST /rule-summary` - 룰 요약
- `GET /games` - 지원 게임 목록

## 🔧 트러블슈팅

### 일반적인 문제들

1. **numpy 버전 충돌**
   ```bash
   pip install --force-reinstall 'numpy<2'
   ```

2. **CUDA 메모리 부족**
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
   ```

3. **패키지 설치 실패**
   ```bash
   pip install --no-cache-dir -r requirements.txt
   ```

4. **모델 로딩 실패**
   - `.env` 파일의 API 키 확인
   - 데이터 파일들이 올바른 위치에 있는지 확인

### 로그 확인
```bash
python main.py  # 콘솔에서 직접 실행하여 로그 확인
```

## 🚀 런팟 최적화 팁

### 1. 메모리 최적화
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export HF_HOME=/workspace/.cache/huggingface
```

### 2. 백그라운드 실행
```bash
nohup python main.py > app.log 2>&1 &
```

### 3. 프로세스 확인
```bash
ps aux | grep python
curl http://localhost:8000/health
```

## 📊 성능 정보

- **모델 로딩**: 30초~2분 (최초 실행시)
- **추천 응답시간**: 1-3초
- **룰 설명 응답시간**: 2-5초
- **메모리 사용량**: 8-12GB (GPU)

## 🛠 개발 정보

### 기술 스택
- **Backend**: FastAPI + Uvicorn
- **AI Models**: 
  - Sentence-Transformers (BAAI/bge-m3)
  - OpenAI GPT-3.5/4
  - KoAlpaca-Polyglot-5.8B (파인튜닝)
- **Vector DB**: FAISS
- **RAG Framework**: LangChain

### 프로젝트 구조
```
runpod_ai_backend/
├── main.py                 # FastAPI 메인 서버
├── services/              # AI 서비스 모듈들
│   ├── rag_service.py     # RAG 기반 추천/질답
│   ├── finetuning_service.py # 파인튜닝 모델
│   └── embedding_service.py  # 임베딩 서비스
├── data/                  # 게임 데이터 및 모델 파일들
├── requirements.txt       # Python 의존성
└── .env                  # 환경변수
```

---

**🎯 문제가 있으시면 이슈 등록하거나 문의해주세요!**
