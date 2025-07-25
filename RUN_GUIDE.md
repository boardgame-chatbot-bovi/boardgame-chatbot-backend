# 🚀 업데이트된 파인튜닝 모델 실행 가이드

## 📋 변경사항 요약
- **기존 모델**: beomi/KoAlpaca-Polyglot-5.8B + LoRA 어댑터
- **새 모델**: `minjeongHuggingFace/koalpaca-bang_e9` (허깅페이스에서 직접 로드)

## 🔧 실행 방법

### 1. 의존성 설치
```bash
# 기존 peft 라이브러리 제거 (선택사항)
pip uninstall peft -y

# 업데이트된 requirements 설치
pip install -r requirements.txt
```

### 2. 환경변수 확인
`.env` 파일에서 다음 변수가 올바르게 설정되어 있는지 확인:
```
FINETUNING_MODEL_ID=minjeongHuggingFace/koalpaca-bang_e9
```

### 3. 모델 테스트 (선택사항)
서버 전체를 시작하기 전에 모델만 테스트:
```bash
python test_model.py
```

### 4. 서버 시작
```bash
python main.py
```

## 🔍 확인 사항

### 서버 시작 시 로그
정상적으로 모델이 로드되면 다음과 같은 로그가 나타납니다:
```
📥 파인튜닝된 모델 로드 중: minjeongHuggingFace/koalpaca-bang_e9
📥 Tokenizer 로드 중...
✅ 파인튜닝 모델 로드 완료
```

### API 테스트
```bash
# 헬스체크
curl http://localhost:8000/health

# 파인튜닝 모델 테스트
curl -X POST http://localhost:8000/explain-rules \\
  -H "Content-Type: application/json" \\
  -d '{
    "game_name": "방 탈출",
    "question": "게임 시작은 어떻게 하나요?",
    "chat_type": "finetuning"
  }'
```

## 🚨 문제 해결

### 모델 로드 실패 시
1. 인터넷 연결 확인 (허깅페이스에서 모델 다운로드)
2. 충분한 디스크 용량 확인 (몇 GB 필요)
3. GPU 메모리 확인 (CUDA 사용 시)

### 폴백 모드
모델 로드가 실패하면 자동으로 기본 모델(`beomi/KoAlpaca-Polyglot-5.8B`)로 대체됩니다.

## 📁 주요 변경 파일
- `services/finetuning_service.py` - 모델 로드 로직 변경
- `requirements.txt` - peft 라이브러리 제거
- `.env` & `.env.example` - 모델 ID 업데이트
- `MODEL_UPDATE_SUMMARY.md` - 상세 변경사항 문서
- `test_model.py` - 모델 테스트 스크립트

---
**💡 팁**: 첫 실행 시 모델 다운로드로 인해 시간이 걸릴 수 있습니다. 인내심을 가지고 기다려주세요!
