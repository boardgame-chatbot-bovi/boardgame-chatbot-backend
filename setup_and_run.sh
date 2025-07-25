#!/bin/bash

# 🚀 런팟 AI 백엔드 원클릭 실행 스크립트
# 이 스크립트 하나로 모든 설치 및 실행이 완료됩니다!

set -e  # 에러 발생시 스크립트 중단

echo "🚀 런팟 AI 백엔드를 시작합니다..."
echo "=================================================="

# 환경변수 설정
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/transformers

# 현재 디렉토리 확인
echo "📍 현재 위치: $(pwd)"

# 패키지 설치 여부 확인
if [ ! -f ".packages_installed" ]; then
    echo "📦 Python 패키지를 설치합니다..."
    echo "⏰ 최초 설치시 5-10분 소요될 수 있습니다."
    
    # pip 업그레이드
    echo "1️⃣ pip 업그레이드 중..."
    pip install --upgrade pip
    
    # 충돌 방지를 위한 단계별 설치
    echo "2️⃣ numpy<2 설치 중... (충돌 방지)"
    pip install --no-cache-dir 'numpy<2'
    
    echo "3️⃣ packaging 설치 중... (충돌 방지)"
    pip install --no-cache-dir 'packaging>=23.2,<25.0'
    
    echo "4️⃣ 나머지 패키지 설치 중..."
    pip install --no-cache-dir -r requirements.txt
    
    # 설치 완료 플래그 생성
    touch .packages_installed
    echo "✅ 패키지 설치 완료!"
else
    echo "✅ 패키지가 이미 설치되어 있습니다."
fi

# 환경변수 파일 확인 및 생성
if [ ! -f ".env" ]; then
    echo "⚙️ 환경변수 파일을 생성합니다..."
    cp .env.example .env
    echo "⚠️ .env 파일에서 OPENAI_API_KEY를 설정해주세요!"
    echo "   편집: nano .env"
fi

# 데이터 디렉토리 확인
if [ ! -d "data" ]; then
    echo "📁 데이터 디렉토리를 생성합니다..."
    mkdir -p data/game_data/game_data
fi

# 설치 검증
echo "🔍 설치 검증 중..."
python -c "import fastapi, torch, transformers; print('✅ 모든 핵심 패키지가 정상 설치되었습니다!')" || {
    echo "❌ 패키지 설치에 문제가 있습니다. 다시 시도해주세요."
    rm -f .packages_installed
    exit 1
}

echo "=================================================="
echo "🎉 모든 준비가 완료되었습니다!"
echo ""
echo "🔧 다음 단계:"
echo "1. .env 파일에서 OPENAI_API_KEY 설정"
echo "2. 게임 데이터 파일들을 data/ 폴더에 업로드"
echo "3. 서버 실행: python main.py"
echo ""
echo "🌐 서버 실행 후 접속: http://localhost:8000"
echo "📚 API 문서: http://localhost:8000/docs"
echo "=================================================="

# 자동 실행 옵션
read -p "지금 바로 서버를 실행하시겠습니까? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 서버를 시작합니다..."
    python main.py
fi
