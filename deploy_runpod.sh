#!/bin/bash

# Runpod 배포 스크립트

echo "🚀 Runpod AI 백엔드 배포를 시작합니다..."

# Docker 이미지 빌드
echo "📦 Docker 이미지를 빌드합니다..."
docker build -t boardgame-ai-backend .

# 환경변수 확인
if [ ! -f .env ]; then
    echo "⚠️ .env 파일이 없습니다. .env.example을 복사하여 .env를 생성하고 값을 설정해주세요."
    cp .env.example .env
    exit 1
fi

# Docker Hub에 푸시 (선택사항)
read -p "Docker Hub에 이미지를 푸시하시겠습니까? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "Docker Hub 사용자명을 입력하세요: " DOCKER_USERNAME
    docker tag boardgame-ai-backend $DOCKER_USERNAME/boardgame-ai-backend:latest
    docker push $DOCKER_USERNAME/boardgame-ai-backend:latest
    echo "✅ Docker Hub에 이미지가 푸시되었습니다: $DOCKER_USERNAME/boardgame-ai-backend:latest"
fi

echo "✅ 배포 준비가 완료되었습니다!"
echo ""
echo "다음 단계:"
echo "1. Runpod에서 새 Pod 생성"
echo "2. Docker 이미지 지정: $DOCKER_USERNAME/boardgame-ai-backend:latest (또는 로컬 이미지)"
echo "3. 포트 8000 노출"
echo "4. 환경변수 설정"
echo "5. GPU 할당 (권장: RTX 4090 또는 A6000)"
