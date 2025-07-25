from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import logging
from typing import List, Optional

# 서비스 import
from services.embedding_service import EmbeddingService
from services.finetuning_service import FinetuningService
from services.rag_service import RAGService

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="보드게임 AI 백엔드",
    description="Runpod에서 실행되는 보드게임 추천 및 룰 설명 AI 서비스",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response 모델들
class GameRecommendationRequest(BaseModel):
    query: str
    session_id: str = ""  # 빈 값이면 새 세션 생성
    top_k: int = 3

class RuleQuestionRequest(BaseModel):
    game_name: str
    question: str
    session_id: str = ""
    chat_type: str = "gpt"

class GameRuleSummaryRequest(BaseModel):
    game_name: str
    session_id: str = ""
    chat_type: str = "gpt"

class SessionCloseRequest(BaseModel):
    session_id: str

class APIResponse(BaseModel):
    status: str
    data: Optional[dict] = None
    message: Optional[str] = None

# 전역 변수로 서비스 인스턴스 저장
services_initialized = False
embedding_service = None
finetuning_service = None
rag_service = None

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 AI 모델들을 로드"""
    global services_initialized
    logger.info("🚀 AI 백엔드 서버를 시작합니다...")
    
    try:
        # 서비스 초기화 (실제 모델 로드)
        global embedding_service, finetuning_service, rag_service
        
        # RAG 서비스는 필수 (게임 추천 및 룰 설명)
        rag_service = RAGService()
        
        # 파인튜닝 서비스는 선택사항 (모델 파일이 있을 때만)
        try:
            finetuning_service = FinetuningService()
            # 파인튜닝 서비스의 세션 정리 작업 시작
            try:
                finetuning_service.start_session_cleanup()
            except AttributeError:
                logger.warning("⚠️ 파인튜닝 서비스에 세션 정리 기능이 없습니다. 계속 진행합니다.")
        except Exception as e:
            logger.warning(f"⚠️ 파인튜닝 서비스 로드 실패 (계속 진행): {str(e)}")
            finetuning_service = None
        
        # 세션 정리 작업 시작
        try:
            rag_service.start_session_cleanup()
        except AttributeError:
            logger.warning("⚠️ RAG 서비스에 세션 정리 기능이 없습니다. 계속 진행합니다.")
        
        # 임베딩 서비스는 현재 RAG에 포함되어 있어 별도 로드하지 않음
        # embedding_service = EmbeddingService()
        
        services_initialized = True
        logger.info("✅ 모든 AI 서비스가 성공적으로 초기화되었습니다!")
        
    except Exception as e:
        logger.error(f"❌ 서비스 초기화 실패: {str(e)}")
        services_initialized = False

@app.get("/health")
async def health_check():
    """헬스체크 엔드포인트"""
    return {
        "status": "healthy" if services_initialized else "initializing",
        "services_loaded": services_initialized,
        "message": "보드게임 AI 백엔드가 정상 작동 중입니다!"
    }

@app.post("/recommend", response_model=APIResponse)
async def recommend_games(request: GameRecommendationRequest):
    """게임 추천 API"""
    try:
        if not services_initialized:
            raise HTTPException(status_code=503, detail="서비스가 아직 초기화되지 않았습니다.")
        
        # 세션 ID 처리: 빈 값이면 새 세션 생성
        session_id = rag_service.get_or_create_session(request.session_id)
        
        logger.info(f"게임 추천 요청: {request.query}, 세션: {session_id}")
        
        # RAG 서비스 호출
        result = await rag_service.recommend_games(request.query, session_id, request.top_k)
        
        return APIResponse(
            status="success",
            data={
                "recommendation": result,
                "session_id": session_id  # 세션 ID 반환
            },
            message="게임 추천이 완료되었습니다."
        )
        
    except Exception as e:
        logger.error(f"게임 추천 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"게임 추천 중 오류가 발생했습니다: {str(e)}")

@app.post("/explain-rules", response_model=APIResponse)
async def explain_rules(request: RuleQuestionRequest):
    """룰 설명 API"""
    try:
        if not services_initialized:
            raise HTTPException(status_code=503, detail="서비스가 아직 초기화되지 않았습니다.")
        
        # 세션 ID 처리
        session_id = rag_service.get_or_create_session(request.session_id)
        
        logger.info(f"룰 질문: {request.game_name} - {request.question}, 세션: {session_id}")
        
        # 서비스 호출
        if request.chat_type == "finetuning" and finetuning_service:
            result = await finetuning_service.answer_question(request.game_name, request.question, session_id)
        else:
            result = await rag_service.answer_rule_question(request.game_name, request.question, session_id)
        
        return APIResponse(
            status="success",
            data={
                "answer": result,
                "session_id": session_id
            },
            message="룰 설명이 완료되었습니다."
        )
        
    except Exception as e:
        logger.error(f"룰 설명 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"룰 설명 중 오류가 발생했습니다: {str(e)}")

@app.post("/rule-summary", response_model=APIResponse)
async def get_rule_summary(request: GameRuleSummaryRequest):
    """게임 룰 요약 API"""
    try:
        if not services_initialized:
            raise HTTPException(status_code=503, detail="서비스가 아직 초기화되지 않았습니다.")
        
        # 세션 ID 처리
        session_id = rag_service.get_or_create_session(request.session_id)
        
        logger.info(f"룰 요약 요청: {request.game_name}, 세션: {session_id}")
        
        # 서비스 호출
        if request.chat_type == "finetuning" and finetuning_service:
            result = await finetuning_service.get_rule_summary(request.game_name, session_id)
        else:
            result = await rag_service.get_rule_summary(request.game_name, session_id)
        
        return APIResponse(
            status="success",
            data={
                "summary": result,
                "session_id": session_id
            },
            message="룰 요약이 완료되었습니다."
        )
        
    except Exception as e:
        logger.error(f"룰 요약 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"룰 요약 중 오류가 발생했습니다: {str(e)}")

@app.get("/games")
async def get_available_games():
    """사용 가능한 게임 목록 API"""
    try:
        # 게임 목록 로드
        games = rag_service.get_available_games()
        
        return APIResponse(
            status="success",
            data={"games": games},
            message=f"총 {len(games)}개의 게임을 지원합니다."
        )
        
    except Exception as e:
        logger.error(f"게임 목록 조회 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"게임 목록 조회 중 오류가 발생했습니다: {str(e)}")

@app.post("/session/close", response_model=APIResponse)
async def close_session(request: SessionCloseRequest):
    """세션 종료 API"""
    try:
        logger.info(f"세션 종료 요청: {request.session_id}")
        
        # 모든 서비스의 세션 종료 (추천, GPT, 파인튜닝)
        rag_success = rag_service.close_session(request.session_id, "all")  # 추천 + GPT 세션 모두 종료
        
        # 파인튜닝 서비스 세션 종료 (있는 경우)
        finetuning_success = True
        if finetuning_service:
            try:
                finetuning_success = finetuning_service.close_session(request.session_id)
            except AttributeError:
                logger.warning("⚠️ 파인튜닝 서비스에 세션 종료 기능이 없습니다.")
                finetuning_success = True
        
        success = rag_success or finetuning_success
        
        return APIResponse(
            status="success" if success else "warning",
            message=f"세션 {request.session_id} 종료 완료" if success else f"세션 {request.session_id}를 찾을 수 없습니다."
        )
        
    except Exception as e:
        logger.error(f"세션 종료 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"세션 종료 중 오류가 발생했습니다: {str(e)}")

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "보드게임 AI 백엔드 서버",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "recommend": "/recommend",
            "explain_rules": "/explain-rules",
            "rule_summary": "/rule-summary",
            "games": "/games",
            "session_close": "/session/close"
        }
    }

if __name__ == "__main__":
    import os
    
    # 환경변수 설정
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    print("🚀 보드게임 AI 백엔드 서버를 시작합니다...")
    print(f"📡 서버 주소: http://{host}:{port}")
    print(f"📚 API 문서: http://{host}:{port}/docs")
    print(f"🔍 헬스체크: http://{host}:{port}/health")
    print("⏰ 모델 로딩에 30초~2분 정도 소요됩니다...")
    print("="*50)
    
    # 서버 실행
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,  # 프로덕션에서는 reload=False
        log_level="info",
        access_log=True
    )
