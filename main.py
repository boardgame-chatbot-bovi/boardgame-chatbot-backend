import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class EmbeddingService:
    """임베딩 모델 서비스 (현재는 RAGService에 포함되어 있어 별도 사용하지 않음)"""
    
    def __init__(self):
        logger.info("🔧 임베딩 서비스를 초기화합니다...")
        
        try:
            # 임베딩 모델 로드
            self.model = SentenceTransformer("BAAI/bge-m3", device="cuda")
            logger.info("✅ 임베딩 모델 로드 완료")
            
        except Exception as e:
            logger.error(f"❌ 임베딩 모델 로드 실패: {str(e)}")
            self.model = None
    
    def encode(self, texts, normalize=True):
        """텍스트를 임베딩으로 변환"""
        try:
            if not self.model:
                raise Exception("임베딩 모델이 로드되지 않았습니다.")
            
            embeddings = self.model.encode(texts, normalize_embeddings=normalize)
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ 임베딩 생성 실패: {str(e)}")
            return None
    
    def get_model_info(self):
        """모델 정보 반환"""
        return {
            "model_loaded": self.model is not None,
            "model_name": "BAAI/bge-m3",
            "device": "cuda" if self.model and hasattr(self.model, 'device') else "unknown"
        }
