import os
import torch
import logging
import uuid
import json
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from typing import Dict, Any

logger = logging.getLogger(__name__)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

class FinetuningService:
    """파인튜닝된 모델을 사용한 RAG 기반 질문-답변 서비스 (모든 게임 지원)"""
    
    def __init__(self):
        logger.info("🔧 파인튜닝 서비스를 초기화합니다...")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"💻 디바이스: {self.device}")
        
        # 시스템 메시지 (고정)
        self.system_msg = "당신은 보드게임 룰 전문가 AI입니다. 사용자의 질문에 상황에 맞게 정확하고 간결하게 답변해주세요."
        
        # 임베딩 모델 로드 (RAG용)
        self.embed_model = SentenceTransformer("BAAI/bge-m3", device=self.device)
        logger.info("✅ 임베딩 모델 로드 완료")
        
        # RAG 데이터 로드
        self._load_rag_data()
        
        # 모델 로드
        self._load_model()
        
        logger.info("✅ 파인튜닝 서비스 초기화 완료")
    
    def _load_model(self):
        """파인튜닝된 모델 로드"""
        try:
            # 모델 ID
            model_id = "minjeongHuggingFace/exaone-bang-merged"
            logger.info(f"📥 모델 로드 중: {model_id}")
            
            # 모델 및 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            ).eval()
            
            # HuggingFace Pipeline 생성 (샘플링 비활성화)
            self.pipe = pipeline(
                task="text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=256,
                do_sample=False
            )
            
            logger.info("✅ 모델 로드 완료")
                
        except Exception as e:
            logger.error(f"❌ 모델 로드 실패: {str(e)}")
            self.model = None
            self.tokenizer = None
            self.pipe = None
    
    def _load_rag_data(self):
        """RAG용 데이터 로드 (모든 게임 지원)"""
        try:
            # 게임별 벡터 인덱스 경로 (개별 게임 룰 청크를 위한 폴더)
            self.game_vector_base_path = "data/game_data/game_data"
            
            # 게임 전체 룰 데이터
            game_data_path = "data/game.json" # 모든 게임의 상세 룰이 담긴 파일
            if os.path.exists(game_data_path):
                with open(game_data_path, "r", encoding="utf-8") as f:
                    self.game_data = json.load(f)
                logger.info("✅ 게임 룰 데이터 로드 완료")
            else:
                logger.warning("⚠️ 게임 룰 파일이 없습니다. 'game.json' 경로를 확인하세요.")
                self.game_data = []
                
        except Exception as e:
            logger.error(f"❌ RAG 데이터 로드 실패: {str(e)}")
            self.game_data = []
    
    def _search_game_context(self, game_name: str, question: str, top_k: int = 3) -> str:
        """게임별 질문에 대한 관련 룰 컨텍스트 검색 (RAG 서비스와 동일한 로직)"""
        try:
            # 게임별 벡터 인덱스 로드
            game_index_path = os.path.join(self.game_vector_base_path, f"{game_name}.faiss")
            game_chunks_path = os.path.join(self.game_vector_base_path, f"{game_name}.json")
            
            if not os.path.exists(game_index_path) or not os.path.exists(game_chunks_path):
                logger.warning(f"'{game_name}' 게임의 RAG 데이터를 찾을 수 없습니다.")
                return ""
            
            # 벡터 인덱스 및 청크 텍스트 로딩
            index = faiss.read_index(game_index_path)
            with open(game_chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            
            # RAG 검색: 룰 질문에 대한 유사 청크 검색
            q_vec = self.embed_model.encode([question], normalize_embeddings=True)
            D, I = index.search(np.array(q_vec), k=top_k)
            retrieved_chunks = [chunks[i] for i in I[0] if i < len(chunks)]
            
            context = "\n\n".join(retrieved_chunks)
            logger.info(f"🔍 RAG 검색 완료 ({game_name}): {len(retrieved_chunks)}개 청크, 총 길이 {len(context)} 글자")
            
            return context
            
        except Exception as e:
            logger.error(f"❌ RAG 검색 실패 ({game_name}): {str(e)}")
            return ""
    
    def _get_game_rule_text(self, game_name: str) -> str:
        """게임 룰 텍스트 가져오기 (전체 룰용)"""
        try:
            game_data_path = "data/game.json"
            if not os.path.exists(game_data_path):
                return ''
            
            with open(game_data_path, "r", encoding="utf-8") as f:
                game_data = json.load(f)
            
            for game in game_data:
                if game.get("game_name") == game_name:
                    return game.get('text', '')
            
            return ''
        except Exception as e:
            logger.error(f"게임 룰 파일 로드 실패: {str(e)}")
            return ''
    
    def _generate_response(self, query: str, context: str = "") -> str:
        """모델을 사용하여 응답 생성 (RAG 컨텍스트 포함)"""
        try:
            if not self.pipe:
                return "모델이 로드되지 않았습니다."
            
            # RAG 컨텍스트가 있으면 시스템 메시지에 포함
            if context:
                enhanced_system_msg = (
                    "너는 보드게임 룰 전문 AI야. 반드시 아래 규칙을 따라야 해:\n"
                    "- 아래 룰 설명에 있는 내용만 기반해서 정확하게 답변해.\n"
                    "- 룰 설명에 없는 정보는 절대로 지어내거나 상상하지 마.\n"
                    "- 전략 질문이면 룰북을 토대로 구체적인 전략을 제시해.\n\n"
                    f"다음은 관련 룰 정보입니다:\n{context}\n\n"
                    "위 룰 정보를 바탕으로 정확하고 구체적으로 답변해줘."
                )
            else:
                enhanced_system_msg = self.system_msg
            
            # 시스템 + 사용자 프롬프트 명시적으로 구성
            prompt = f"[|system|]{enhanced_system_msg}\n[|user|]{query}\n[|assistant|]"
            
            # HuggingFace Pipeline 사용
            response = self.pipe(prompt, max_new_tokens=256, do_sample=False)
            
            # Pipeline 결과에서 텍스트 추출
            generated_text = response[0]['generated_text'] if response else ""
            
            # 원본 프롬프트 제거하고 생성된 부분만 추출
            if prompt in generated_text:
                content = generated_text.replace(prompt, "").strip()
            else:
                content = generated_text
            
            # 불필요한 토큰 제거
            for marker in ["[|assistant|]", "[|user|]", "[|system|]"]:
                content = content.split(marker)[-1].strip()
            
            return content if content else "죄송합니다. 답변을 생성할 수 없습니다."
            
        except Exception as e:
            logger.error(f"❌ 응답 생성 실패: {str(e)}")
            return f"응답 생성 중 오류가 발생했습니다: {str(e)}"
    
    def get_or_create_session(self, session_id: str) -> str:
        """세션 ID 처리 (단순히 새 ID 생성용)"""
        if not session_id or session_id.strip() == "":
            new_session_id = str(uuid.uuid4())
            logger.info(f"🆕 새 세션 ID 생성: {new_session_id}")
            return new_session_id
        return session_id
    
    async def answer_question(self, game_name: str, question: str, session_id: str = ""):
        """질문 답변 (RAG 검색 후 파인튜닝 모델로 답변 - RAG 서비스와 동일한 로직)"""
        try:
            # 세션 ID만 생성 (실제 히스토리는 저장하지 않음)
            session_id = self.get_or_create_session(session_id)
            
            logger.info(f"🤖 질문 답변 (RAG): {game_name} - {question[:50]}...")
            
            # 1. RAG 검색: 게임별 룰 질문에 대한 유사 청크 검색
            context = self._search_game_context(game_name, question, top_k=4)
            
            # RAG 검색 실패 시 전체 룰을 기반으로 재시도
            if not context or context.strip() == "":
                logger.info(f"RAG 검색 실패. 전체 룰을 기반으로 재시도: {game_name}")
                return await self.get_rule_summary_answer(game_name, question, session_id)
            
            # 2. 파인튜닝 모델로 응답 생성 (RAG 컨텍스트 포함)
            response = self._generate_response(question, context)
            
            logger.info("✅ 질문 답변 완료 (RAG)")
            return response.strip()
            
        except Exception as e:
            logger.error(f"❌ 질문 답변 실패: {str(e)}")
            return f"질문 답변 중 오류가 발생했습니다: {str(e)}"
    
    async def get_rule_summary_answer(self, game_name: str, question: str, session_id: str):
        """전체 룰을 기반으로 질문에 답변 (RAG 서비스와 동일한 로직)"""
        try:
            game_info = next((g for g in self.game_data if g.get("game_name") == game_name), None)
            if not game_info:
                return f"'{game_name}' 게임의 룰 데이터를 찾을 수 없습니다."

            game_rule_text = game_info.get("text", "")
            if not game_rule_text:
                return f"'{game_name}' 게임의 룰 텍스트가 없습니다."
            
            game_info = None
            for game in self.game_data:
                if game.get("game_name") == game_name:
                    game_info = game
                    break
            
            if not game_info:
                return f"'{game_name}' 게임의 전체 룰 정보를 찾을 수 없습니다. 'game.json' 파일을 확인해주세요."
            
            game_rule_text = game_info.get('text', '')
            
            if game_name == "뱅":
                game_data_path = "data/game2.json" # 모든 게임의 상세 룰이 담긴 파일
                os.path.exists(game_data_path)
                with open(game_data_path, "r", encoding="utf-8") as f:
                    self.game_data = json.load(f)
                logger.info("✅ 게임 룰 데이터 로드 완료")
                
                game_info = None
                for game in self.game_data:
                    if game.get("game_name") == game_name:
                        game_info = game
                        break
            
                game_rule_text = game_info.get('text', '')

            # 전체 룰 기반 질문 답변을 위한 시스템 메시지
            enhanced_system_msg = (
                "너는 보드게임 룰 전문 AI야. 반드시 아래 규칙을 따라야 해:\n"
                "- 사용자의 질문에 대해 아래 전체 룰 설명에 있는 내용만 기반해서 답변해.\n"
                "- 룰 설명에 없는 정보는 절대로 지어내거나 상상하지 마.\n"
                "- 사람 이름, 장소, 시간, 인원수 등을 추측하거나 새로 만들어내지 마.\n"
                "- 룰 북을 물어보는게 아닌 전략을 물어보면 너는 룰북을 토대로 전략을 짜줘.\n\n"
                f"아래는 '{game_name}' 보드게임의 전체 룰 설명입니다:\n\n{game_rule_text}\n\n"
                "이 룰을 바탕으로 다음 질문에 정확하고 구체적으로 답변해줘."
            )
            
            # 전체 룰을 시스템 메시지에 포함하여 질문 처리
            prompt = f"[|system|]{enhanced_system_msg}\n[|user|]{question}\n[|assistant|]"
            
            # HuggingFace Pipeline 사용
            response = self.pipe(prompt, max_new_tokens=256, do_sample=False)
            
            # Pipeline 결과에서 텍스트 추출
            generated_text = response[0]['generated_text'] if response else ""
            
            # 원본 프롬프트 제거하고 생성된 부분만 추출
            if prompt in generated_text:
                content = generated_text.replace(prompt, "").strip()
            else:
                content = generated_text
            
            # 불필요한 토큰 제거
            for marker in ["[|assistant|]", "[|user|]", "[|system|]"]:
                content = content.split(marker)[-1].strip()
                
            content = game_rule_text
            
            return content if content else "죄송합니다. 답변을 생성할 수 없습니다."
            
        except Exception as e:
            logger.error(f"❌ 전체 룰 기반 질문 처리 실패: {str(e)}")
            return f"전체 룰 기반 질문 처리 중 오류가 발생했습니다: {str(e)}"
    
    async def get_rule_summary(self, game_name: str, session_id: str = ""):
        """룰 요약 (전체 룰 텍스트 기반)"""
        try:
            session_id = self.get_or_create_session(session_id)
            logger.info(f"🤖 룰 요약: {game_name}")
            
            # 게임 정보 찾기
            game_info = None
            for game in self.game_data:
                if game.get("game_name") == game_name:
                    game_info = game
                    break
            
            if not game_info:
                return f"'{game_name}' 게임의 전체 룰 정보를 찾을 수 없습니다. 'game.json' 파일을 확인해주세요."
            
            game_rule_text = game_info.get('text', '')
            
            if not game_rule_text:
                return f"'{game_name}' 게임의 룰 내용이 비어 있습니다."
            
            # 게임 룰 요약 요청
            query = f"{game_name} 게임의 기본 규칙과 플레이 방법을 설명해주세요."
            
            # 전체 룰을 컨텍스트로 사용하여 요약 생성
            enhanced_system_msg = (
                "너는 보드게임 룰 전문 AI야. 반드시 아래 규칙을 따라야 해:\n"
                "- 사용자의 질문에 대해 아래 전체 룰 설명에 있는 내용만 기반해서 답변해.\n"
                "- 룰 설명에 없는 정보는 절대로 지어내거나 상상하지 마.\n"
                "- 사람 이름, 장소, 시간, 인원수 등을 추측하거나 새로 만들어내지 마.\n"
                "- 룰 북을 물어보는게 아닌 전략을 물어보면 너는 룰북을 토대로 전략을 짜줘.\n\n"
                f"게임 이름: {game_name}\n\n룰 전체:\n{game_rule_text}\n\n"
                "이 게임의 룰을 설명해주세요."
            )
            
            prompt = f"[|system|]{enhanced_system_msg}\n[|user|]{query}\n[|assistant|]"
            
            # HuggingFace Pipeline 사용
            response = self.pipe(prompt, max_new_tokens=256, do_sample=False)
            
            # Pipeline 결과에서 텍스트 추출
            generated_text = response[0]['generated_text'] if response else ""
            
            # 원본 프롬프트 제거하고 생성된 부분만 추출
            if prompt in generated_text:
                content = generated_text.replace(prompt, "").strip()
            else:
                content = generated_text
            
            # 불필요한 토큰 제거
            for marker in ["[|assistant|]", "[|user|]", "[|system|]"]:
                content = content.split(marker)[-1].strip()
            
            logger.info("✅ 룰 요약 완료")
            return content if content else "죄송합니다. 룰 요약을 생성할 수 없습니다."
            
        except Exception as e:
            logger.error(f"❌ 룰 요약 실패: {str(e)}")
            return f"룰 요약 중 오류가 발생했습니다: {str(e)}"
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """세션 정보 조회 (히스토리 없으므로 기본 정보만)"""
        return {
            "session_id": session_id,
            "exists": True,
            "message_count": 0,
            "note": "히스토리를 저장하지 않습니다"
        }
    
    def get_active_sessions(self) -> list:
        """활성 세션 조회 (히스토리 없으므로 빈 목록)"""
        return []
    
    def close_session(self, session_id: str) -> bool:
        """세션 종료 (히스토리 없으므로 항상 성공)"""
        logger.info(f"🗑️ 세션 종료 (히스토리 없음): {session_id}")
        return True
    
    def start_session_cleanup(self):
        """세션 정리 (히스토리 없으므로 불필요)"""
        logger.info("🧹 히스토리가 없으므로 세션 정리가 불필요합니다")
    
    def get_model_info(self):
        """모델 정보 조회"""
        return {
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "pipeline_loaded": self.pipe is not None,
            "embedding_model_loaded": self.embed_model is not None,
            "game_data_loaded": len(self.game_data) > 0 if self.game_data else False,
            "game_count": len(self.game_data) if self.game_data else 0,
            "device": self.device,
            "model_name": "minjeongHuggingFace/exaone-bang-merged",
            "embedding_model": "BAAI/bge-m3",
            "implementation": "huggingface_pipeline_with_rag",
            "features": {
                "stateless": True,
                "no_memory": True,
                "rag_enabled": True,
                "semantic_search": True,
                "multi_game_support": True,
                "fallback_to_full_rules": True
            }
        }
