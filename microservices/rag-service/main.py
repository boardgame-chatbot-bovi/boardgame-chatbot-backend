import json
import faiss
import numpy as np
import os
import re
import logging
import time
import uuid
import threading
import gc
from sentence_transformers import SentenceTransformer

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# 세션 기반 클래스 메모리 정의 (LangChain용)
class InMemoryHistory(BaseChatMessageHistory):
    def __init__(self):
        self.messages = []
        self.last_access = time.time()  # 마지막 접근 시간
        logger.info(f"🧠 새 InMemoryHistory 인스턴스 생성")

    def add_messages(self, messages):
        logger.info(f"📝 메시지 추가: {len(messages)}개 (기존: {len(self.messages)}개)")
        self.messages.extend(messages)
        self.last_access = time.time()  # 접근 시간 업데이트
        logger.info(f"📝 추가 후 총 메시지: {len(self.messages)}개")

    def clear(self):
        logger.info(f"🗑️ 메시지 히스토리 클리어 (기존: {len(self.messages)}개)")
        self.messages = []
        self.last_access = time.time()

    def __repr__(self):
        return f"InMemoryHistory({len(self.messages)} messages)"

# 기능별 독립적인 세션 저장소
recommendation_store = {}  # 추천 전용 세션
gpt_rule_store = {}       # GPT 룰 설명 전용 세션

# 추천 기능용 세션 히스토리
def get_session_history_for_recommendation(session_id: str) -> InMemoryHistory:
    if session_id not in recommendation_store:
        recommendation_store[session_id] = InMemoryHistory()
    else:
        recommendation_store[session_id].last_access = time.time()
    return recommendation_store[session_id]

# GPT 룰 설명용 세션 히스토리
def get_session_history_for_gpt_rules(session_id: str) -> InMemoryHistory:
    if session_id not in gpt_rule_store:
        logger.info(f"🆕 새 GPT 룰 세션 생성: {session_id}")
        gpt_rule_store[session_id] = InMemoryHistory()
    else:
        logger.info(f"🔄 기존 GPT 룰 세션 접근: {session_id} (메시지: {len(gpt_rule_store[session_id].messages)}개)")
        gpt_rule_store[session_id].last_access = time.time()
    return gpt_rule_store[session_id]


class RAGService:
    """RAG 기반 게임 추천 및 룰 설명 서비스"""
    
    def __init__(self):
        logger.info("🔧 RAG 서비스를 초기화합니다...")
        
        # 임베딩 모델 로드
        self.embed_model = SentenceTransformer("BAAI/bge-m3", device="cuda")
        logger.info("✅ 임베딩 모델 로드 완료")
        
        # OpenAI 설정 (LangChain ChatOpenAI 사용)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.model_id = "gpt-4o" # 파인튜닝 모델 ID
        self.llm = ChatOpenAI(model_name=self.model_id, temperature=0.7, openai_api_key=self.openai_api_key)
        
        # 세션 관리 설정
        self.session_timeout = 40 * 60  # 40분 (초 단위)
        self.cleanup_interval = 5 * 60  # 5분마다 정리 (초 단위)
        self.cleanup_thread = None
        self.cleanup_running = False
        
        # 게임 추천용 데이터 로드
        self._load_recommendation_data()
        
        # 게임 룰 데이터 로드
        self._load_game_rules_data()
        
        # LangChain 체인 설정
        self._setup_langchain_chains()
        
        logger.info("✅ RAG 서비스 초기화 완료")
    
    def get_or_create_session(self, session_id: str) -> str:
        """세션 ID가 빈 값이면 새로 생성, 아니면 기존 세션 반환"""
        if not session_id or session_id.strip() == "":
            # 새 세션 생성
            new_session_id = str(uuid.uuid4())
            logger.info(f"🆕 새 세션 생성: {new_session_id}")
            return new_session_id
        else:
            # 기존 세션 반환
            return session_id
    
    def close_session(self, session_id: str, session_type: str = "all") -> bool:
        """세션 종료 (메모리에서 삭제)"""
        global recommendation_store, gpt_rule_store
        
        closed = False
        
        if session_type in ["all", "recommendation"]:
            if session_id in recommendation_store:
                del recommendation_store[session_id]
                logger.info(f"🗑️ 추천 세션 종료: {session_id}")
                closed = True
        
        if session_type in ["all", "gpt"]:
            if session_id in gpt_rule_store:
                del gpt_rule_store[session_id]
                logger.info(f"🗑️ GPT 룰 세션 종료: {session_id}")
                closed = True
        
        if not closed:
            logger.warning(f"⚠️ 종료할 세션을 찾을 수 없음: {session_id} (타입: {session_type})")
        
        return closed
    
    def start_session_cleanup(self):
        """백그라운드 세션 정리 작업 시작"""
        if self.cleanup_thread is None or not self.cleanup_thread.is_alive():
            self.cleanup_running = True
            self.cleanup_thread = threading.Thread(target=self._cleanup_sessions_worker, daemon=True)
            self.cleanup_thread.start()
            logger.info(f"🧹 세션 정리 작업 시작 (간격: {self.cleanup_interval//60}분)")
    
    def _cleanup_sessions_worker(self):
        """백그라운드에서 실행되는 세션 정리 작업"""
        global recommendation_store, gpt_rule_store
        while self.cleanup_running:
            try:
                current_time = time.time()
                
                # 추천 세션 정리
                expired_recommendation_sessions = []
                for session_id, history in recommendation_store.items():
                    if current_time - history.last_access > self.session_timeout:
                        expired_recommendation_sessions.append(session_id)
                
                for session_id in expired_recommendation_sessions:
                    del recommendation_store[session_id]
                    logger.info(f"⏰ 추천 세션 타임아웃으로 삭제: {session_id}")
                
                # GPT 룰 세션 정리
                expired_gpt_sessions = []
                for session_id, history in gpt_rule_store.items():
                    if current_time - history.last_access > self.session_timeout:
                        expired_gpt_sessions.append(session_id)
                
                for session_id in expired_gpt_sessions:
                    del gpt_rule_store[session_id]
                    logger.info(f"⏰ GPT 룰 세션 타임아웃으로 삭제: {session_id}")
                
                total_expired = len(expired_recommendation_sessions) + len(expired_gpt_sessions)
                total_active = len(recommendation_store) + len(gpt_rule_store)
                
                if total_expired > 0:
                    logger.info(f"🧹 {total_expired}개 세션 정리 완료. 현재 활성 세션: {total_active}개 (추천: {len(recommendation_store)}, GPT룰: {len(gpt_rule_store)})")
                
                time.sleep(self.cleanup_interval)
                
            except Exception as e:
                logger.error(f"❌ 세션 정리 작업 오류: {str(e)}")
                time.sleep(60)  # 오류 시 1분 후 재시도
    
    def _load_recommendation_data(self):
        """게임 추천용 데이터 로드"""
        try:
            # 게임 추천용 FAISS 인덱스
            index_path = "data/game_index.faiss"
            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)
                logger.info("✅ 게임 추천 인덱스 로드 완료")
            else:
                logger.warning("⚠️ 게임 추천 인덱스 파일이 없습니다. 'game_index.faiss' 경로를 확인하세요.")
                self.index = None
            
            # 게임 텍스트 데이터
            texts_path = "data/texts.json"
            if os.path.exists(texts_path):
                with open(texts_path, "r", encoding="utf-8") as f:
                    self.texts = json.load(f)
                logger.info("✅ 게임 텍스트 데이터 로드 완료")
            else:
                logger.warning("⚠️ 게임 텍스트 파일이 없습니다. 'texts.json' 경로를 확인하세요.")
                self.texts = []
            
            # 게임 이름 데이터
            names_path = "data/game_names.json"
            if os.path.exists(names_path):
                with open(names_path, "r", encoding="utf-8") as f:
                    self.game_names = json.load(f)
                logger.info("✅ 게임 이름 데이터 로드 완료")
            else:
                logger.warning("⚠️ 게임 이름 파일이 없습니다. 'game_names.json' 경로를 확인하세요.")
                self.game_names = []
                
        except Exception as e:
            logger.error(f"❌ 게임 추천 데이터 로드 실패: {str(e)}")
            self.index = None
            self.texts = []
            self.game_names = []
    
    def _load_game_rules_data(self):
        """게임 룰 데이터 로드"""
        try:
            # 게임 전체 룰 데이터
            game_data_path = "data/game.json" # 모든 게임의 상세 룰이 담긴 파일
            if os.path.exists(game_data_path):
                with open(game_data_path, "r", encoding="utf-8") as f:
                    self.game_data = json.load(f)
                logger.info("✅ 게임 룰 데이터 로드 완료")
            else:
                logger.warning("⚠️ 게임 룰 파일이 없습니다. 'game.json' 경로를 확인하세요.")
                self.game_data = []
            
            # 게임별 벡터 인덱스 경로 (개별 게임 룰 청크를 위한 폴더)
            self.game_vector_base_path = "data/game_data/game_data"
            
        except Exception as e:
            logger.error(f"❌ 게임 룰 데이터 로드 실패: {str(e)}")
            self.game_data = []

    def _setup_langchain_chains(self):
        """LangChain 체인 및 프롬프트 설정"""
        # 게임 추천 프롬프트 (search_similar_context의 결과를 {context}로 받음)
        recommendation_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "너는 보드게임 추천 도우미야. 다음은 추천 가능한 게임 설명들이야:\n\n{context}\n\n"
                "반드시 이 게임 목록 안에서만 추천해. 새로운 게임을 지어내지 마.\n"
                "질문에 맞는 게임 3개를 골라 아래 형식으로 답해:\n"
                "게임명1: 이유\n게임명2: 이유\n게임명3: 이유"
            ),
            MessagesPlaceholder(variable_name="history"), # 세션 히스토리
            ("human", "{query}")
        ])

        # 게임 추천 체인 (RunnableWithMessageHistory로 히스토리 관리)
        self.recommendation_chain = RunnableWithMessageHistory(
            recommendation_prompt | self.llm,
            get_session_history=get_session_history_for_recommendation,  # 추천 전용 세션
            input_messages_key="query",  # 사용자의 실제 입력 쿼리
            history_messages_key="history" # 프롬프트의 히스토리 placeholder
        )

        # 룰 질문 답변 프롬프트 (룰 청크 검색 결과를 {context}로 받음)
        rule_question_prompt = ChatPromptTemplate.from_messages([
            (
                 "system",
                    "너는 보드게임 '뱅' 룰 전문 AI야. 반드시 아래 규칙을 따라야 해:\n"
                    "- 사용자와의 대화 기록을 참고해서 맥락을 이해하고 답변해.\n"
                    "- 사용자가 이전에 말한 역할(보안관, 부관, 무법자, 배신자)을 기억해.\n"
                    "- 아래 룰 설명에 있는 내용만 기반해서 정확하게 답변해.\n"
                    "- 룰 설명에 없는 정보는 절대로 지어내거나 상상하지 마.\n"
                    "- 전략 질문이면 룰북을 토대로 구체적인 전략을 제시해.\n"
                    "- 대화 맥락을 고려해서 일관된 답변을 해줘.\n"
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "아래는 '{game_name}' 보드게임의 룰 설명 일부입니다:\n\n{context}\n\n이 룰을 바탕으로 다음 질문에 정확하고 구체적으로 답변해줘:\n\n질문: {question}")
        ])

        # 룰 질문 답변 체인
        self.rule_question_chain = RunnableWithMessageHistory(
            rule_question_prompt | self.llm,
            get_session_history=get_session_history_for_gpt_rules,  # GPT 룰 전용 세션
            input_messages_key="question",
            history_messages_key="history"
        )

        # 룰 요약 프롬프트 (전체 룰 텍스트를 {game_rule_text}로 받음)
        rule_summary_prompt = ChatPromptTemplate.from_messages([
            (
                 "system",
                    "너는 보드게임 룰 전문 AI야. 반드시 아래 규칙을 따라야 해:\n"
                    "- 사용자의 질문에 대해 아래 전체 룰 설명에 있는 내용만 기반해서 답변해.\n"
                    "- 룰 설명에 없는 정보는 절대로 지어내거나 상상하지 마.\n"
                    "- 사람 이름, 장소, 시간, 인원수 등을 추측하거나 새로 만들어내지 마.\n"
                    "- 룰 북을 물어보는게 아닌 전략을 물어보면 너는 룰북을 토대로 전략을 짜줘.\n"
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "게임 이름: {game_name}\n\n룰 전체:\n{game_rule_text}\n\n이 게임의 룰을 설명해주세요.")
        ])

        # 룰 요약 체인
        self.rule_summary_chain = RunnableWithMessageHistory(
            rule_summary_prompt | self.llm,
            get_session_history=get_session_history_for_gpt_rules,  # GPT 룰 전용 세션
            input_messages_key="game_name", # 게임 이름이 주 입력값이 됨
            history_messages_key="history"
        )

    def _search_similar_context(self, query, top_k=3):
        """
        첫 번째 코드의 search_similar_context 함수와 동일한 RAG 검색 로직.
        쿼리를 임베딩하여 FAISS 인덱스에서 유사한 게임 설명을 찾습니다.
        """
        if not self.index or not self.texts or not self.game_names:
            logger.warning("RAG 검색을 위한 인덱스나 텍스트 데이터가 로드되지 않았습니다.")
            return ""

        query_vec = self.embed_model.encode([query], normalize_embeddings=True)
        D, I = self.index.search(np.array(query_vec), top_k)

        context_blocks = []
        for i in I[0]:
            if 0 <= i < len(self.game_names) and 0 <= i < len(self.texts):
                context_blocks.append(f"[{self.game_names[i]}]\n{self.texts[i]}")
            else:
                logger.warning(f"인덱스 {i}에 해당하는 게임 이름 또는 텍스트를 찾을 수 없습니다.")
        
        # 메모리 정리 - 검색 후 벡터 메모리 해제
        del query_vec, D, I
        gc.collect()
        
        return "\n\n".join(context_blocks)
    
    async def recommend_games(self, query: str, session_id: str = "default_session", top_k: int = 3):
        """게임 추천 (RAG 검색 후 LangChain으로 LLM 호출)"""
        try:
            # 쿼리에서 추천 개수 추출
            number_match = re.search(r'(\d+)\s*개', query)
            if number_match:
                top_k = int(number_match.group(1))

            # 1. RAG 검색: query를 기반으로 유사한 게임 설명을 가져옴 (첫 번째 코드의 핵심 로직)
            context = self._search_similar_context(query, top_k=top_k)
            
            if not context:
                return "추천할 게임 데이터를 찾을 수 없습니다. 인덱스나 데이터 로드를 확인해주세요."

            # 2. LangChain 체인 호출: 검색된 context와 사용자 쿼리를 LLM에 전달
            response = await self.recommendation_chain.ainvoke(
                {"query": query, "context": context},
                config={"configurable": {"session_id": session_id}}
            )
            
            raw_output = response.content
            
            # 3. 출력 후처리 - 간단하게 겵백 제거만
            return raw_output.strip()
            
        except Exception as e:
            logger.error(f"❌ 게임 추천 실패: {str(e)}")
            return f"게임 추천 중 오류가 발생했습니다: {str(e)}"
    
    async def answer_rule_question(self, game_name: str, question: str, session_id: str):
        """룰 질문 답변 (룰 청크 검색 후 LangChain으로 LLM 호출)"""
        try:
            # 🔍 세션 히스토리 디버깅 로그 추가
            if session_id in gpt_rule_store:
                history = gpt_rule_store[session_id]
                logger.info(f"🧠 세션 {session_id} 기존 메시지 수: {len(history.messages)}")
                for i, msg in enumerate(history.messages[-3:]):  # 최근 3개만 로그
                    logger.info(f"   - 메시지 {i}: {type(msg).__name__} - {str(msg)[:100]}...")
            else:
                logger.info(f"🆕 세션 {session_id} 새로 생성됨 (GPT 룰 스토어)")
            
            # 게임별 벡터 인덱스 로드
            game_index_path = os.path.join(self.game_vector_base_path, f"{game_name}.faiss")
            game_chunks_path = os.path.join(self.game_vector_base_path, f"{game_name}.json")
            
            if not os.path.exists(game_index_path) or not os.path.exists(game_chunks_path):
                return f"'{game_name}' 게임의 룰 데이터를 찾을 수 없습니다. 해당 게임의 데이터가 올바른 경로에 있는지 확인해주세요."
            
            # 벡터 인덱스 및 청크 텍스트 로딩
            index = faiss.read_index(game_index_path)
            with open(game_chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            
            # RAG 검색: 룰 질문에 대한 유사 청크 검색
            q_vec = self.embed_model.encode([question], normalize_embeddings=True)
            D, I = index.search(np.array(q_vec), k=4)
            retrieved_chunks = [chunks[i] for i in I[0] if i < len(chunks)]
            
            context = "\n\n".join(retrieved_chunks)
            logger.info(f"🔍 RAG 검색된 컨텍스트 길이: {len(context)} 글자")
            
            # RAG 검색 실패 or 관련 청크 없음
            if not context or context.strip() == "":
                logger.info(f"RAG 검색 실패. 전체 룰을 기반으로 재시도: {game_name}")
                return await self.get_rule_summary_answer(game_name, question, session_id)

            # LangChain 체인 호출
            logger.info(f"🔗 LangChain 체인 호출 시작 (세션: {session_id})")
            response = await self.rule_question_chain.ainvoke(
                {"game_name": game_name, "question": question, "context": context},
                config={"configurable": {"session_id": session_id}}
            )
            
            # 🔍 체인 호출 후 세션 상태 재확인
            if session_id in gpt_rule_store:
                history_after = gpt_rule_store[session_id]
                logger.info(f"🧠 체인 호출 후 세션 {session_id} 메시지 수: {len(history_after.messages)}")
            
            answer = response.content
            logger.info(f"✅ LangChain 답변 생성 완료 (길이: {len(answer)} 글자)")
            return answer.strip()
            
        except Exception as e:
            logger.error(f"❌ 룰 질문 답변 실패: {str(e)}")
            return f"룰 질문 답변 중 오류가 발생했습니다: {str(e)}"
    
    async def get_rule_summary(self, game_name: str, session_id: str):
        """게임 룰 요약 (전체 룰 텍스트를 LangChain으로 LLM 호출)"""
        try:
            # 게임 정보 찾기
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
            

            # LangChain 체인 호출
            response = await self.rule_summary_chain.ainvoke(
                {"game_name": game_name, "game_rule_text": game_rule_text},
                config={"configurable": {"session_id": session_id}}
            )
            
            summary = game_rule_text
            return summary.strip()
            
        except Exception as e:
            logger.error(f"❌ 룰 요약 실패: {str(e)}")
            return f"룰 요약 중 오류가 발생했습니다: {str(e)}"
        
    async def get_rule_summary_answer(self, game_name: str, question: str, session_id: str):
        """전체 룰을 기반으로 질문에 답변 (LangChain 세션 히스토리 적용)"""
        try:
            game_info = next((g for g in self.game_data if g.get("game_name") == game_name), None)
            if not game_info:
                return f"'{game_name}' 게임의 룰 데이터를 찾을 수 없습니다."

            game_rule_text = game_info.get("text", "")
            if not game_rule_text:
                return f"'{game_name}' 게임의 룰 텍스트가 없습니다."

            # 전체 룰 기반 질문 답변 프롬프트 (세션 히스토리 포함)
            full_rule_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "너는 보드게임 룰 전문 AI야. 반드시 아래 규칙을 따라야 해:\n"
                    "- 사용자의 질문에 대해 아래 전체 룰 설명에 있는 내용만 기반해서 답변해.\n"
                    "- 룰 설명에 없는 정보는 절대로 지어내거나 상상하지 마.\n"
                    "- 사람 이름, 장소, 시간, 인원수 등을 추측하거나 새로 만들어내지 마.\n"
                    "- 룰 북을 물어보는게 아닌 전략을 물어보면 너는 룰북을 토대로 전략을 짜줘.\n"
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "아래는 '{game_name}' 보드게임의 전체 룰 설명입니다:\n\n{game_rule_text}\n\n이 룰을 바탕으로 다음 질문에 정확하고 구체적으로 답변해줘:\n\n질문: {question}")
            ])

            # 전체 룰 기반 질문 답변 체인 (GPT 룰 전용 세션 사용)
            full_rule_chain = RunnableWithMessageHistory(
                full_rule_prompt | self.llm,
                get_session_history=get_session_history_for_gpt_rules,  # GPT 룰 전용 세션
                input_messages_key="question",
                history_messages_key="history"
            )

            response = await full_rule_chain.ainvoke({
                "game_name": game_name,
                "game_rule_text": game_rule_text,
                "question": question
            }, config={"configurable": {"session_id": session_id}})
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"❌ 전체 룰 기반 질문 처리 실패: {str(e)}")
            return f"전체 룰 기반 질문 처리 중 오류가 발생했습니다: {str(e)}"

        
    def get_available_games(self):
        """사용 가능한 게임 목록 반환"""
        if self.game_names:
            return self.game_names
        elif self.game_data:
            return [game.get("game_name", "") for game in self.game_data if game.get("game_name")]
        else:
            # 기본 게임 목록
            return [
                "카탄", "스플렌더", "아줄", "윙스팬", "뱅", 
                "킹 오브 도쿄", "7 원더스", "도미니언", "스몰 월드", "티켓 투 라이드"
            ]