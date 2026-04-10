"""
config.py — LLM / ChromaDB / 공통 설정
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

load_dotenv()

# Hugging Face Hub: HF_TOKEN(또는 HUGGINGFACE_HUB_TOKEN)이 있으면 인증 다운로드로
# rate limit이 올라가고 대용량 가중치 받기가 훨씬 빨라짐(무료 계정 토큰으로 가능).
# https://huggingface.co/settings/tokens
_hf = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
if _hf:
    os.environ.setdefault("HF_TOKEN", _hf.strip())
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", _hf.strip())

# 선택: pip install hf_transfer 후 대용량 파일 고속 전송
try:
    import hf_transfer  # noqa: F401

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
except ImportError:
    pass

# ─────────────────────────────────────────
# LLM
# ─────────────────────────────────────────
def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """분석·초안 생성 등 정밀도가 필요한 곳에 사용"""
    return ChatOpenAI(
        model="gpt-4o",
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

def get_fast_llm(temperature: float = 0.0) -> ChatOpenAI:
    """Query Planning 등 가벼운 작업에 사용"""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


# ─────────────────────────────────────────
# ChromaDB (RAG)
# ─────────────────────────────────────────
# 다국어 지원 + 반도체 도메인에서 성능이 검증된 임베딩 모델 사용
# 후보: "BAAI/bge-m3" (다국어 BGE), "intfloat/multilingual-e5-large"
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL", "BAAI/bge-m3"
)

def get_chroma_client() -> chromadb.ClientAPI:
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    return chromadb.PersistentClient(path=persist_dir)

def get_collection(
    client: chromadb.ClientAPI,
    collection_name: str = "semiconductor_docs",
) -> chromadb.Collection:
    ef = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


# ─────────────────────────────────────────
# 분석 기본값
# ─────────────────────────────────────────
DEFAULT_TECHNOLOGIES = ["HBM4", "PIM", "CXL"]
DEFAULT_COMPANIES = ["Samsung", "Micron", "SK Hynix", "Broadcom", "TSMC"]
MAX_VALIDATION_RETRY = 2
MAX_FLOW_RETRY = 3

# 균형/MMR 튜닝은 환경변수 (retrieval_utils.py) — README 표 참고
