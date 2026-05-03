"""Pipeline RAG: chunking, embedding (Gemini) e busca vetorial (ChromaDB).

Responsabilidade:
1. `chunk_text`: quebrar texto longo em pedacos com overlap.
2. `_clean_html`: remover lixo SGML/HTML do submission da SEC.
3. `_gemini_embed`: gerar embedding via Gemini gemini-embedding-001.
4. `get_collection`: factory do ChromaDB persistente.
5. `retrieve`: busca top-k chunks por similaridade.
6. `index_filings`: pipeline batch para indexar todos os filings em data/filings.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import chromadb
import google.generativeai as genai
import yaml  # type: ignore[import-untyped]
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "sec_filings"

# Free tier Gemini: ~15 req/min. 4s de delay deixa margem.
EMBED_DELAY_SECONDS = 4.0
# Em caso de 429 (rate limit), backoff de 60s.
RATE_LIMIT_BACKOFF_SECONDS = 60.0
MAX_RETRIES_PER_CHUNK = 3


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Chunking simples por numero de caracteres (proxy de tokens).

    chunk_size e overlap sao em "tokens"; convertemos para chars usando 1 token ~= 4 chars.
    """
    chunk_size_chars = chunk_size * 4
    overlap_chars = overlap * 4
    if len(text) <= chunk_size_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size_chars
        chunks.append(text[start:end])
        start = end - overlap_chars
    return chunks


def _clean_html(raw: str) -> str:
    """Remove tags HTML/SGML, XBRL e ruido do submission da SEC.

    SEC full-submission.txt mistura o documento principal com XBRL,
    anexos e tabelas financeiras estruturadas que poluem o RAG.
    """
    # Remove blocos XBRL (ix:..., us-gaap:..., dei:..., xbrli:...) inteiros
    text = re.sub(
        r"<(ix|us-gaap|dei|xbrli|xbrldi|link|xlink|xsi):[^>]*>.*?</\1:[^>]+>",
        " ",
        raw,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Remove tags simples remanescentes
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove sequências longas de números/datas (tabelas financeiras)
    text = re.sub(r"(\b[\d,.\-$()%]+\s+){8,}", " ", text)
    # Remove entidades HTML
    text = re.sub(r"&[a-zA-Z]+;|&#\d+;", " ", text)
    # Normaliza whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _gemini_embed(
    text: str,
    model: str = "models/gemini-embedding-001",
    task_type: str = "retrieval_document",
) -> list[float]:
    """Embedding via Gemini. Levanta RuntimeError se nao houver API key."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY nao definida no .env")
    genai.configure(api_key=api_key)
    result = genai.embed_content(model=model, content=text, task_type=task_type)
    return result["embedding"]


def _embed_with_retry(text: str, max_retries: int = MAX_RETRIES_PER_CHUNK) -> list[float] | None:
    """Embedding com retry/backoff em caso de rate limit (429)."""
    for attempt in range(max_retries):
        try:
            return _gemini_embed(text)
        except Exception as e:
            msg = str(e).lower()
            is_rate = "429" in msg or "quota" in msg or "rate" in msg
            if is_rate and attempt < max_retries - 1:
                logger.warning(
                    "Rate limit atingido, aguardando %ds (tentativa %d/%d)",
                    RATE_LIMIT_BACKOFF_SECONDS,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(RATE_LIMIT_BACKOFF_SECONDS)
                continue
            logger.warning("Embed falhou apos %d tentativas: %s", attempt + 1, e)
            return None
    return None


def get_collection() -> Any:
    """Factory do ChromaDB persistente; cria a colecao se nao existir."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(COLLECTION_NAME)


def retrieve(
    query: str,
    collection: Any,
    embed_fn: Callable[[str], list[float]],
    top_k: int = 3,
    ticker_filter: str | None = None,
) -> list[dict]:
    """Busca top_k chunks relevantes para a query.

    Retorna lista de dicts com chaves: ticker, tipo, ano, secao, trecho, distance.
    """
    embedding = embed_fn(query)
    where = {"ticker": ticker_filter} if ticker_filter else None
    res = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        where=where,
    )
    chunks: list[dict] = []
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    for doc, meta, dist in zip(docs, metas, dists, strict=False):
        chunks.append(
            {
                "ticker": meta.get("ticker"),
                "tipo": meta.get("filing_type"),
                "ano": meta.get("year"),
                "secao": meta.get("section", "N/A"),
                "trecho": doc,
                "distance": dist,
            }
        )
    return chunks


def index_filings(max_chunks_per_filing: int | None = 30) -> int:
    """Le filings em data/filings, faz chunking + embedding + indexa no Chroma.

    Args:
        max_chunks_per_filing: limite de chunks por filing (None = sem limite).
            Como SEC full-submission e enorme, limitamos para os primeiros N
            chunks (onde costuma estar Risk Factors e MD&A — partes mais uteis
            para analise). Sem isso, free tier do Gemini levaria horas.

    Returns:
        numero total de chunks indexados.
    """
    config_path = Path("configs/model_config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f).get("rag", {})
    else:
        cfg = {}
    chunk_size = cfg.get("chunk_size", 800)
    chunk_overlap = cfg.get("chunk_overlap", 100)

    collection = get_collection()
    base = Path("data/filings")
    files = list(base.rglob("full-submission.txt"))
    logger.info("Encontrados %d filings em %s", len(files), base)

    chunk_idx = 0
    for fp in files:
        # path: data/filings/sec-edgar-filings/<TICKER>/<TYPE>/<acc>/full-submission.txt
        parts = fp.parts
        try:
            ticker = parts[-4]
            filing_type = parts[-3]
            accession = parts[-2]
        except IndexError:
            logger.warning("Path inesperado: %s", fp)
            continue

        raw = fp.read_text(errors="ignore")
        clean = _clean_html(raw)
        all_chunks = chunk_text(clean, chunk_size, chunk_overlap)
        chunks = all_chunks[:max_chunks_per_filing] if max_chunks_per_filing else all_chunks
        logger.info(
            "%s %s -> %d chunks (de %d totais)", ticker, filing_type, len(chunks), len(all_chunks)
        )

        for c in chunks:
            emb = _embed_with_retry(c)
            if emb is None:
                continue
            chunk_idx += 1
            collection.add(
                ids=[f"{ticker}_{filing_type}_{accession}_{chunk_idx}"],
                documents=[c],
                metadatas=[
                    {
                        "ticker": ticker,
                        "filing_type": filing_type,
                        "year": "2024",
                        "accession": accession,
                    }
                ],
                embeddings=[emb],
            )
            # Respeitar rate limit do free tier
            time.sleep(EMBED_DELAY_SECONDS)

    logger.info("Indexado total: %d chunks", chunk_idx)
    return chunk_idx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reindex", action="store_true", help="re-indexa filings em data/filings")
    parser.add_argument(
        "--limit-chunks-per-filing",
        type=int,
        default=30,
        help="maximo de chunks por filing (default 30; usar 0 para sem limite)",
    )
    args = parser.parse_args()
    if args.reindex:
        limit = args.limit_chunks_per_filing or None
        index_filings(max_chunks_per_filing=limit)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
