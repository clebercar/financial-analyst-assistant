"""Download de filings 10-K e 10-Q da SEC EDGAR.

Usa sec-edgar-downloader (limita por user-agent identificado, regra da SEC).

Estrutura de saida (sec-edgar-downloader 5.x):
    output_dir/sec-edgar-filings/<TICKER>/<FILING_TYPE>/<acc>/full-submission.txt
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from sec_edgar_downloader import Downloader

load_dotenv()
logger = logging.getLogger(__name__)

TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "META"]
FILING_TYPES = ["10-K", "10-Q"]


def build_filing_id(ticker: str, filing_type: str, year: str) -> str:
    """Identificador estavel: AAPL_10-K_2024."""
    return f"{ticker}_{filing_type}_{year}"


def download_filings(
    output_dir: Path = Path("data/filings"),
    tickers: list[str] | None = None,
    filing_types: list[str] | None = None,
    limit: int = 1,
) -> list[Path]:
    """Baixa ultimos `limit` filings por (ticker, type).

    Args:
        output_dir: pasta destino. sec-edgar-downloader cria
            `output_dir/sec-edgar-filings/<TICKER>/<TYPE>/<acc>/full-submission.txt`.
        tickers: lista de tickers (default: TICKERS).
        filing_types: lista de tipos (default: FILING_TYPES).
        limit: quantos filings mais recentes baixar por (ticker, tipo).

    Returns:
        Lista de paths dos full-submission.txt baixados.
    """
    if tickers is None:
        tickers = TICKERS
    if filing_types is None:
        filing_types = FILING_TYPES

    user_agent = os.getenv("SEC_USER_AGENT", "Financial Analyst Assistant contact@example.com")
    # Esperado formato "Nome Email"; partimos para passar separado pro Downloader 5.x
    parts = user_agent.split(" ", 1)
    company_name = parts[0] if parts else "Financial-Analyst-Assistant"
    email_address = parts[-1] if len(parts) > 1 else "contact@example.com"

    output_dir.mkdir(parents=True, exist_ok=True)
    dl = Downloader(company_name, email_address, str(output_dir))

    for ticker in tickers:
        for ft in filing_types:
            try:
                logger.info("Baixando %s %s (limit=%d)", ticker, ft, limit)
                dl.get(ft, ticker, limit=limit, download_details=False)
            except Exception as e:
                logger.warning("Falha em %s %s: %s", ticker, ft, e)

    paths = list(output_dir.rglob("full-submission.txt"))
    logger.info("Baixados %d filings", len(paths))
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    download_filings()
