"""
Generate synthetic financial Q&A training data from SEC 10-K filings.

Pipeline: SEC EDGAR download -> HTML/text parsing -> chunking -> Gemini Q&A
generation -> JSONL output in instruction-tuned format.

Usage:
    export GEMINI_API_KEY=...
    python generate_qa.py --tickers AAPL MSFT NVDA --limit 1 \
        --sec-user-agent "Your Name your.email@example.com"

    # Cheap smoke test without calling Gemini:
    python generate_qa.py --tickers AAPL --limit 1 --dry-run
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from bs4 import BeautifulSoup

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"

PROMPT_TEMPLATE = "### Context:\n{context}\n\n### Question:\n{question}\n\n### Answer:"

QA_GENERATION_PROMPT = """You are a financial analyst creating training data for a \
question-answering model. Given the excerpt below from a company's SEC 10-K filing, \
write {n} question-answer pairs that:

- Can be answered using ONLY the information in the excerpt (no outside knowledge)
- Cover concrete facts: numbers, dates, risks, financial metrics, named entities
- Vary in phrasing (some direct, some analytical)
- Have concise, factual answers (1-3 sentences)

Return ONLY a JSON array, no markdown fences, no commentary:
[{{"question": "...", "answer": "..."}}, ...]

Excerpt:
\"\"\"
{chunk}
\"\"\"
"""

BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*table of contents\s*$", re.IGNORECASE),
    re.compile(r"^\s*page \d+\s*$", re.IGNORECASE),
]


def download_filings(tickers, filing_type, limit, user_agent, raw_dir):
    from sec_edgar_downloader import Downloader

    raw_dir.mkdir(parents=True, exist_ok=True)
    name, email = user_agent.rsplit(" ", 1)
    dl = Downloader(name, email, raw_dir)

    downloaded = []
    for ticker in tickers:
        print(f"[download] {ticker}: fetching {limit} {filing_type} filing(s)...")
        dl.get(filing_type, ticker, limit=limit, download_details=True)
        ticker_dir = raw_dir / "sec-edgar-filings" / ticker / filing_type
        if ticker_dir.exists():
            downloaded.extend(sorted(ticker_dir.glob("*/primary-document.html")))
    return downloaded


def parse_filing_to_text(path: Path) -> str:
    raw = path.read_text(errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def is_boilerplate(chunk: str) -> bool:
    stripped = chunk.strip()
    if len(stripped) < 200:
        return True
    return any(p.match(stripped) for p in BOILERPLATE_PATTERNS)


def chunk_text(text: str, chunk_size: int, overlap: int):
    words = text.split()
    if not words:
        return []
    step = max(chunk_size - overlap, 1)
    chunks = []
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return [c for c in chunks if not is_boilerplate(c)]


def call_gemini(model_tuple, chunk: str, n_pairs: int, max_retries: int = 3):
    client, model_name = model_tuple
    prompt = QA_GENERATION_PROMPT.format(chunk=chunk, n=n_pairs)
    delay = 2
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.interactions.create(model=model_name, input=prompt)
            raw = resp.output_text.strip()
            raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
            pairs = json.loads(raw)
            return [
                p
                for p in pairs
                if isinstance(p, dict) and p.get("question") and p.get("answer")
            ]
        except Exception as e:
            print(f"  [gemini] attempt {attempt}/{max_retries} failed: {e}", file=sys.stderr)
            if attempt == max_retries:
                return []
            time.sleep(delay)
            delay *= 2
    return []


def mock_qa_pairs(chunk: str, n_pairs: int):
    return [
        {
            "question": f"[MOCK] What is discussed in this excerpt (pair {i + 1})?",
            "answer": f"[MOCK] {chunk[:80]}...",
        }
        for i in range(n_pairs)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="+", required=True, help="e.g. AAPL MSFT NVDA")
    parser.add_argument("--filing-type", default="10-K")
    parser.add_argument("--limit", type=int, default=1, help="filings per ticker")
    parser.add_argument(
        "--sec-user-agent",
        default=os.environ.get("SEC_USER_AGENT", ""),
        help='Required by SEC EDGAR, format: "Your Name your.email@example.com"',
    )
    parser.add_argument("--chunk-size", type=int, default=500, help="approx tokens per chunk")
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--qa-per-chunk", type=int, default=4)
    parser.add_argument("--max-chunks-per-filing", type=int, default=200)
    parser.add_argument("--max-pairs", type=int, default=5000)
    parser.add_argument("--gemini-model", default="gemini-3.5-flash-lite")
    parser.add_argument("--output", default=str(DATA_DIR / "train.jsonl"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip SEC download + Gemini calls; use mock data to validate the pipeline",
    )
    args = parser.parse_args()

    if not args.dry_run and not args.sec_user_agent:
        parser.error("--sec-user-agent is required (or set SEC_USER_AGENT) unless --dry-run")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = None
    if not args.dry_run:
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            parser.error("GEMINI_API_KEY environment variable is required unless --dry-run")
        client = genai.Client(api_key=api_key)
        model = (client, args.gemini_model)

    if args.dry_run:
        filing_paths = []
        raw_texts = [
            ("MOCK", " ".join(["revenue increased significantly this fiscal year"] * 600))
        ]
    else:
        raw_dir = DATA_DIR / "raw"
        filing_paths = download_filings(
            args.tickers, args.filing_type, args.limit, args.sec_user_agent, raw_dir
        )
        if not filing_paths:
            print("No filings downloaded; exiting.", file=sys.stderr)
            sys.exit(1)
        raw_texts = [(p.parent.name, parse_filing_to_text(p)) for p in filing_paths]

    total_pairs = 0
    with output_path.open("w") as out_f:
        for source_id, text in raw_texts:
            chunks = chunk_text(text, args.chunk_size, args.chunk_overlap)
            chunks = chunks[: args.max_chunks_per_filing]
            print(f"[chunk] {source_id}: {len(chunks)} usable chunks")

            for chunk in chunks:
                if total_pairs >= args.max_pairs:
                    break
                if args.dry_run:
                    pairs = mock_qa_pairs(chunk, args.qa_per_chunk)
                else:
                    pairs = call_gemini(model, chunk, args.qa_per_chunk)

                for pair in pairs:
                    record = {
                        "prompt": PROMPT_TEMPLATE.format(context=chunk, question=pair["question"]),
                        "completion": pair["answer"],
                    }
                    out_f.write(json.dumps(record) + "\n")
                    total_pairs += 1

            if total_pairs >= args.max_pairs:
                break

    print(f"[done] wrote {total_pairs} Q&A pairs to {output_path}")


if __name__ == "__main__":
    main()
