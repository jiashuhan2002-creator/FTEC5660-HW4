"""FTEC5660 Homework 4: the retrieval half of a RAG assistant over company reports.

Fill in the five functions marked ### YOUR CODE HERE. You may add imports, constants and helper
functions above them, but do not change the provided code below them (the models, the generator,
the loaders, the runner and the scoring).

    python3 hw4.py --data-folder public_test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
import traceback
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek

load_dotenv()


# =============================================================================
# Your code: add imports, constants and helper functions here.
# =============================================================================

import math
import re
from collections import Counter

from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")

# Words that carry no retrieval signal in a question about a report.
STOPWORDS = set("""
a an the of for in on at to from by with and or as is was were are be been being did do does
what which who whom whose when where why how much many this that these those it its their they them
company companies value amount number report reported hold held have has had end fiscal year years
usd million millions percent percentage please tell given give set makes make
""".split())

LEGAL_WORDS = {"inc", "corp", "corporation", "ltd", "limited", "plc", "llc", "co", "sa", "ag", "nv", "oyj", "spa"}

# Verbs and generic nouns: they say how the answer is phrased, not what is being asked.
# A word outside this set that appears nowhere in the company's documents means the corpus does
# not disclose the figure (Task 3).
SOFT_WORDS = set("""
generate generated generates announce announced announces announce spend spent spends forecast
forecasted forecasts report reported reports reporting hold held holds holding earn earned earns
make made makes set gives gave given give audit audited auditor firm company companies business
total value amount number end fiscal period statement statements financial result results review
largest higher more less many much large small new current annual full fourth quarter
city guide guided guidance describe explained see seen drive driven led lead
q1 q2 q3 q4 h1 h2 fy call earnings
""".split())

# Capitalised words that are never part of a company name in these questions.
NOT_COMPANY_WORDS = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "much", "many",
    "chief", "executive", "officer", "financial", "operating", "technology", "legal", "general",
    "counsel", "chair", "board", "director", "committee", "audit", "auditor", "independent",
    "asia", "pacific", "europe", "middle", "east", "africa", "americas", "north", "south", "latin",
    "fourth", "quarter", "full", "annual", "report", "call", "conference", "question", "answer",
    "united", "states", "kingdom", "hong", "kong", "china", "japan", "germany", "france",
    "fy", "q1", "q2", "q3", "q4", "esg", "ceo", "cfo", "coo", "cto",
}

# Names that look like a broker, an auditor or a data vendor rather than a company under review.
BROKER_WORDS = {"capital", "securities", "research", "equity", "partners", "advisors", "advisers",
                "bank", "banking", "investments", "markets", "asset", "management", "llp", "assurance"}

QUESTION_WORDS = {"what", "which", "who", "whom", "whose", "when", "where", "why", "how"}

# Filled while chunk_document runs, so parse_query can recognise the companies of the corpus.
CORPUS_COMPANY_KEYS: set[str] = set()
CORPUS_COMPANY_NAMES: dict[str, str] = {}

# Feature switches for the ablation in Task 5. The submitted run has all of them on.
ABLATION = {"filter": True, "boost": True, "per_company": True, "abstain": True}

# Section keywords, used to prefer the chunk that actually holds the figure.
SECTION_HINTS = {
    "receivable": ("accounts receivable", "note 4"),
    "borrow": ("borrowings", "note 5"),
    "employee": ("employees", "note 6"),
    "headcount": ("employees", "note 6"),
    "segment": ("revenue by segment", "note 2"),
    "region": ("revenue by region", "note 3"),
    "auditor": ("independent auditor", "audit"),
    "audit firm": ("independent auditor", "audit"),
    "chief financial": ("executive officers",),
    "chief executive": ("executive officers",),
    "cfo": ("executive officers",),
    "ceo": ("executive officers",),
    "board": ("board of directors",),
    "target price": ("target price", "valuation"),
    "rating": ("rating", "investment summary"),
    "repurchase": ("share repurchase", "capital"),
    "dividend": ("dividend",),
    "inventor": ("inventories", "balance sheet"),
    "cash flow": ("cash flow statement",),
    "total assets": ("balance sheet",),
    "total equity": ("balance sheet",),
    "revenue": ("income statement", "revenue"),
    "net income": ("income statement",),
    "gross profit": ("income statement",),
    "operating income": ("income statement",),
    "headquarter": ("headquarters",),
    "city": ("headquarters",),
}


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def _content_words(text: str) -> list[str]:
    words = []
    for word in _tokens(text):
        if word in STOPWORDS or len(word) < 2:
            continue
        if re.fullmatch(r"(?:fy)?\d{4}", word):  # years are handled by parse_query
            continue
        words.append(word)
    return words


def table_to_text(table_html: str) -> str:
    """Task 1.1: one HTML <table> -> one line per table row, each value with its row and column.

    The caption (which carries the unit) is reproduced verbatim on the first line, because the
    grader looks for it in the output.  Numbers are never reformatted: a value such as ``2,710.4``
    or ``(1,762.4)`` has to survive unchanged.
    """
    soup = BeautifulSoup(table_html, "lxml")
    table = soup.find("table") or soup

    lines: list[str] = []

    caption_tag = table.find("caption")
    caption = _clean(caption_tag.get_text(" ", strip=True)) if caption_tag else ""
    if caption:
        lines.append(caption)

    rows = table.find_all("tr")
    if not rows:
        return "\n".join(lines)

    # The header row: the last row of <thead>, or the first row when it is made of <th>.
    header_row = None
    thead = table.find("thead")
    if thead and thead.find_all("tr"):
        header_row = thead.find_all("tr")[-1]
    elif rows[0].find("th"):
        header_row = rows[0]

    headers: list[str] = []
    if header_row is not None:
        for cell in header_row.find_all(["th", "td"]):
            span = int(cell.get("colspan", 1) or 1)
            headers.extend([_clean(cell.get_text(" ", strip=True))] * span)

    for row in rows:
        if row is header_row:
            continue
        raw_cells = row.find_all(["td", "th"])
        if not raw_cells:
            continue

        # A section row ("Assets", "Current liabilities") spans the whole width and carries no value.
        if len(raw_cells) == 1 and int(raw_cells[0].get("colspan", 1) or 1) >= max(2, len(headers) - 1):
            label = _clean(raw_cells[0].get_text(" ", strip=True))
            if label:
                lines.append(label)
            continue

        cells: list[str] = []
        for cell in raw_cells:
            span = int(cell.get("colspan", 1) or 1)
            cells.extend([_clean(cell.get_text(" ", strip=True))] * span)

        label, values = cells[0], cells[1:]
        if not label and not any(values):
            continue

        parts = []
        for i, value in enumerate(values, start=1):
            if not value:
                continue
            column = headers[i] if i < len(headers) else ""
            parts.append(f"{column}: {value}" if column else value)

        if not parts:
            if label:
                lines.append(label)
            continue

        # Keep every line under the 400-character limit by splitting a wide row, repeating the label.
        current = label
        for part in parts:
            candidate = f"{current} | {part}" if current else part
            if len(candidate) > 380 and current != label:
                lines.append(current)
                current = f"{label} | {part}" if label else part
            else:
                current = candidate
        if current:
            lines.append(current)

    return "\n".join(line for line in lines if line)


def _document_type(plain: str) -> str:
    head = plain[:800].lower()
    if "earnings conference call" in head or "earnings call" in head:
        return "earnings_call"
    if "equity research" in head or "results review" in head:
        return "research_report"
    return "annual_report"


def _fiscal_year(plain: str, doc_type: str) -> int:
    head = plain[:800]
    patterns = [r"annual report\s+(\d{4})", r"(?:full year|fiscal year)\s+(\d{4})",
                r"\bfy\s*(\d{4})\b", r"\b(20\d{2})\b"]
    for pattern in patterns:
        match = re.search(pattern, head, re.I)
        if match:
            return int(match.group(1))
    return 2025


def _split_long(text: str, limit: int) -> list[str]:
    """Split one block (a long table, usually) into pieces of at most `limit` characters."""
    if len(text) <= limit:
        return [text]
    pieces, current = [], ""
    for line in text.splitlines():
        while len(line) > limit:  # a single pathological line
            if current:
                pieces.append(current)
                current = ""
            pieces.append(line[:limit])
            line = line[limit:]
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > limit and current:
            pieces.append(current)
            current = line
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def chunk_document(file_name: str, html: str) -> list[Document]:
    """Task 1.2: section-aware chunks with metadata, each at most MAX_CHUNK_CHARS characters.

    Every chunk repeats a short header (company, year, document type, section) so that a chunk of
    a balance sheet still says whose balance sheet and for which year it is.  Nothing but the
    document's own text and the output of table_to_text goes into the body.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav"]):
        tag.decompose()
    body = soup.body or soup

    h1 = body.find("h1")
    company = _clean(h1.get_text(" ", strip=True)) if h1 else ""
    company = re.sub(r"\s*\([^)]*\)\s*$", "", company).strip()
    if not company:
        company = _clean(file_name.replace("_", " ").replace(".html", ""))

    plain = _clean(body.get_text(" ", strip=True))
    doc_type = _document_type(plain)
    fiscal_year = _fiscal_year(plain, doc_type)

    CORPUS_COMPANY_KEYS.add(company_key(company))
    CORPUS_COMPANY_NAMES[company_key(company)] = company

    blocks: list[tuple[str, str]] = []
    state = {"section": ""}

    def walk(node) -> None:
        for child in node.children:
            name = getattr(child, "name", None)
            if name is None:
                continue
            if name in ("h1", "h2", "h3"):
                state["section"] = _clean(child.get_text(" ", strip=True))
            elif name == "table":
                text = table_to_text(str(child))
                if text:
                    blocks.append((state["section"], text))
            elif name in ("p", "blockquote"):
                text = _clean(child.get_text(" ", strip=True))
                if text:
                    blocks.append((state["section"], text))
            elif name in ("ul", "ol"):
                items = [_clean(li.get_text(" ", strip=True)) for li in child.find_all("li", recursive=False)]
                text = "\n".join(f"- {item}" for item in items if item)
                if text:
                    blocks.append((state["section"], text))
            elif name in ("div", "section", "article", "main", "header", "footer", "body"):
                walk(child)

    walk(body)

    chunks: list[Document] = []
    buffer_texts: list[str] = []
    buffer_section = None

    def emit() -> None:
        nonlocal buffer_texts, buffer_section
        if not buffer_texts:
            return
        header = f"{company} | FY{fiscal_year} | {doc_type.replace('_', ' ')}"
        if buffer_section:
            header = f"{header} | {buffer_section}"
        content = header + "\n" + "\n".join(buffer_texts)
        chunks.append(Document(page_content=content[:MAX_CHUNK_CHARS], metadata={
            "source": file_name,
            "company": company,
            "fiscal_year": fiscal_year,
            "doc_type": doc_type,
            "section": buffer_section or "",
        }))
        buffer_texts = []

    for section, text in blocks:
        header_len = len(company) + 30 + len(section or "")
        for piece in _split_long(text, max(400, MAX_CHUNK_CHARS - header_len - 5)):
            if section != buffer_section:
                emit()
                buffer_section = section
            used = header_len + sum(len(t) + 1 for t in buffer_texts)
            if buffer_texts and used + len(piece) + 1 > MAX_CHUNK_CHARS:
                emit()
                buffer_section = section
            buffer_texts.append(piece)
    emit()

    return chunks


def _known_companies(question: str) -> list[str]:
    """Every corpus company that the question names, matched on whole words."""
    words = set(_tokens(question))
    found = []
    for key, name in CORPUS_COMPANY_NAMES.items():
        parts = [w for w in key.split() if w]
        if parts and all(part in words for part in parts):
            found.append(name)
    return found


def _spans(question: str) -> list[str]:
    """Capitalised runs of two to four words, the shape of a company name in these questions."""
    pattern = re.compile(r"\b([A-Z][A-Za-z&.\-]*(?:\s+[A-Z][A-Za-z&.\-]*){1,3})")
    spans = []
    for match in pattern.finditer(question):
        span = _clean(match.group(1))
        words = span.split()
        # A run that starts with a question word is usually the start of a sentence, not a name.
        if words and words[0].lower() in QUESTION_WORDS:
            span = " ".join(words[1:])
            words = span.split()
        if len(words) < 2:
            continue
        lowered = [w.lower().strip(".,") for w in words]
        if any(w in NOT_COMPANY_WORDS for w in lowered):
            continue
        if any(w in BROKER_WORDS for w in lowered[-2:]):
            continue
        spans.append(span)
    return spans


def _companies(question: str) -> list[str]:
    names: dict[str, str] = {}
    for name in _known_companies(question):
        names.setdefault(company_key(name), name)
    for span in _spans(question):
        key = company_key(span)
        if not key or key in names:
            continue
        # Drop a span that is already contained in a longer one ("Northgate Robotics Inc" vs "Northgate Robotics").
        if any(key in other or other in key for other in names):
            if len(key) > max(len(o) for o in names):
                names = {k: v for k, v in names.items() if k not in key}
                names[key] = span
            continue
        names[key] = span
    return list(names.values())


def _year(question: str) -> int | None:
    match = re.search(r"\bFY\s?(\d{4})\b", question, re.I)
    if match:
        return int(match.group(1))
    match = re.search(r"(?:31\s+December|December\s+31|year ended)\s*,?\s*(\d{4})", question, re.I)
    if match:
        return int(match.group(1))
    match = re.search(r"\bin\s+(\d{4})\b", question, re.I)
    if match:
        return int(match.group(1))
    match = re.search(r"\b(20\d{2})\b", question)
    return int(match.group(1)) if match else None


def _doc_type(question: str) -> str:
    low = question.lower()
    if "earnings call" in low or "conference call" in low or "on the call" in low:
        return "earnings_call"
    # Whole words only: "operating income" must not be read as the word "rating".
    if re.search(r"\b(?:target price|price target|rating|ratings|forecast|forecasts|forecasted|"
                 r"estimate|estimates|results review|broker|analyst|12-month|investment rating)\b", low):
        return "research_report"
    return "annual_report"


def parse_query(question: str) -> dict:
    """Task 2.2: the companies, the fiscal year and the document type a question is about."""
    question = question.replace("\u2019", "'")
    return {"companies": _companies(question), "fiscal_year": _year(question), "doc_type": _doc_type(question)}


def build_index(chunks: list[Document]) -> Any:
    """Task 2: a lexical index (BM25) plus the metadata needed to filter before scoring."""
    texts = [chunk.page_content for chunk in chunks]
    tokens = [_tokens(text) for text in texts]
    bm25 = BM25Okapi(tokens) if tokens else None
    companies = {company_key((chunk.metadata or {}).get("company", "")) for chunk in chunks}
    # Every token that appears anywhere in a company's documents, for the abstention check.
    # Tokens are cut to six characters so that "borrowing", "borrowings" and "borrow" all match.
    company_tokens: dict[str, set[str]] = {}
    for chunk, chunk_tokens in zip(chunks, tokens):
        key = company_key((chunk.metadata or {}).get("company", ""))
        company_tokens.setdefault(key, set()).update(t[:6] for t in chunk_tokens)
    return {"chunks": chunks, "bm25": bm25, "tokens": tokens, "companies": companies,
            "company_tokens": company_tokens}


def _allowed_years(doc_type: str, year: int | None) -> set[int] | None:
    """Which report years can hold a figure of `year`: its own report and the next one's prior-year column."""
    if doc_type != "annual_report" or year is None:
        return None
    if year not in (2023, 2024, 2025):
        return set()
    years = {year}
    if year < 2025:
        years.add(year + 1)
    return years


def retrieve(index: Any, question: str) -> list[Document]:
    """Tasks 2 and 3: metadata-filtered BM25, per-company for comparisons, [] when unanswerable."""
    chunks = index["chunks"]
    bm25 = index["bm25"]
    if bm25 is None or not chunks:
        return []

    parsed = parse_query(question)
    query_keys = [company_key(name) for name in parsed["companies"]]
    known = index["companies"]
    resolved = [key for key in query_keys if key in known]
    flags = ABLATION

    # Task 3: a company the corpus does not contain can never be answered.
    if flags["abstain"] and (not query_keys or len(resolved) != len(query_keys)):
        return []

    years = _allowed_years(parsed["doc_type"], parsed["fiscal_year"])
    if flags["abstain"] and years is not None and not years:
        return []

    scores = bm25.get_scores(_tokens(question))
    qwords = set(_content_words(question))
    lowered = question.lower()
    hints = [phrase for keyword, phrases in SECTION_HINTS.items() if keyword in lowered for phrase in phrases]

    candidates: list[tuple[float, int]] = []
    for i, chunk in enumerate(chunks):
        meta = chunk.metadata or {}
        if flags["filter"]:
            if resolved and company_key(meta.get("company", "")) not in resolved:
                continue
            if meta.get("doc_type") != parsed["doc_type"]:
                continue
            if years is not None and meta.get("fiscal_year") not in years:
                continue

        text_low = chunk.page_content.lower()
        score = float(scores[i])

        if flags["boost"]:
            # Prefer the report of the asked year over the next one's prior-year column.
            if parsed["fiscal_year"] is not None and meta.get("fiscal_year") == parsed["fiscal_year"]:
                score += 1.2
            # Prefer the section the question is about.
            section = (meta.get("section") or "").lower()
            if hints and any(hint in section or hint in text_low[:400] for hint in hints):
                score += 1.0
            # A chunk that literally contains the question's own phrases is usually the right one.
            covered = sum(1 for word in qwords if word in text_low)
            if qwords:
                score += 1.5 * covered / len(qwords)
        candidates.append((score, i))

    if not candidates:
        return []

    candidates.sort(key=lambda pair: pair[0], reverse=True)

    if flags["abstain"]:
        # Abstention 1: a content word that appears nowhere in the company's documents means the
        # corpus does not disclose what is asked ("customer churn rate", "lobbying expenses").
        company_tokens: set[str] = set()
        for key in query_keys:
            company_tokens |= index.get("company_tokens", {}).get(key, set())
        undisclosed = [word for word in qwords - SOFT_WORDS if word[:6] not in company_tokens]
        if undisclosed:
            return []

        # Abstention 2: even when every word occurs somewhere, the best chunk may cover too little.
        best_coverage = 0.0
        for _, i in candidates[:60]:
            text_low = chunks[i].page_content.lower()
            if qwords:
                best_coverage = max(best_coverage, sum(1 for w in qwords if w in text_low) / len(qwords))
        if qwords and best_coverage < 0.55:
            return []
        if candidates[0][0] <= 0.0:
            return []

    picked: list[int] = []
    if flags["per_company"] and len(query_keys) >= 2:
        # A comparison needs evidence for each company, not five chunks of the larger one.
        per_company = max(2, MAX_RETRIEVED // len(query_keys))
        for key in query_keys:
            taken = 0
            for _, i in candidates:
                if taken >= per_company:
                    break
                if company_key(chunks[i].metadata.get("company", "")) == key and i not in picked:
                    picked.append(i)
                    taken += 1
    for _, i in candidates:
        if len(picked) >= MAX_RETRIEVED:
            break
        if i not in picked:
            picked.append(i)

    return [chunks[i] for i in picked[:MAX_RETRIEVED]]


# =============================================================================
# Provided code below. Do not change it.
# =============================================================================

LLM_MODEL = "deepseek-v4-flash"
DOC_TYPES = ("annual_report", "earnings_call", "research_report")

MAX_CHUNK_CHARS = 1500  # characters per chunk
MAX_RETRIEVED = 5       # chunks per question
TIME_LIMIT_SECONDS = 30 * 60

# USD per 1M tokens, only for the cost estimate that the runner prints.
PRICE_LLM_INPUT, PRICE_LLM_OUTPUT = 0.15, 0.60

# Points (Section 2 of the handout).
POINTS = {"1.1": 8, "1.2": 7, "2.1": 5, "2.2": 10, "2.3": 15, "3": 10, "4": 20}
RECALL_ZERO, RECALL_FULL = 0.50, 0.90  # Task 2.3: 0 points at or below RECALL_ZERO, full points from RECALL_FULL
MAX_LINE_CHARS = 400                   # Task 1.1
MIN_PROBES_FOUND = 0.9                 # Task 1.2


class UsageTracker(BaseCallbackHandler):
    """Counts LLM tokens, for the cost estimate. There is no budget."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.llm_calls = self.llm_input_tokens = self.llm_output_tokens = 0

    def on_llm_end(self, response, **kwargs) -> None:
        tokens_in = tokens_out = 0
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                tokens_in += usage.get("input_tokens", 0)
                tokens_out += usage.get("output_tokens", 0)
        with self._lock:
            self.llm_calls += 1
            self.llm_input_tokens += tokens_in
            self.llm_output_tokens += tokens_out

    def cost(self) -> float:
        return (self.llm_input_tokens * PRICE_LLM_INPUT + self.llm_output_tokens * PRICE_LLM_OUTPUT) / 1e6

    def summary(self) -> dict:
        return {"llm_calls": self.llm_calls, "llm_input_tokens": self.llm_input_tokens,
                "llm_output_tokens": self.llm_output_tokens, "cost_usd": round(self.cost(), 4)}


USAGE = UsageTracker()            # your calls
GENERATOR_USAGE = UsageTracker()  # the generator's calls


def _chat(usage: UsageTracker, temperature: float, thinking: bool, max_tokens: int | None) -> ChatDeepSeek:
    return ChatDeepSeek(
        model=LLM_MODEL,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=3,
        # Thinking mode ignores `temperature` and its tokens count as output.
        extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
        callbacks=[usage],
    )


def make_llm(temperature: float = 0.0, thinking: bool = False, max_tokens: int | None = None) -> ChatDeepSeek:
    """deepseek-v4-flash. Use it for every LLM call you make (only in parse_query, build_index and retrieve)."""
    return _chat(USAGE, temperature, thinking, max_tokens)


# ----------------------------------------------------------------------------- the generator

GENERATOR_PROMPT = """You answer questions about company reports, using only the context below.

Context:
{context}

Question: {question}

Rules:
- Use only the context. The header of each passage says which document it comes from.
- Check that the passage you use is about the company and the fiscal year that the question asks about.
- If the context does not contain the answer, reply exactly NOT_FOUND.
- For a number, reply with that one number only, in the unit the question asks for, without units, words or other numbers (for example: 1913.4).
- For a name (a person, a company, a city, an audit firm or a rating), reply with the name only.

Answer:"""

_generator: ChatDeepSeek | None = None


def format_context(chunks: list[Document]) -> str:
    """The retrieved chunks as the generator sees them: a header from the metadata, then the text."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata or {}
        header = " | ".join(f"{key}: {str(meta[key])[:100]}"
                            for key in ("source", "company", "fiscal_year", "doc_type") if meta.get(key) not in (None, ""))
        parts.append(f"[{i}] {header}\n{chunk.page_content}")
    return "\n\n".join(parts)


def generate(question: str, chunks: list[Document]) -> str:
    """The fixed generator: deepseek-v4-flash answers from the retrieved chunks only."""
    global _generator
    if not chunks:
        return "NOT_FOUND"
    if _generator is None:
        _generator = _chat(GENERATOR_USAGE, 0.0, False, 64)
    prompt = GENERATOR_PROMPT.format(context=format_context(chunks), question=question)
    return str(_generator.invoke(prompt).content).strip()


# ----------------------------------------------------------------------------- loaders

def load_corpus(folder: str | Path) -> dict[str, str]:
    """{file name: raw HTML} for every .html file in the folder."""
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(Path(folder).glob("*.html"))}


def load_json(path: str | Path) -> Any:
    """The parsed file, or None if it does not exist (the private set has no answers)."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- scoring

# One number: optional minus sign, optional thousands separators, optional decimals.
NUMBER_RE = re.compile(r"(?<![\d.,])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
TITLES = {"mr", "mrs", "ms", "dr", "prof", "professor", "sir"}
LEGAL_SUFFIXES = {"inc", "corp", "corporation", "ltd", "limited", "plc", "llc", "co"}


def parse_number(text: str) -> float | None:
    """The single number in the text, or None if there is not exactly one."""
    found = NUMBER_RE.findall(text.replace("\u2212", "-"))
    if len(found) != 1:
        return None
    return float(found[0].replace(",", ""))


def normalize(text: str) -> str:
    """Lower case, no punctuation, no titles, padded with spaces for whole-word matching."""
    text = unicodedata.normalize("NFKC", str(text)).lower().replace("_", " ")
    words = [w for w in re.sub(r"[^a-z0-9]+", " ", text).split() if w not in TITLES]
    return " " + " ".join(words) + " "


def company_key(name: Any) -> str:
    """A company name without case, punctuation or legal suffix: 'Northgate Robotics Inc.' -> 'northgate robotics'."""
    return " ".join(w for w in normalize(name).split() if w not in LEGAL_SUFFIXES)


def squash(text: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip().casefold()


def as_year(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def answer_correct(answer: str, gold: dict) -> bool:
    answer = str(answer or "")
    said_not_found = " not found " in normalize(answer)
    if gold["type"] == "not_found":
        return said_not_found
    if said_not_found:
        return False
    if gold["type"] == "number":
        value = parse_number(answer)
        if value is None:
            return False
        return abs(value - gold["answer"]) <= max(0.01 * abs(gold["answer"]), 0.05)
    names = [gold["answer"], *gold.get("aliases", [])]
    if not any(normalize(name) in normalize(answer) for name in names):
        return False
    return not any(normalize(other) in normalize(answer) for other in gold.get("reject", []))


def table_score(output: Any, table: dict) -> float:
    """Task 1.1, one table: the share of cells found on a line with their row and column labels and
    without the label of another row, halved if the caption is missing."""
    if not isinstance(output, str) or not table["cells"]:
        return 0.0
    lines = [squash(line) for line in output.splitlines()]
    lines = [line for line in lines if line and len(line) <= MAX_LINE_CHARS]
    labels = {squash(row) for row, _, _ in table["cells"]}

    def kept(row: str, col: str, value: str) -> bool:
        row, col, value = squash(row), squash(col), squash(value)
        others = [o for o in labels if o not in row]  # e.g. "total liabilities" may be in "total liabilities and equity"
        return any(row in line and col in line and value in line and not any(o in line for o in others)
                   for line in lines)

    share = sum(kept(*cell) for cell in table["cells"]) / len(table["cells"])
    return share if squash(table["caption"]) in squash(output) else share / 2


def document_problems(chunks: Any, error: str | None, meta: dict, file_name: str) -> list[str]:
    """Task 1.2, one document: why its chunks fail the checks (an empty list means they pass)."""
    if error:
        return [f"error: {error}"]
    if not isinstance(chunks, list) or not chunks:
        return ["no chunks"]
    if not all(isinstance(c, Document) for c in chunks):
        return ["not a list of Document"]
    problems = []
    longest = max(len(c.page_content) for c in chunks)
    if longest > MAX_CHUNK_CHARS:
        problems.append(f"a chunk has {longest} characters")
    for key, ok in (("source", lambda v: v == file_name),
                    ("company", lambda v: company_key(v) == company_key(meta["company"])),
                    ("fiscal_year", lambda v: as_year(v) == meta["fiscal_year"]),
                    ("doc_type", lambda v: v == meta["doc_type"])):
        wrong = [c.metadata.get(key) for c in chunks if not ok(c.metadata.get(key))]
        if wrong:
            problems.append(f"{key} {wrong[0]!r} (expected {meta.get(key, file_name)!r}) on {len(wrong)} chunk(s)")
    text = squash("\n".join(c.page_content for c in chunks))
    missing = [p for p in meta["probes"] if squash(p) not in text]
    if len(missing) > (1 - MIN_PROBES_FOUND) * len(meta["probes"]):
        problems.append(f"text missing from the chunks: {missing[:3]}")
    return problems


def parse_score(parsed: Any, gold: dict) -> float:
    """Task 2.2, one question: the share of the three fields that are right."""
    if not isinstance(parsed, dict):
        return 0.0
    expected = gold["query"]
    companies = parsed.get("companies")
    if isinstance(companies, str):
        companies = [companies]
    right = 0
    if isinstance(companies, (list, tuple)) and {company_key(c) for c in companies} == {company_key(c) for c in expected["companies"]}:
        right += 1
    if "fiscal_year" in parsed and as_year(parsed["fiscal_year"]) == expected["fiscal_year"] and \
            (parsed["fiscal_year"] is None) == (expected["fiscal_year"] is None):
        right += 1
    if parsed.get("doc_type") == expected["doc_type"]:
        right += 1
    return right / 3


def contains(chunk_text: str, printed: str) -> bool:
    """The fact as printed in the document is in the chunk (a number only as a whole number)."""
    text, printed = squash(chunk_text), squash(printed)
    if re.fullmatch(r"[\d,.%-]+", printed):
        return re.search(rf"(?<![\d.,]){re.escape(printed)}(?![\d]|[.,]\d)", text) is not None
    return printed in text


def retrieval_hit(chunks: list[Document], gold: dict) -> bool:
    """For every fact the question needs, a retrieved chunk of a document that states it contains it."""
    return bool(gold["evidence"]) and all(
        any(c.metadata.get("source") == file and contains(c.page_content, printed)
            for c in chunks for file, printed in group)
        for group in gold["evidence"])


def clip(x: float) -> float:
    return max(0.0, min(1.0, x))


# ----------------------------------------------------------------------------- runner

def main() -> None:
    parser = argparse.ArgumentParser(description="FTEC5660 HW4: retrieval for RAG over company reports")
    parser.add_argument("--data-folder", default="public_test",
                        help="folder with corpus/, questions.json and (optionally) the answer files")
    parser.add_argument("--output", default="results.csv")
    args = parser.parse_args()

    folder = Path(args.data_folder)
    corpus = load_corpus(folder / "corpus")
    questions = load_json(folder / "questions.json")
    ground_truth = load_json(folder / "ground_truth.json")
    docs_meta = load_json(folder / "docs.json")
    tables = load_json(folder / "tables.json")
    print(f"{len(corpus)} documents, {len(questions)} questions")
    start = time.perf_counter()
    points: dict[str, float] = {}

    # Task 1.1: tables (no model calls allowed)
    if tables:
        before = USAGE.llm_calls
        scores = []
        for table in tables:
            try:
                scores.append(table_score(table_to_text(table["html"]), table))
            except Exception as e:
                print(f"table_to_text failed on a table of {table['source']}: {e!r}")
                scores.append(0.0)
        if USAGE.llm_calls != before:
            print("WARNING: table_to_text called the LLM, which is not allowed (Task 1.1 = 0).")
            scores = [0.0]
        points["1.1"] = POINTS["1.1"] * sum(scores) / len(scores)
        print(f"\nTask 1.1 table_to_text: {sum(scores) / len(scores):.1%} of the table cells kept with their labels")

    # Task 1.2: chunks (no model calls allowed)
    before = USAGE.llm_calls
    chunks_by_file: dict[str, list] = {}
    errors: dict[str, str] = {}
    for file_name, html in corpus.items():
        try:
            chunks_by_file[file_name] = list(chunk_document(file_name, html) or [])
        except Exception as e:
            traceback.print_exc()
            chunks_by_file[file_name], errors[file_name] = [], repr(e)
    called_models = USAGE.llm_calls != before
    if called_models:
        print("WARNING: chunk_document called the LLM, which is not allowed (Task 1.2 = 0).")
    if docs_meta:
        problems = {f: document_problems(chunks_by_file.get(f), errors.get(f), m, f) for f, m in docs_meta.items()}
        passed = sum(not p for p in problems.values()) / len(problems)
        points["1.2"] = 0.0 if called_models else POINTS["1.2"] * passed
        print(f"Task 1.2 chunk_document: {passed:.1%} of the documents pass")
        for f, p in [(f, p) for f, p in problems.items() if p][:5]:
            print(f"  {f}: {'; '.join(p)}")

    # the chunks that retrieve() may return
    all_chunks = [c for f in sorted(chunks_by_file) for c in chunks_by_file[f]
                  if isinstance(c, Document) and len(c.page_content) <= MAX_CHUNK_CHARS]
    by_text: dict[str, list[Document]] = {}
    for c in all_chunks:
        by_text.setdefault(c.page_content, []).append(c)
    print(f"{len(all_chunks)} chunks to index")

    # Task 2.2: query parsing
    parsed: dict[str, Any] = {}
    for qid, question in questions.items():
        try:
            parsed[qid] = parse_query(question)
        except Exception as e:
            parsed[qid] = f"error: {e!r}"

    # Tasks 2.1, 2.3 and 3: retrieval
    index = build_index(all_chunks)
    index_seconds = time.perf_counter() - start
    print(f"Index built in {index_seconds:.1f}s | {USAGE.summary()}")
    retrieved: dict[str, list[Document]] = {}
    valid: dict[str, bool] = {}
    for qid, question in questions.items():
        try:
            result = retrieve(index, question)
            ok = isinstance(result, (list, tuple)) and len(result) <= MAX_RETRIEVED
            kept = []
            for chunk in list(result or []) if isinstance(result, (list, tuple)) else []:
                matches = by_text.get(getattr(chunk, "page_content", None), [])
                if not matches:
                    ok = False
                    continue
                source = (getattr(chunk, "metadata", None) or {}).get("source")
                kept.append(next((m for m in matches if m.metadata.get("source") == source), matches[0]))
            retrieved[qid], valid[qid] = kept[:MAX_RETRIEVED], ok
        except Exception:
            traceback.print_exc()
            retrieved[qid], valid[qid] = [], False
    retrieval_seconds = time.perf_counter() - start - index_seconds

    # Task 4: the fixed generator
    def answer(qid: str) -> str:
        try:
            return generate(questions[qid], retrieved[qid])
        except Exception as e:
            print(f"Generator failed on {qid}: {e!r}")
            return ""

    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = dict(zip(questions, pool.map(answer, questions)))
    total_seconds = time.perf_counter() - start

    rows = []
    for qid, question in questions.items():
        sources = sorted({c.metadata.get("source", "") for c in retrieved[qid]})
        row = {"id": qid, "category": "", "question": question, "parsed": json.dumps(parsed[qid], default=str),
               "parse_score": "", "retrieved": ";".join(sources), "valid": valid[qid], "retrieval_hit": "",
               "answer": answers[qid], "correct": "", "expected": ""}
        if ground_truth and qid in ground_truth:
            gold = ground_truth[qid]
            row["category"] = gold["category"]
            row["parse_score"] = round(parse_score(parsed[qid], gold), 3)
            row["retrieval_hit"] = retrieval_hit(retrieved[qid], gold) if gold["evidence"] else ""
            row["correct"] = answer_correct(answers[qid], gold)
            if not row["correct"]:
                row["expected"] = gold["answer"]
        rows.append(row)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")

    if ground_truth:
        graded = [r for r in rows if r["id"] in ground_truth]
        answerable = [r for r in graded if ground_truth[r["id"]]["evidence"]]
        unanswerable = [r for r in graded if not ground_truth[r["id"]]["evidence"]]
        parse_mean = sum(r["parse_score"] for r in graded) / len(graded)
        non_empty = sum(r["valid"] and bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        recall = sum(r["retrieval_hit"] for r in answerable) / max(len(answerable), 1)
        kept = sum(bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        rejected = sum(not r["retrieved"] for r in unanswerable) / max(len(unanswerable), 1)
        accuracy = sum(r["correct"] for r in graded) / len(graded)
        points["2.1"] = POINTS["2.1"] * non_empty
        points["2.2"] = POINTS["2.2"] * parse_mean
        points["2.3"] = POINTS["2.3"] * clip((recall - RECALL_ZERO) / (RECALL_FULL - RECALL_ZERO))
        points["3"] = POINTS["3"] * clip(kept + rejected - 1)
        points["4"] = POINTS["4"] * accuracy
        print(f"Task 2.1 valid, non-empty retrieval on answerable questions: {non_empty:.1%}")
        print(f"Task 2.2 parse_query fields right: {parse_mean:.1%}")
        print(f"Task 2.3 retrieval recall@{MAX_RETRIEVED} on answerable questions: {recall:.1%}")
        print(f"Task 3   answerable questions with chunks: {kept:.1%} | unanswerable questions with []: {rejected:.1%}")
        print(f"Task 4   answer accuracy: {accuracy:.1%}")
        by_category: dict[str, list[tuple]] = {}
        for r in graded:
            by_category.setdefault(r["category"], []).append((r["correct"], r["retrieval_hit"]))
        print("\nBy category       accuracy   recall")
        for category, results in sorted(by_category.items()):
            hits = [h for _, h in results if h != ""]
            recall_text = f"{sum(hits) / len(hits):.1%}" if hits else "-"
            print(f"  {category:<14}{sum(c for c, _ in results) / len(results):>9.1%}{recall_text:>9}")

    if points:
        print("\nEstimated points (the grade uses the private set):")
        for task in POINTS:
            if task in points:
                print(f"  Task {task:<4}{points[task]:6.2f} / {POINTS[task]}")
        print(f"  Total    {sum(points.values()):6.2f} / {sum(POINTS[t] for t in points)}")

    print(f"\nYour usage: {USAGE.summary()}")
    print(f"Generator usage: {GENERATOR_USAGE.summary()}")
    print(f"Time: {total_seconds:.1f}s (index {index_seconds:.1f}s, retrieval {retrieval_seconds:.1f}s)")
    if total_seconds > TIME_LIMIT_SECONDS:
        print(f"WARNING: the run took longer than the {TIME_LIMIT_SECONDS // 60}-minute limit.")


if __name__ == "__main__":
    main()
