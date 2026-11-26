"""Task 5: the ablation ladder B0 -> V5 on public_test.

Each row changes exactly one thing against the row above it:

  B0  naive RAG          BeautifulSoup get_text() (no table_to_text), RecursiveCharacterTextSplitter,
                         BM25 top-5 over the whole corpus, no query parsing, never returns []
  V1  + table_to_text    the submitted chunker: caption + one line per row, section-aware chunks
  V2  + metadata filter  parse_query() narrows company / document type / report year before scoring
  V3  + ranking boosts   year, section and question-word coverage are added to the BM25 score
  V4  + per-company      a comparison gets at most 2 chunks per company before the global fill
  V5  + abstention       [] for unknown companies, uncovered years and undisclosed figures

Run:  python3 ablation.py
"""
import json
from pathlib import Path

import hw4
from hw4 import Document, company_key, contains, normalize, parse_query, retrieval_hit
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup

FOLDER = Path("public_test")
corpus = hw4.load_corpus(FOLDER / "corpus")
questions = hw4.load_json(FOLDER / "questions.json")
gt = hw4.load_json(FOLDER / "ground_truth.json")


def naive_chunks() -> list[Document]:
    """B0: no table handling, fixed-size splitting, metadata still required by Task 1.2."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for file_name, html in corpus.items():
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav"]):
            tag.decompose()
        h1 = soup.find("h1")
        company = hw4._clean(h1.get_text(" ", strip=True)) if h1 else file_name
        company = hw4.re.sub(r"\s*\([^)]*\)\s*$", "", company)
        plain = hw4._clean(soup.get_text(" ", strip=True))
        doc_type = hw4._document_type(plain)
        year = hw4._fiscal_year(plain, doc_type)
        for piece in splitter.split_text(plain):
            chunks.append(Document(page_content=piece, metadata={
                "source": file_name, "company": company, "fiscal_year": year, "doc_type": doc_type}))
    return chunks


def build_variant_chunks(use_tables: bool) -> list[Document]:
    if not use_tables:
        return naive_chunks()
    return [c for f, html in corpus.items() for c in hw4.chunk_document(f, html)]


def evaluate(chunks, flags: dict, with_generator: bool = False) -> dict:
    hw4.ABLATION = dict(flags)
    index = hw4.build_index(chunks)
    retrieved = {qid: hw4.retrieve(index, q) for qid, q in questions.items()}
    answerable = [q for q in questions if gt[q]["evidence"]]
    unanswerable = [q for q in questions if not gt[q]["evidence"]]
    recall = sum(retrieval_hit(retrieved[q], gt[q]) for q in answerable) / len(answerable)
    kept = sum(bool(retrieved[q]) for q in answerable) / len(answerable)
    rejected = sum(not retrieved[q] for q in unanswerable) / len(unanswerable)

    def in_context(qid: str) -> bool:
        gold, chunks_ = gt[qid], retrieved[qid]
        if gold["type"] == "not_found":
            return not chunks_
        names = [gold["answer"], *gold.get("aliases", [])]
        for chunk in chunks_:
            if gold["type"] == "number":
                if contains(chunk.page_content, str(gold["answer"])) or \
                        contains(chunk.page_content, f"{gold['answer']:,}"):
                    return True
            elif any(normalize(n) in normalize(chunk.page_content) for n in names):
                return True
        return False

    hits = sum(in_context(q) for q in questions)
    result = {"recall": recall, "kept": kept, "rejected": rejected,
              "task3": max(0.0, kept + rejected - 1.0), "in_context": hits / len(questions),
              "chunks": len(chunks)}

    if with_generator:
        try:
            correct = sum(hw4.answer_correct(hw4.generate(q, retrieved[qid]), gt[qid])
                          for qid, q in questions.items())
            result["accuracy"] = correct / len(questions)
        except Exception as error:  # no key, or the API is down
            print(f"  (generator unavailable: {error!r})")
    return result


def main() -> None:
    import sys

    with_generator = "--with-generator" in sys.argv
    rows = []
    flags_none = {"filter": False, "boost": False, "per_company": False, "abstain": False}
    flags_filter = {"filter": True, "boost": False, "per_company": False, "abstain": False}
    flags_boost = {"filter": True, "boost": True, "per_company": False, "abstain": False}
    flags_company = {"filter": True, "boost": True, "per_company": True, "abstain": False}
    flags_all = {"filter": True, "boost": True, "per_company": True, "abstain": True}

    plan = [
        ("B0", "naive RAG (tutorial 7)", False, flags_none),
        ("V1", "+ table_to_text & section chunks", True, flags_none),
        ("V2", "+ metadata filter (query parsing)", True, flags_filter),
        ("V3", "+ year/section/coverage boosts", True, flags_boost),
        ("V4", "+ per-company quota", True, flags_company),
        ("V5", "+ abstention (submitted)", True, flags_all),
    ]

    for name, label, use_tables, flags in plan:
        hw4.CORPUS_COMPANY_NAMES.clear()
        hw4.CORPUS_COMPANY_KEYS.clear()
        chunks = build_variant_chunks(use_tables)
        result = evaluate(chunks, flags, with_generator)
        result.update({"variant": name, "change": label})
        rows.append(result)
        accuracy = f"  accuracy {result['accuracy']:.1%}" if "accuracy" in result else ""
        print(f"{name:3s} recall {result['recall']:.1%}  kept {result['kept']:.1%}  "
              f"rejected {result['rejected']:.1%}  in-context {result['in_context']:.1%}  "
              f"chunks {result['chunks']}{accuracy}")

    Path("ablation_results.json").write_text(json.dumps(rows, indent=2))

    has_accuracy = any("accuracy" in r for r in rows)
    if has_accuracy:
        header = ("| Variant | Change | recall@5 | kept | rejected | answer accuracy | gold in context |\n"
                  "|---|---|---|---|---|---|---|")
    else:
        header = "| Variant | Change | recall@5 | kept | rejected | gold in context |\n|---|---|---|---|---|---|"
    lines = [header]
    for r in rows:
        row = (f"| {r['variant']} | {r['change']} | {r['recall']:.1%} | {r['kept']:.1%} | "
               f"{r['rejected']:.1%} | ")
        if has_accuracy:
            row += f"{r['accuracy']:.1%} | " if "accuracy" in r else "— | "
        row += f"{r['in_context']:.1%} |"
        lines.append(row)
    table = "\n".join(lines)
    Path("ablation_table.md").write_text(table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
