"""Local evaluation without the LLM: Tasks 1.1, 1.2, 2.1, 2.2, 2.3, 3 and a proxy for Task 4."""
import json
from pathlib import Path

import hw4
from hw4 import (Document, table_score, document_problems, parse_score, retrieval_hit,
                 contains, company_key, normalize, MAX_RETRIEVED)

folder = Path("public_test")
corpus = hw4.load_corpus(folder / "corpus")
questions = hw4.load_json(folder / "questions.json")
gt = hw4.load_json(folder / "ground_truth.json")
docs_meta = hw4.load_json(folder / "docs.json")
tables = hw4.load_json(folder / "tables.json")

# Task 1.1
scores = []
print("=== Task 1.1 tables ===")
for t in tables:
    out = hw4.table_to_text(t["html"])
    s = table_score(out, t)
    scores.append(s)
    if s < 1.0:
        print(f"  {t['source']:14s} {s:.2f}  cells={len(t['cells'])}  {t['caption'][:50]}")
print(f"mean = {sum(scores)/len(scores):.3f}")

# Task 1.2
chunks_by_file, errors = {}, {}
for f, html in corpus.items():
    try:
        chunks_by_file[f] = hw4.chunk_document(f, html)
    except Exception as e:
        chunks_by_file[f], errors[f] = [], repr(e)
problems = {f: document_problems(chunks_by_file.get(f), errors.get(f), m, f) for f, m in docs_meta.items()}
passed = sum(not p for p in problems.values())
print(f"=== Task 1.2 ===\npassed {passed}/{len(problems)}")
for f, p in problems.items():
    if p:
        print(f"  {f}: {'; '.join(p)[:160]}")

all_chunks = [c for f in sorted(chunks_by_file) for c in chunks_by_file[f]
              if isinstance(c, Document) and len(c.page_content) <= hw4.MAX_CHUNK_CHARS]
print(f"chunks {len(all_chunks)}  max len {max(len(c.page_content) for c in all_chunks)}")

# Task 2.2
parse_total = 0.0
bad_parse = []
for qid, q in questions.items():
    p = hw4.parse_query(q)
    s = parse_score(p, gt[qid])
    parse_total += s
    if s < 1:
        bad_parse.append((qid, s, p, gt[qid]["query"]))
print(f"=== Task 2.2 ===\nmean {parse_total/len(questions):.3f}")
for qid, s, p, e in bad_parse:
    print(f"  {qid} {s:.2f}\n     got {p}\n     exp {e}")

# index / retrieve
index = hw4.build_index(all_chunks)
by_text = {}
for c in all_chunks:
    by_text.setdefault(c.page_content, []).append(c)

retrieved, valid = {}, {}
for qid, q in questions.items():
    res = hw4.retrieve(index, q)
    kept = []
    for c in res:
        matches = by_text.get(c.page_content, [])
        kept.append(matches[0] if matches else c)
    retrieved[qid] = kept
    valid[qid] = isinstance(res, list) and len(res) <= MAX_RETRIEVED

answerable = [q for q in questions if gt[q]["evidence"]]
unanswerable = [q for q in questions if not gt[q]["evidence"]]
recall = sum(retrieval_hit(retrieved[q], gt[q]) for q in answerable) / len(answerable)
kept = sum(bool(retrieved[q]) for q in answerable) / len(answerable)
rejected = sum(not retrieved[q] for q in unanswerable) / len(unanswerable)
nonempty = sum(bool(retrieved[q]) for q in answerable) / len(answerable)

# proxy for Task 4: is the gold answer present in what the generator would see?
def proxy_ok(qid):
    gold = gt[qid]
    chunks = retrieved[qid]
    if gold["type"] == "not_found":
        return not chunks
    names = [gold["answer"], *gold.get("aliases", [])]
    for c in chunks:
        if gold["type"] == "number":
            if contains(c.page_content, f"{gold['answer']}") or contains(c.page_content, f"{gold['answer']:,}"):
                return True
        else:
            if any(normalize(n) in normalize(c.page_content) for n in names):
                return True
    return False

proxy = sum(proxy_ok(q) for q in questions) / len(questions)
print(f"=== Tasks 2.1 / 2.3 / 3 ===")
print(f"2.1 non-empty on answerable : {nonempty:.1%}")
print(f"2.3 recall@5                : {recall:.1%}")
print(f"3   kept {kept:.1%} | rejected {rejected:.1%} | task3 = {max(0, kept+rejected-1):.3f}")
print(f"Task4 proxy (gold in context): {proxy:.1%}")

pts = {
    "1.1": 8 * sum(scores) / len(scores),
    "1.2": 7 * passed / len(problems),
    "2.1": 5 * nonempty,
    "2.2": 10 * parse_total / len(questions),
    "2.3": 15 * max(0, min(1, (recall - 0.5) / 0.4)),
    "3": 10 * max(0, kept + rejected - 1),
}
print("\npoints:", {k: round(v, 2) for k, v in pts.items()}, "sum", round(sum(pts.values()), 2))

print("\nmisses:")
for qid in answerable:
    if not retrieval_hit(retrieved[qid], gt[qid]):
        print(f"  {qid} {gt[qid]['category']:10s} {questions[qid][:80]}")
        print(f"        got {[c.metadata['source'] for c in retrieved[qid]]} exp {[e[0][0] for e in gt[qid]['evidence']]}")
print("\nwrongly kept:")
for qid in unanswerable:
    if retrieved[qid]:
        print(f"  {qid} {questions[qid][:90]} -> {[c.metadata['source'] for c in retrieved[qid]]}")
