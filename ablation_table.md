| Variant | Change | recall@5 | kept | rejected | gold in context |
|---|---|---|---|---|---|
| B0 | naive RAG (tutorial 7) | 26.1% | 100.0% | 0.0% | 40.0% |
| V1 | + table_to_text & section chunks | 84.8% | 100.0% | 0.0% | 66.7% |
| V2 | + metadata filter (query parsing) | 89.1% | 100.0% | 35.7% | 78.3% |
| V3 | + year/section/coverage boosts | 97.8% | 100.0% | 35.7% | 83.3% |
| V4 | + per-company quota | 100.0% | 100.0% | 35.7% | 85.0% |
| V5 | + abstention (submitted) | 100.0% | 100.0% | 100.0% | 100.0% |
