# `corpus_mini` — synthetic Phase 3.1 corpus fixture

Two short markdown documents shaped like the real Phase 3.1 corpus
(RACGP Red Book CVD chapter + NVDPA absolute-CVD-risk quick
reference) plus a [`sources.json`](sources.json) manifest the
`build_corpus.py --use-fixture` driver reads.

The wording is **paraphrased and lightly invented** for testing
purposes. Nothing in these files is clinical advice; they exist so
the chunkers can be exercised end-to-end without redistributing
copyrighted RACGP / NVDPA text and without requiring network access
in CI.

The line-exact marker `<!-- page break -->` between sections of each
document drives the multi-page parse path in
[`cardiorisk.rag.ingest.parse.parse_markdown_fixture`](../../../cardiorisk/rag/ingest/parse.py),
so the parser produces a `ParsedDoc` with two pages per fixture
file. This matters for the chunkers' page-range tests and for the
eval-set's `expected_page_range` validation.

The seed retrieval questions in
[`eval/retrieval/questions.jsonl`](../../../../eval/retrieval/questions.jsonl)
target paragraphs of these fixtures by `expected_doc_id` and
`expected_span_keywords`. Keep the prose stable across edits — if you
have to change a paragraph, update the corresponding seed question
in the same PR.
