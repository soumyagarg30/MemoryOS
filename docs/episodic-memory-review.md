# Episodic memory review

Reviewed 2026-09-24. The deterministic pipeline has substantial automated
coverage, but this is not certification of the model's semantic accuracy.

## Correctness fix

Chat previously discarded a retrieved episode's metadata and recording timestamp
before requesting an answer. An event date or context stored only in metadata
could therefore be unavailable to the model. Chat now supplies those fields,
source, and confidence. A regression test exercises storage, retrieval, and the
actual model request. The prompt distinguishes recording time from event time
and tells the model to respect corrections in the current message.

## Covered behavior

Existing tests check per-user candidate isolation, lifecycle and expiration
filters, distinct conflict classifications, confidence gates, minimum three-source
consolidation, pairwise similarity, provenance, preservation of source episodes,
repeat-run exclusion, and rollback on provider or persistence failures.
These tests substitute model responses; they establish application behavior,
not whether a real model labels every event correctly.

## Remaining limitations

1. Chat has no conversation history beyond recalled memory. Extraction processes
   only the current message, so pronouns and short follow-ups may lack context.
2. No event timestamp or reference timezone is captured by extraction. Relative
   dates remain ambiguous; an episode's creation date is not its occurrence date.
3. Retrieval lacks a relevance cutoff and can recall unrelated nearest neighbors.
4. Consolidation searches within bounded pages. Related episodes split between
   pages can fail to form a cluster.
5. A consolidated summary is not automatically invalidated when a source is
   later superseded. Provenance exists, but dependency revalidation does not.
6. Consolidation and decay are explicit operations, not autonomous chat behavior.

## Validation

On 2026-09-24, the backend suite completed with 273 passed and one PostgreSQL
integration test skipped (no test database configured). Ruff, the frontend lint
check, the frontend production build, and frozen offline dependency sync passed.

The opt-in live check `uv run python -m evaluation.episodic` also completed using
the locally configured Ollama models, with synthetic data and no database writes:

| Scenario | Observed result |
| --- | --- |
| Two dated trips | Two stored-intent EPISODIC candidates with dates preserved |
| Paris work trip versus later Berlin holiday | CONTEXT_SPECIFIC; both can coexist |
| Event date supplied only in metadata | Answered 2025-06-12, not the recording date |
| Unknown sister's name | Answered "I do not know." |

Both extracted events received maximum importance and confidence. This small
sample does not establish score calibration. The live check exercises extraction,
conflict classification, and reply generation; database retrieval and consolidation
were covered by mocked tests, not this live-model run.

Run `uv run pytest` and `uv run ruff check .`. PostgreSQL-specific integration
coverage additionally requires `MEMORYOS_TEST_DATABASE_URL` pointing to a
pgvector-enabled database. The frontend is checked with `npm run build` and
`npm run lint` in `frontend/`; no frontend test cases are currently included.

For a live-model evaluation, use a separate test user and dated, distinct episodes,
then verify date-specific recall, coexistence of different events, correction of
one event, abstention on unknown personal facts, and consolidation provenance.
Do not interpret passing mocked tests as a successful live-model evaluation.
