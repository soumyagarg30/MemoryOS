# MemoryOS

Memory archive and chat application with a React/TypeScript frontend and a
FastAPI backend using Python 3.11+, SQLAlchemy 2, PostgreSQL 17,
pgvector, Alembic, and Pydantic v2. Includes structured LLM extraction, semantic
conflict detection during storage, episodic consolidation, decay, and ranked retrieval. Missing memory-content
embeddings and query embeddings are generated using the configured embedding model.

## Local setup

Install Python, uv, and Docker with Compose, then run from the repository root:

```sh
cp .env.example .env
uv sync --extra dev
docker compose up -d --wait db
uv run alembic upgrade head
uv run uvicorn memoryos.main:app --reload
```

API: http://127.0.0.1:8000/health
Interactive documentation: http://127.0.0.1:8000/docs

Install and start Ollama with the models listed under Memory Extraction below
before using chat, extraction, or memory writes. The default chat model is
`qwen2.5:7b`, configurable through `MEMORYOS_CHAT_MODEL`.

### Frontend

With Node.js 22+ installed, run in a second terminal:

```sh
cd frontend
npm ci
API_PROXY_TARGET=http://127.0.0.1:8000 npm run dev
```

Open the local URL printed by Vite (normally http://127.0.0.1:5173).
The explicit proxy target matches the backend command above; the frontend's
default proxy target is port 8001. See `frontend/.env.example` for overrides.
For a production bundle, run `npm run build` from `frontend/` and serve `dist/`
with an `/api` reverse proxy to the backend.

### Chat and episodic memory

`POST /chat` accepts `user_id` (UUID) and `message`. It retrieves up to five
memories, generates a reply, then extracts and stores up to ten candidates.
The response includes recalled, created, reinforced, and superseded memories,
activity events, and warnings for failed recall or storage. A reply alone does
not confirm that storage succeeded. Chat recall includes recording timestamps,
metadata, source, and confidence so episode context reaches the model.

An `EPISODIC` memory represents an event; consolidation derives a `SEMANTIC`
summary from at least three eligible related episodes while preserving the
originals and their provenance. Consolidation and decay require explicit API
calls or maintenance commands; chat does not schedule them automatically.

Known behavior limits:

- Chat receives the current message and recalled memories, not the full prior
  conversation. Follow-ups that depend on unstored conversation can lose context.
- Extraction has no reference date or user timezone. Relative dates such as
  "yesterday" are not resolved, and `created_at` is recording time, not event time.
- Retrieval ranks the nearest candidates without a minimum relevance threshold;
  unrelated records can reach the model. Its answer must still be grounded.
- Consolidation only compares episodes within each 100-record page and does not
  revalidate derived summaries when a source is subsequently superseded.
- Model accuracy is not certified by mocked tests. See
  [the episodic-memory review](docs/episodic-memory-review.md) for verification scope.

This is a local-development application without authentication. UUID ownership
filters are not access control; add authentication and authorization before
exposing the API to other users. Keep `.env`, installed dependencies, and local
database contents out of Git; example configuration and lockfiles are included.

`GET /health` returns `200` with `{"status":"ok"}`. This is a process liveness
check and deliberately does not query PostgreSQL or assert database readiness.

## Layout

```text
memoryos/
  api/          HTTP routes and routing
  models/       SQLAlchemy model registry
  schemas/      Pydantic request/response contracts
  services/     Storage, conflict detection, extraction, retrieval, decay, and consolidation
  memory/       Model prompts, clustering, retrieval ranking, and decay policy
  commands/     Periodic maintenance commands
  db/           Declarative base, engine, request-scoped sessions
  config.py     Environment settings
  main.py       Application factory and lifespan
alembic/        Migration environment and versioned migrations
frontend/       React archive UI, chat, and memory inspection
docs/           Review notes and known limitations
tests/          Endpoint tests
compose.yaml    Local PostgreSQL with pgvector and persistent storage
```

## Database lifecycle

Compose creates the database and role on the first start of an empty volume.
Alembic initializes the schema and enables `vector` through the first migration.
The PostgreSQL installation must include pgvector; the Compose image provides it.
The migration role needs permission to create extensions. In production, a DBA
can provision the extension first and use a separate migration role.

Schema creation is explicit: application startup never runs `create_all()` or
migrations. Run `alembic upgrade head` once as a deployment step before workers
start. The initial downgrade leaves the extension installed because it may be
shared; removing it is a separate administrative operation.

For future models, inherit from `memoryos.db.base.Base`, import the model in
`memoryos/models/__init__.py`, then generate and review a migration:

```sh
uv run alembic revision --autogenerate -m "describe schema change"
uv run alembic upgrade head
```

Use `DbSession` from `memoryos.db.session` in synchronous route handlers. Each
request gets its own session; services explicitly commit successful writes.
Uncommitted transactions are rolled back when the session closes. The engine
checks pooled connections before use and disposes its pool on shutdown.

## Memory API

Apply the new migration with `uv run alembic upgrade head` before using these routes.

| Method | Path | Result |
| --- | --- | --- |
| POST | `/memories` | Create (201) or reinforce an existing memory (200); Location header in both cases |
| GET | `/memories` | List memories, newest first; 200 |
| GET | `/memories/{id}` | Fetch one memory; 200 or 404 |
| DELETE | `/memories/{id}` | Permanently delete; 204 or 404 |

Example request body:

```json
{
  "user_id": "342568d5-38d1-4cf0-9b6a-e1735edc06b4",
  "content": "Prefers concise answers",
  "memory_type": "PREFERENCE",
  "importance": 0.8,
  "confidence": 1.0,
  "source": "user",
  "metadata": {"topic": "communication"}
}
```

Memory types: `WORKING`, `EPISODIC`, `SEMANTIC`, `PREFERENCE`, `TASK`.
States: `ACTIVE` (default), `STALE`, `ARCHIVED`, `SUPERSEDED`.
IDs and user IDs are UUIDs. Importance and confidence are finite numbers in
`[0, 1]`, defaulting to `0.5` and `1.0`. Content cannot be blank. Unknown creation
fields are rejected. `expires_at` must include a timezone when supplied.

Optional fields include `embedding`, `expires_at`, `source`, `superseded_by`,
and `metadata` (defaults to `{}`). Embeddings are stored as pgvector `vector`
with no fixed dimension until an embedding model is selected; supplied vectors
must contain 1 to 16,000 finite float32-compatible values with a nonzero norm.
If omitted, an embedding is generated before conflict detection. Metadata is JSONB;
the ORM attribute is `metadata_` because SQLAlchemy reserves `metadata`.
See the [pgvector SQLAlchemy integration](https://github.com/pgvector/pgvector-python#sqlalchemy).

The list route accepts optional `user_id`, `memory_type`, and `status` filters,
plus `limit` (1-100, default 50) and `offset` (default 0). It returns a JSON array
with stable ordering by creation time and ID. All states and expired memories
remain visible unless filtered. This scaffold has no authentication; `user_id`
is a filter and stored ownership field, not an authorization mechanism.

Creation generates `id`, `created_at`, and `updated_at`; `access_count` starts
at zero and `last_accessed_at` is null. The CRUD GET routes do not mutate access
counters; semantic retrieval updates the selected memories. Contradiction detection
automatically supersedes old memories during creation. SQLAlchemy updates `updated_at`
on subsequent ORM writes; retrieval leaves that content timestamp unchanged.
A supplied `superseded_by` must reference an existing memory for the same user.
Deleting a referenced replacement or a consolidation source/summary returns 409
to preserve those relationships. `consolidated_at` is an output-only timestamp
marking an episodic memory already incorporated into a semantic summary.

## Conflict Detection

`POST /memories` requires a running Ollama server with the configured models installed.
`MEMORYOS_CONFLICT_MODEL` defaults to `qwen2.5:7b` and must support structured
output. The detector sends Pydantic JSON schemas to Ollama's native chat API.
All model calls are mocked in tests.

For a new active, unexpired memory with no explicit `superseded_by`, storage
searches the top 20 cosine matches belonging to the same user. Existing candidates
must also be active, unexpired, and not superseded. All memory types are eligible;
the LLM decides whether semantic similarity actually represents a relationship.
Each comparison includes the existing memory ID, a relationship enum, and a reason:

| Relationship | Database action |
| --- | --- |
| `REINFORCEMENT` | Raise existing confidence by 0.05, capped at 1.0; do not insert a duplicate |
| `CONTRADICTION` | Insert the new memory, mark conflicting records `SUPERSEDED`, and point their `superseded_by` to it |
| `CONTEXT_SPECIFIC` | Keep existing memories and insert the new memory with its context intact |
| `UNRELATED` | Keep existing memories and insert the new memory |

Reinforcement requires the same fact in the same context, without distinct added
information. The prompt preserves differences such as home/work preferences,
historical periods, and additional compatible facts. Content, type, source,
metadata, and temporal qualifiers are provided to the classifier. Memory text is
explicitly treated as data rather than instructions.

If several records reinforce the new memory, all receive the confidence increment.
The closest reinforcing record (UUID breaks ties) is returned as the canonical
record. If other records contradict it, they are superseded by that canonical
record without inserting a duplicate. Existing content, metadata, importance,
embeddings, and access counters remain unchanged on reinforcement. The response
retains the existing `MemoryResponse` shape: 200 for reinforcement, 201 for insertion.

Historical imports (`STALE`, `ARCHIVED`, `SUPERSEDED`, already expired, or explicitly
linked to a successor) are stored without replacing current facts. Null, zero,
or dimension-incompatible legacy embeddings cannot be searched; existing data is
not automatically backfilled. Supplied vectors must use the configured model's
embedding space. Detection is limited to the 20 retrieved candidates and the
classifier's semantic accuracy.

PostgreSQL acquires a transaction-scoped advisory lock per user before searching,
including when no candidates exist, and locks selected rows through classification
and commit. A competing write receives 409 and can be retried. Model failure,
refusal, incomplete output, missing/duplicate/foreign IDs, or invalid labels abort
the entire write. Confidence changes, insertion, and supersession commit together;
no partial changes remain after a failure. Locks are held during the model call,
so writes for one user are serialized while different users can proceed independently.
No new database migration is required.

Provider timeouts return 504; rate limits and database availability failures
return 503; invalid/provider output returns 502; refusals and invalid embeddings
return 422; concurrent writes, expired candidates during classification, and
integrity conflicts return 409. Reads, deletes, and health checks remain usable
without a running Ollama server.

## Memory Extraction

`POST /memory/extract` accepts `{"message": "I prefer concise answers."}` and
returns `{"candidates": [...]}`. Every candidate includes `content`, `memory_type`,
`importance`, `confidence`, `should_store`, and `reason`. Scores are finite numbers
between 0 and 1. Messages must be nonblank and at most 20,000 characters.

Start Ollama locally and install the models:

```bash
ollama serve  # unnecessary if the Ollama application is already running
ollama pull qwen2.5:7b
ollama pull nomic-embed-text:latest
```

`MEMORYOS_OLLAMA_BASE_URL` defaults to `http://localhost:11434`; no API key is needed.
The model defaults to `qwen2.5:7b`; override it with `MEMORYOS_EXTRACTION_MODEL`.
The native [Ollama chat API](https://docs.ollama.com/api/chat) receives a Pydantic
JSON schema, and responses are validated before use. Requests are non-streaming,
with temperature zero, a 4,096-token output cap, no automatic retries, and a
120-second timeout configurable through `MEMORYOS_OLLAMA_TIMEOUT`.
There is no OpenAI dependency or fallback. Health checks do not contact Ollama.

The prompt instructs the model to return no candidates for greetings, thanks,
and other trivial conversation, and to mark unhelpful or uncertain candidates
`should_store=false`. These decisions are made by the LLM; schema validation
enforces field types and score bounds. Rejected candidates remain visible in
the result for inspection. No candidate is persisted, including those marked
`should_store=true`, and the endpoint does not use a database session.
Requests go to the configured Ollama server; extraction does not persist candidates.

Missing configuration or rate limiting returns 503; provider timeouts return
504; invalid/incomplete output and other provider errors return 502; refusals
return 422. Read/delete and health endpoints remain available when Ollama is offline. Tests mock the
LLM client, require no key, and prohibit database access during extraction.

## Memory Retrieval

`POST /memory/retrieve` embeds the query, selects up to 20 memories using
[pgvector cosine distance](https://github.com/pgvector/pgvector#querying), reranks
that candidate set, and returns the best five by default:

```json
{
  "user_id": "342568d5-38d1-4cf0-9b6a-e1735edc06b4",
  "query": "How does this user prefer to communicate?",
  "limit": 5,
  "debug": true
}
```

`limit` is optional and accepts integers from 1 to 20. `query` must be nonblank
and at most 8,000 characters; the embedding provider also applies token limits.
The response is `{"memories": [{"memory": {...}, "score": 0.85, "components": {...}}]}`.
`components` is included only when `debug` is true. Scores use these exact weights:

| Component | Weight | Calculation |
| --- | --- | --- |
| `semantic_similarity` | 0.50 | `1 - cosine_distance`, bounded to [-1, 1] for rounding |
| `importance` | 0.20 | Stored importance, from 0 to 1 |
| `recency` | 0.15 | `2 ** (-age_days / 30)`, using `created_at`; future dates count as age zero |
| `access_frequency` | 0.10 | Access count divided by the maximum count among the 20 candidates; zero if all counts are zero |
| `confidence` | 0.05 | Stored confidence, from 0 to 1 |

The final score is the weighted sum, not a probability; negative cosine similarity
can produce a negative score. Equal final scores use UUID ordering for stable ties.
Debug components describe the values before access tracking is updated. Returned
memory objects contain the updated counters. Only selected memories are updated,
using atomic SQL increments and a nondecreasing `last_accessed_at` in one transaction.
If the database write fails, the transaction rolls back and the request returns 503.

Candidates must belong to the requested user, have `ACTIVE` status, have no
`superseded_by` reference, and be unexpired. All memory types are eligible.
Null/zero embeddings and vectors with incompatible dimensions are excluded before
cosine evaluation. Fewer than `limit` matches produce a shorter list; no matches
produce `{"memories": []}`. Candidate selection uses exact search on the existing
variable-dimension column; no approximate vector index is added.

The query embedder uses the same local Ollama server. Its model defaults to
`nomic-embed-text:latest`; override it with `MEMORYOS_EMBEDDING_MODEL` and optionally
`MEMORYOS_EMBEDDING_DIMENSIONS` for models supporting custom dimensions. The
integration uses the native [Ollama embedding API](https://docs.ollama.com/api/embed)
with truncation disabled. Overlong inputs are rejected rather than silently truncated.
Stored embeddings must come from the same model and dimensionality as query
embeddings. The current schema does not record model provenance, so matching
dimensions alone cannot enforce that requirement. Retrieval does not generate
or backfill memory-content embeddings. Regenerate any existing OpenAI embeddings
using the configured Ollama model before retrieval. No database migration is needed.

Missing API configuration returns 503; embedding timeouts return 504; provider
rate limiting returns 503; rejected input returns 422; other embedding failures
return 502. Existing access fields are untouched on embedding failures.

## Memory Consolidation

Apply migration `0003` before using the updated application:

```sh
uv run alembic upgrade head
```

`POST /memory/consolidate` explicitly consolidates one user's episodic memories:

```json
{
  "user_id": "342568d5-38d1-4cf0-9b6a-e1735edc06b4"
}
```

The response contains `memories` (each with a semantic `memory` and its
`source_memory_ids`), `candidates_examined`, `clusters_skipped`, and `next_cursor`.
For another page, send the returned cursor as `after_id` with the same user ID.

Eligibility requires an `EPISODIC` memory in `ACTIVE` or `STALE` state, no successor,
no prior consolidation, no elapsed expiration, a nonzero embedding, and confidence
at least 0.80. Other users' memories are never candidates. The service uses pgvector
cosine comparisons and forms deterministic, disjoint clusters of 3-20 memories.
Every pair must reach cosine similarity 0.85; a chain of similar neighbors is not
enough. Different embedding dimensions cannot be clustered together.

Only qualifying clusters are sent to the LLM using a Pydantic structured-output
schema. It proposes a semantic statement of at most 500 characters, confidence,
a consolidation decision, and a short reason. It is instructed to retain context,
avoid unsupported generalizations, and decline incoherent or conflicting clusters.
The proposal must also meet the 0.80 confidence threshold. Declined and low-confidence
proposals leave their sources unchanged. These semantic judgments depend on the
model; structural validation and similarity/confidence gates are enforced in code.

Successful proposals become `SEMANTIC` memories with generated embeddings,
mean source importance, confidence capped at the least-confident source, and the
earliest non-null source expiration. Sources keep their content, embeddings,
confidence, access history, and lifecycle states. Their `consolidated_at` timestamps
mark them as processed; later runs exclude them. No source record is deleted.

The `memory_consolidation_sources` table stores foreign-key links from each source
to its semantic memory. A source can be consolidated only once. Both source and
summary deletion are restricted while these links exist. Source IDs are also
included in the summary's metadata for inspection through the existing GET API.

Each request examines up to 100 eligible memories in UUID order and compares them
within that page. Clusters crossing page boundaries can be missed; this bounded,
greedy workflow is not a global optimal clustering algorithm. Stored embeddings
must use a compatible model and embedding space. Existing semantic memories are
not merged by this workflow; consolidation creates a derived summary per cluster.

The service shares the storage flow's per-user PostgreSQL advisory lock and locks
candidate rows. The entire page's summaries, source markers, and provenance commit
together. A provider, embedding, or database failure rolls back all changes in that
request, including earlier clusters. A competing write or a source expiring during
generation returns 409. Configuration/provider/database errors use the same
503/504/502 conventions as the other model-backed endpoints; model refusal returns 422.

Configuration defaults:

```dotenv
MEMORYOS_CONSOLIDATION_MODEL=qwen2.5:7b
MEMORYOS_CONSOLIDATION_SIMILARITY=0.85
MEMORYOS_CONSOLIDATION_MIN_CONFIDENCE=0.8
```

Consolidation uses the same Ollama server and embedding settings. It is not run
automatically during ordinary memory creation. Tests use mocked model responses
and cover clustering, both confidence gates, provenance, preservation, idempotency,
user isolation, pagination, and transactional rollback.

## Memory Decay

Run the maintenance command from the project root:

```sh
uv run python -m memoryos.commands.decay
uv run python -m memoryos.commands.decay --dry-run
uv run python -m memoryos.commands.decay --at 2026-09-22T12:00:00Z --dry-run
```

The decay score is `original_importance * 2 ** (-inactive_days / half_life_days)`.
Inactivity is measured from `last_accessed_at`, falling back to `created_at` for
never-accessed memories. Fractional days are preserved; future timestamps count
as zero inactivity. Original `importance` is never overwritten, so repeated runs
do not compound decay. The policy is defined in `memoryos/memory/decay.py`:

| Memory type | Half-life |
| --- | --- |
| WORKING | 6 hours |
| TASK | 7 days |
| EPISODIC | 30 days |
| SEMANTIC | 180 days |
| PREFERENCE | 365 days |

Scores at or above 0.20 remain active, scores from 0.05 up to 0.20 become stale,
and scores below 0.05 become archived. Low original importance can therefore
cause a transition even for a recent memory. Expired memories become archived
regardless of score. A run can move directly from active to archived if enough
time has passed. Stale memories are not automatically reactivated. Archived and
superseded records, and records linked through `superseded_by`, are excluded.

The service changes only lifecycle status and `updated_at`. It never deletes
records or changes content, importance, confidence, or access history. Existing
retrieval already excludes stale and archived memories. No migration, API key,
or LLM call is required.

The command emits a JSON summary with `processed`, `stale`, `archived`, and
`unchanged` counts. `--dry-run` reports proposed changes without writing;
`--user-id UUID` limits scope. `--at` requires an explicit timezone and defaults
to one timestamp captured at the start of the run. `--batch-size` defaults to 500
and accepts 1-10,000. Use the service with a dedicated session because it owns
the transaction boundaries.

PostgreSQL row locks protect each batch from simultaneous access updates or
conflict resolution. Busy rows are skipped and retried on the next scheduled
run. Each batch commits independently; a failing batch rolls back and the command
exits with status 1, while earlier batches remain committed. Repeating the job is
safe. Exit status 0 indicates success; invalid command arguments return 2.

An external scheduler can run it hourly; for example, adapt this cron entry:

```cron
0 * * * * cd /absolute/path/to/MemoryOS && .venv/bin/python -m memoryos.commands.decay
```

No scheduler is installed automatically. Tests use fixed timestamps for all
decay calculations, boundary transitions, dry runs, idempotency, and rollback.

## Configuration

Settings read `.env` and environment variables (environment takes precedence).
`MEMORYOS_DATABASE_URL` must use `postgresql+psycopg://`.
`MEMORYOS_DEBUG` and `MEMORYOS_SQL_ECHO` default to false.

The supplied credentials are for local development. Configure deployment
credentials through environment variables or a secret manager. If changing
Compose credentials or port, update `MEMORYOS_DATABASE_URL` to match. Compose
initialization variables only apply to a new database volume. The database port
is bound to localhost. Use the hostname `db` when connecting from a container
on the same Compose network.

## Verification

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run alembic upgrade head --sql
```

Health tests need no running database and fail if a connection is attempted.
The default CRUD tests use an isolated SQLite database with a test-only JSONB
compiler. Retrieval and conflict tests mock model calls and substitute SQLite functions
for pgvector to test filtering, candidate limits, ranking, and transactional
updates. Separate unit tests verify score formulas and PostgreSQL SQL compilation.
Conflict tests cover all four relationships, mixed classifications, multiple
matches, user isolation, confidence caps, exact ID validation, expiration races,
provider errors, rollback, and PostgreSQL locking statements. Mocked labels verify
the application policy, not the quality of the live model's semantic judgments.
These tests do not certify the live pgvector implementation. Offline migration
tests check PostgreSQL SQL generation.

To also run the database-backed suites on PostgreSQL and test migration upgrade/downgrade
and model/schema consistency, provide a dedicated pgvector-enabled test database:

```sh
MEMORYOS_TEST_DATABASE_URL=postgresql+psycopg://memoryos:memoryos@localhost:5432/memoryos uv run pytest
```

PostgreSQL tests run in unique schemas inside rolled-back transactions; the
test role needs schema and extension creation privileges. Without this variable,
the PostgreSQL migration test is skipped.

The database image follows the [official pgvector distribution](https://github.com/pgvector/pgvector).
Migration wiring follows the [Alembic tutorial](https://alembic.sqlalchemy.org/en/latest/tutorial.html).
