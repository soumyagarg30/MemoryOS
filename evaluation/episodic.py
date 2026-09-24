"""Run with uv run python -m evaluation.episodic; inspect the JSON results."""

import asyncio
import json
from datetime import UTC, datetime
from uuid import uuid4

from memoryos.config import Settings
from memoryos.models import Memory, MemoryType
from memoryos.schemas.chat import AssistantReply
from memoryos.schemas.memory import MemoryCreate
from memoryos.services.chat import CHAT_PROMPT
from memoryos.services.conflicts import MemoryConflictDetector
from memoryos.services.extraction import MemoryExtractionService
from memoryos.services.ollama import OllamaClient


async def main():
    settings = Settings()
    client = OllamaClient(str(settings.ollama_base_url), timeout=settings.ollama_timeout)
    try:
        extraction = MemoryExtractionService(client, settings.extraction_model)
        result = await extraction.extract(
            "I visited Paris for a work conference on June 12, 2025. "
            "I visited Berlin for a holiday on August 4, 2025."
        )
        print(json.dumps({"extraction": result.model_dump(mode="json")}), flush=True)
        user_id = uuid4()
        episode = Memory(
            id=uuid4(), user_id=user_id,
            content="Visited Paris for a work conference on June 12, 2025.",
            memory_type=MemoryType.EPISODIC,
            source="evaluation", metadata_={"event_date": "2025-06-12"},
            created_at=datetime(2026, 9, 24, tzinfo=UTC),
        )
        detector = MemoryConflictDetector(client, settings.conflict_model)
        comparison = await detector.classify(
            MemoryCreate(
                user_id=user_id,
                content="Visited Berlin for a holiday on August 4, 2025.",
                memory_type=MemoryType.EPISODIC,
            ),
            [episode],
        )
        print(json.dumps({"distinct_events": comparison.model_dump(mode="json")}), flush=True)
        for message in ("When did I visit Paris?", "What is my sister's name?"):
            response = await client.parse(
                model=settings.chat_model,
                text_format=AssistantReply,
                max_output_tokens=512,
                input=[
                    {"role": "system", "content": CHAT_PROMPT},
                    {"role": "user", "content": json.dumps({
                        "message": message,
                        "memories": [{
                            "id": str(episode.id), "type": "EPISODIC",
                            "content": "Visited Paris for a work conference.",
                            "created_at": episode.created_at.isoformat(),
                            "metadata": episode.metadata_, "source": "evaluation",
                            "confidence": 0.95,
                        }],
                    })},
                ],
            )
            print(json.dumps({
                "question": message, "answer": response.output_parsed.model_dump(mode="json"),
            }), flush=True)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
