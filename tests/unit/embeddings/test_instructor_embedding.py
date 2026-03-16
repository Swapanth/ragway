from __future__ import annotations

from ragway.embeddings.instructor_embedding import InstructorEmbedding


async def test_instructor_embedding_returns_vectors() -> None:
    """Instructor adapter should return one vector per input text."""
    adapter = InstructorEmbedding(dimensions=7)
    vectors = await adapter.embed(["first", "second"])
    assert len(vectors) == 2
    assert all(len(vector) == 7 for vector in vectors)


async def test_instructor_embedding_instruction_changes_output() -> None:
    """Different instructions should produce different vectors for same text."""
    left = InstructorEmbedding(instruction="retrieve", dimensions=6)
    right = InstructorEmbedding(instruction="classify", dimensions=6)

    left_vector = (await left.embed(["same text"]))[0]
    right_vector = (await right.embed(["same text"]))[0]

    assert left_vector != right_vector

