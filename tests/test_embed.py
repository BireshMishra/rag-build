import json

import pytest
from chromadb import EmbeddingFunction

from rag_build import embed


class FakeEmbedding(EmbeddingFunction):
    """Deterministic offline embedder that counts how many texts it was asked to embed."""

    def __init__(self):
        self.calls = 0  # intentionally skips super().__init__, which is a no-op

    def __call__(self, input):
        self.calls += len(input)
        return [[float(len(text)), float(sum(map(ord, text)) % 97), 1.0] for text in input]

    @staticmethod
    def name():
        return "fake-embedding"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return FakeEmbedding()

    def default_space(self):
        return "l2"

    def supported_spaces(self):
        return ["l2"]


def write_chunks(folder, file_name, texts):
    path = folder / f"dataset_{file_name}.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for chunk_id, text in enumerate(texts, 1):
            row = {"file": file_name, "page": 1, "chunk_id": chunk_id, "chunk": text}
            handle.write(json.dumps(row) + "\n")
    return path


@pytest.fixture
def fake():
    return FakeEmbedding()


@pytest.fixture
def collection(tmp_path, fake):
    return embed.get_collection(str(tmp_path / "db"), "test", embedding_function=fake)


def test_read_chunks_builds_ids_and_metadata(tmp_path):
    path = write_chunks(tmp_path, "a", ["one", "two"])
    ids, documents, metadatas = embed.read_chunks(path)
    assert ids == ["a:1:1", "a:1:2"]
    assert documents == ["one", "two"]
    assert metadatas[1] == {"file": "a", "page": 1, "chunk_id": 2}


def test_save_embeddings_skips_chunks_already_stored(tmp_path, collection, fake):
    write_chunks(tmp_path, "a", ["one", "two", "three"])
    embed.save_embeddings(collection, str(tmp_path))
    assert collection.count() == 3
    first_run_calls = fake.calls

    embed.save_embeddings(collection, str(tmp_path))
    assert collection.count() == 3
    assert fake.calls == first_run_calls


def test_save_embeddings_embeds_only_new_chunks(tmp_path, collection, fake):
    write_chunks(tmp_path, "a", ["one", "two"])
    embed.save_embeddings(collection, str(tmp_path))
    calls_before = fake.calls

    write_chunks(tmp_path, "a", ["one", "two", "three"])
    embed.save_embeddings(collection, str(tmp_path))
    assert collection.count() == 3
    assert fake.calls == calls_before + 1


def test_store_chunks_batches(tmp_path, collection, monkeypatch):
    monkeypatch.setattr(embed, "BATCH_SIZE", 2)
    ids = [f"a:1:{i}" for i in range(5)]
    metadatas = [{"file": "a", "page": 1, "chunk_id": i} for i in range(5)]
    assert embed.store_chunks(collection, ids, [f"t{i}" for i in range(5)], metadatas) == 5
    assert collection.count() == 5


def test_save_embeddings_missing_folder_raises(collection, tmp_path):
    with pytest.raises(NotADirectoryError):
        embed.save_embeddings(collection, str(tmp_path / "nope"))


def test_save_embeddings_empty_folder_is_a_noop(collection, tmp_path, capsys):
    embed.save_embeddings(collection, str(tmp_path))
    assert collection.count() == 0
    assert "no .jsonl files" in capsys.readouterr().out


def test_format_results_pairs_fields_per_hit(tmp_path, collection):
    write_chunks(tmp_path, "a", ["alpha", "beta", "gamma"])
    embed.save_embeddings(collection, str(tmp_path))
    lines = embed.format_results(embed.query_embeddings(collection, "alpha", n_results=2))
    assert len(lines) == 2
    assert lines[0].startswith("1. a:1:1 (file=a, page=1, distance=")
    assert "'alpha'" in lines[0]


def test_get_collection_requires_api_key_without_embedding_function(tmp_path, monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="VOYAGE_API_KEY"):
        embed.get_collection(str(tmp_path / "db"), "test")
