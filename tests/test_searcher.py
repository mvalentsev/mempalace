"""
test_searcher.py -- Tests for both search() (CLI) and search_memories() (API).

Uses the real ChromaDB fixtures from conftest.py for integration tests,
plus mock-based tests for error paths.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from _chroma_palace_helper import make_minimal_chroma_sqlite

from mempalace.backends import BackendMismatchError
from mempalace.searcher import (
    SearchError,
    build_where_filter,
    get_collection,
    search,
    search_memories,
)


# ── build_where_filter (unit) ──────────────────────────────────────────


class TestBuildWhereFilter:
    """build_where_filter composes a ChromaDB where clause from optional
    wing / room / source_file constraints (#1815). ChromaDB needs a ``$and``
    only when ≥2 clauses are present; a single clause is returned bare and
    zero clauses yield an empty filter."""

    def test_no_filters_returns_empty(self):
        assert build_where_filter() == {}

    def test_wing_only(self):
        assert build_where_filter(wing="backend") == {"wing": "backend"}

    def test_room_only(self):
        assert build_where_filter(room="auth") == {"room": "auth"}

    def test_wing_and_room(self):
        assert build_where_filter(wing="backend", room="auth") == {
            "$and": [{"wing": "backend"}, {"room": "auth"}]
        }

    def test_source_file_only(self):
        assert build_where_filter(source_file="auth.py") == {"source_file": "auth.py"}

    def test_wing_and_source_file(self):
        assert build_where_filter(wing="backend", source_file="auth.py") == {
            "$and": [{"wing": "backend"}, {"source_file": "auth.py"}]
        }

    def test_room_and_source_file(self):
        assert build_where_filter(room="auth", source_file="auth.py") == {
            "$and": [{"room": "auth"}, {"source_file": "auth.py"}]
        }

    def test_wing_room_and_source_file(self):
        assert build_where_filter(wing="backend", room="auth", source_file="auth.py") == {
            "$and": [{"wing": "backend"}, {"room": "auth"}, {"source_file": "auth.py"}]
        }


# ── search_memories (API) ──────────────────────────────────────────────


class TestSearchMemories:
    def test_basic_search(self, palace_path, seeded_collection):
        result = search_memories("JWT authentication", palace_path)
        assert "results" in result
        assert len(result["results"]) > 0
        assert result["query"] == "JWT authentication"

    def test_wing_filter(self, palace_path, seeded_collection):
        result = search_memories("planning", palace_path, wing="notes")
        assert all(r["wing"] == "notes" for r in result["results"])

    def test_room_filter(self, palace_path, seeded_collection):
        result = search_memories("database", palace_path, room="backend")
        assert all(r["room"] == "backend" for r in result["results"])

    def test_wing_and_room_filter(self, palace_path, seeded_collection):
        result = search_memories("code", palace_path, wing="project", room="frontend")
        assert all(r["wing"] == "project" and r["room"] == "frontend" for r in result["results"])

    def test_source_file_filter(self, palace_path, seeded_collection):
        result = search_memories("authentication module", palace_path, source_file="auth.py")
        assert result["results"], "exact source_file match should return its drawer"
        assert all(r["source_file"] == "auth.py" for r in result["results"])

    def test_source_file_with_wing_filter(self, palace_path, seeded_collection):
        result = search_memories("database", palace_path, wing="project", source_file="db.py")
        assert result["results"]
        assert all(
            r["source_file"] == "db.py" and r["wing"] == "project" for r in result["results"]
        )

    def test_nonmatching_source_file_returns_empty_not_error(self, palace_path, seeded_collection):
        result = search_memories("authentication", palace_path, source_file="nope.md")
        assert "error" not in result
        assert result["results"] == []

    def test_filters_envelope_includes_source_file(self, palace_path, seeded_collection):
        result = search_memories("authentication", palace_path, source_file="auth.py")
        assert result["filters"]["source_file"] == "auth.py"

    def test_result_exposes_full_source_path(self, palace_path, seeded_collection):
        # The displayed source_file is a basename; source_path carries the full
        # stored value so a caller can round-trip it back into a source_file filter.
        result = search_memories("authentication module", palace_path)
        hit = result["results"][0]
        assert hit["source_file"] == "auth.py"
        assert hit["source_path"] == "auth.py"

    def test_source_file_filter_matches_full_path_not_basename(self, palace_path):
        from mempalace.palace import get_collection

        col = get_collection(palace_path, create=True)
        col.upsert(
            ids=["fp1"],
            documents=["The deploy script restarts the gunicorn workers nightly."],
            metadatas=[{"wing": "ops", "room": "deploy", "source_file": "/srv/app/deploy.sh"}],
        )
        # The full stored path matches and round-trips via source_path.
        hit = search_memories(
            "deploy gunicorn workers", palace_path, source_file="/srv/app/deploy.sh"
        )
        assert [h["source_path"] for h in hit["results"]] == ["/srv/app/deploy.sh"]
        assert [h["source_file"] for h in hit["results"]] == ["deploy.sh"]
        # The basename does NOT match — exact full-path semantics only (issue v1).
        miss = search_memories("deploy gunicorn workers", palace_path, source_file="deploy.sh")
        assert miss["results"] == []

    def test_source_file_filter_honored_in_bm25_fallback(self, palace_path, seeded_collection):
        # vector_disabled routes through _bm25_only_via_sqlite (#1222); the
        # source_file filter must hold there too, not silently no-op.
        result = search_memories(
            "authentication module",
            palace_path,
            source_file="auth.py",
            vector_disabled=True,
            collection_name="mempalace_drawers",
        )
        assert "error" not in result
        assert result["results"], "BM25 fallback should still find the auth drawer"
        assert all(r["source_file"] == "auth.py" for r in result["results"])

    def test_n_results_limit(self, palace_path, seeded_collection):
        result = search_memories("code", palace_path, n_results=2)
        assert len(result["results"]) <= 2

    def test_no_palace_returns_error(self, tmp_path):
        result = search_memories("anything", str(tmp_path / "missing"))
        assert "error" in result

    def test_result_fields(self, palace_path, seeded_collection):
        result = search_memories("authentication", palace_path)
        hit = result["results"][0]
        assert "text" in hit
        assert "wing" in hit
        assert "room" in hit
        assert "source_file" in hit
        assert "similarity" in hit
        assert isinstance(hit["similarity"], float)
        assert "created_at" in hit

    def test_created_at_contains_filed_at(self, palace_path, seeded_collection):
        """created_at surfaces the filed_at metadata from the drawer."""
        result = search_memories("JWT authentication", palace_path)
        hit = result["results"][0]
        assert hit["created_at"] == "2026-01-01T00:00:00"

    def test_created_at_fallback_when_filed_at_missing(self):
        """created_at defaults to 'unknown' when filed_at is absent."""
        mock_col = MagicMock()
        mock_col.query.return_value = {
            "ids": [["drawer_no_date"]],
            "documents": [["Some text without a date"]],
            "metadatas": [[{"wing": "project", "room": "backend", "source_file": "x.py"}]],
            "distances": [[0.1]],
        }

        with patch("mempalace.searcher.get_collection", return_value=mock_col):
            result = search_memories("test", "/fake/path")
        hit = result["results"][0]
        assert hit["created_at"] == "unknown"

    def test_search_memories_query_error(self):
        """search_memories returns error dict when query raises."""
        mock_col = MagicMock()
        mock_col.query.side_effect = RuntimeError("query failed")

        with patch("mempalace.searcher.get_collection", return_value=mock_col):
            result = search_memories("test", "/fake/path")
        assert "error" in result
        assert "query failed" in result["error"]

    def test_search_memories_vector_path_uses_explicit_collection_name(self):
        mock_col = MagicMock()
        mock_col.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
            "ids": [[]],
        }

        with patch("mempalace.searcher.get_collection", return_value=mock_col) as get_collection:
            search_memories("test", "/fake/path", collection_name="custom_drawers")

        get_collection.assert_called_once_with(
            "/fake/path",
            collection_name="custom_drawers",
            create=False,
        )

    def test_search_memories_filters_in_result(self, palace_path, seeded_collection):
        result = search_memories("test", palace_path, wing="project", room="backend")
        assert result["filters"]["wing"] == "project"
        assert result["filters"]["room"] == "backend"

    def test_search_memories_handles_none_metadata(self):
        """API path: `None` entries in the drawer results' metadatas list must
        fall back to the sentinel strings (wing/room 'unknown', source '?')
        rather than raising `AttributeError: 'NoneType' object has no
        attribute 'get'` while the rest of the result set renders."""
        mock_col = MagicMock()
        mock_col.query.return_value = {
            "documents": [["first doc", "second doc"]],
            "metadatas": [[{"source_file": "a.md", "wing": "w", "room": "r"}, None]],
            "distances": [[0.1, 0.2]],
            "ids": [["d1", "d2"]],
        }

        def mock_get_collection(path, collection_name=None, create=False):
            # First call: drawers. Second call: closets — raise so hybrid
            # degrades to pure drawer search (the catch block covers it).
            if not hasattr(mock_get_collection, "_called"):
                mock_get_collection._called = True
                return mock_col
            raise RuntimeError("no closets")

        with patch("mempalace.searcher.get_collection", side_effect=mock_get_collection):
            result = search_memories("anything", "/fake/path")
        assert "results" in result
        assert len(result["results"]) == 2
        # The None-metadata hit renders with sentinel values, not a crash.
        none_hit = result["results"][1]
        assert none_hit["text"] == "second doc"
        assert none_hit["wing"] == "unknown"
        assert none_hit["room"] == "unknown"

    def test_effective_distance_clamped_to_valid_cosine_range(self):
        """A strong closet boost (up to 0.40) applied to a low-distance drawer
        can drive ``dist - boost`` negative. That violates the cosine-distance
        invariant ``[0, 2]``: the API returns ``similarity > 1.0`` and the
        internal ``_sort_key`` sinks below ordinary positive distances,
        inverting the ranking so the best hybrid matches sort last.

        With the clamp, ``effective_distance`` stays in ``[0, 2]``,
        ``similarity`` stays in ``[0, 1]``, and the sort order is stable.
        """
        # Drawer a.md gets a tiny base distance (0.08) — nearly exact match.
        # Drawer b.md gets a larger base distance (0.35).
        drawers_col = MagicMock()
        drawers_col.query.return_value = {
            "documents": [["doc-a", "doc-b"]],
            "metadatas": [
                [
                    {"source_file": "a.md", "wing": "w", "room": "r", "chunk_index": 0},
                    {"source_file": "b.md", "wing": "w", "room": "r", "chunk_index": 0},
                ]
            ],
            "distances": [[0.08, 0.35]],
            "ids": [["d-a", "d-b"]],
        }
        # A strong closet at rank 0 points at a.md → boost = 0.40,
        # which exceeds a.md's base distance and would go negative without
        # the clamp. No closet for b.md.
        closets_col = MagicMock()
        closets_col.query.return_value = {
            "documents": [["closet-preview-a"]],
            "metadatas": [[{"source_file": "a.md"}]],
            "distances": [[0.2]],  # within CLOSET_DISTANCE_CAP (1.5)
            "ids": [["c-a"]],
        }

        with (
            patch("mempalace.searcher.get_collection", return_value=drawers_col),
            patch("mempalace.searcher.get_closets_collection", return_value=closets_col),
        ):
            result = search_memories("query", "/fake/path", n_results=5)

        hits = result["results"]
        assert hits, "should return results"

        # Invariants on every hit.
        for h in hits:
            assert 0.0 <= h["similarity"] <= 1.0, (
                f"similarity out of range: {h['similarity']} for {h['source_file']}"
            )
            assert 0.0 <= h["effective_distance"] <= 2.0, (
                f"effective_distance out of range: {h['effective_distance']} for {h['source_file']}"
            )

        # With the clamp, the closet-boosted a.md still ranks ahead of b.md —
        # the boost still wins, but it no longer flips the ranking.
        assert hits[0]["source_file"] == "a.md"
        assert hits[0]["matched_via"] == "drawer+closet"


# ── BM25 internals: None / empty document safety ─────────────────────


class TestBM25NoneSafety:
    """Regression tests for the AttributeError observed in production when
    Chroma returned ``None`` documents inside a hybrid-rerank pass.

    Trace from the daemon log (2026-04-24 21:07:05):
        File "mempalace/searcher.py", line 81, in _bm25_scores
            tokenized = [_tokenize(d) for d in documents]
        File "mempalace/searcher.py", line 52, in _tokenize
            return _TOKEN_RE.findall(text.lower())
        AttributeError: 'NoneType' object has no attribute 'lower'
    """

    def test_tokenize_handles_none(self):
        from mempalace.searcher import _tokenize

        assert _tokenize(None) == []

    def test_tokenize_handles_empty_string(self):
        from mempalace.searcher import _tokenize

        assert _tokenize("") == []

    def test_bm25_scores_does_not_crash_on_none_documents(self):
        """A ``None`` mixed into the corpus must yield score 0.0 for that doc
        and finite scores for the rest, not raise AttributeError."""
        from mempalace.searcher import _bm25_scores

        scores = _bm25_scores(
            "postgres migration", ["postgres migration done", None, "kafka rebalance"]
        )
        assert len(scores) == 3
        assert scores[1] == 0.0
        assert scores[0] > 0.0


# ── search() (CLI print function) ─────────────────────────────────────


@pytest.fixture
def fake_palace_path(tmp_path):
    """tmp_path with chroma.sqlite3 touched so searcher.search's
    filesystem-first state checks (#1498) pass through to the mocked
    backend instead of raising on State A / State B."""
    p = tmp_path / "palace"
    p.mkdir()
    make_minimal_chroma_sqlite(p)
    return str(p)


class TestSearchCLI:
    def test_search_prints_results(self, palace_path, seeded_collection, capsys):
        search("JWT authentication", palace_path)
        captured = capsys.readouterr()
        assert "JWT" in captured.out or "authentication" in captured.out

    def test_search_with_wing_filter(self, palace_path, seeded_collection, capsys):
        search("planning", palace_path, wing="notes")
        captured = capsys.readouterr()
        assert "Results for" in captured.out

    def test_search_with_room_filter(self, palace_path, seeded_collection, capsys):
        search("database", palace_path, room="backend")
        captured = capsys.readouterr()
        assert "Room:" in captured.out

    def test_search_with_wing_and_room(self, palace_path, seeded_collection, capsys):
        search("code", palace_path, wing="project", room="frontend")
        captured = capsys.readouterr()
        assert "Wing:" in captured.out
        assert "Room:" in captured.out

    def test_search_no_palace_raises(self, tmp_path):
        with pytest.raises(SearchError, match="No palace found"):
            search("anything", str(tmp_path / "missing"))

    def test_search_no_results(self, palace_path, collection, capsys):
        """Empty collection returns no results message."""
        # collection is empty (no seeded data)
        result = search("xyzzy_nonexistent_query", palace_path, n_results=1)
        captured = capsys.readouterr()
        # Either prints "No results" or returns None
        assert result is None or "No results" in captured.out

    def test_search_query_error_raises(self, fake_palace_path):
        """search raises SearchError when query fails."""
        mock_col = MagicMock()
        mock_col.query.side_effect = RuntimeError("boom")

        with patch("mempalace.searcher.get_collection", return_value=mock_col):
            with pytest.raises(SearchError, match="Search error"):
                search("test", fake_palace_path)

    def test_search_n_results(self, palace_path, seeded_collection, capsys):
        search("code", palace_path, n_results=1)
        captured = capsys.readouterr()
        # Should have output with at least one result block
        assert "[1]" in captured.out

    def test_search_applies_bm25_hybrid_rerank(self, fake_palace_path, capsys):
        """CLI search must call the same hybrid rerank that the MCP path uses.

        Regression for a bug where the CLI only consulted ChromaDB cosine
        distance: a drawer whose body contained every query term still
        scored zero similarity if its embedding happened to be far from
        the query (e.g. the drawer was a shell-output fragment that
        embeds as "file tree noise"). Hybrid rerank fixes this by
        combining BM25 with cosine — lexical matches rise above pure
        vector noise.

        Simulates: three candidates, all with distance >= 1.0 (cosine = 0);
        candidate 2 contains every query term. After the fix, candidate 2
        should rank first and display a non-zero bm25 score.
        """
        mock_col = MagicMock()
        mock_col.metadata = {"hnsw:space": "cosine"}
        mock_col.query.return_value = {
            "documents": [
                [
                    "unrelated directory listing -rw-rw-r-- file.txt",
                    "foo bar baz is a multi-word phrase",
                    "another unrelated chunk about colors",
                ]
            ],
            "metadatas": [
                [
                    {"source_file": "a.md", "wing": "w", "room": "r"},
                    {"source_file": "b.md", "wing": "w", "room": "r"},
                    {"source_file": "c.md", "wing": "w", "room": "r"},
                ]
            ],
            "distances": [[1.5, 1.5, 1.5]],
        }
        with patch("mempalace.searcher.get_collection", return_value=mock_col):
            search("foo bar baz", fake_palace_path)
        captured = capsys.readouterr()
        first_block, _, _ = captured.out.partition("[2]")
        # Lexical match must rank first
        assert "b.md" in first_block, (
            f"expected lexical match 'b.md' at rank 1, got:\n{captured.out}"
        )
        # Non-zero bm25 reported
        assert "bm25=" in first_block
        assert "bm25=0.0" not in first_block
        # Metric-labeled vector similarity still reported for transparency.
        # Label is now "<metric>_sim=" (honest about the backend's metric)
        # rather than a hard-coded "cosine=".
        assert "cosine_sim=" in first_block

    def test_search_warns_when_palace_uses_wrong_distance_metric(self, fake_palace_path, capsys):
        """Legacy palaces created without `hnsw:space=cosine` silently
        use L2, which breaks similarity interpretation. CLI must warn
        the user and point them at `mempalace repair` rather than
        pretending the `Match` scores are meaningful."""
        mock_col = MagicMock()
        mock_col.metadata = {}  # legacy: no hnsw:space set
        mock_col.query.return_value = {
            "documents": [["some drawer content"]],
            "metadatas": [[{"source_file": "a.md", "wing": "w", "room": "r"}]],
            "distances": [[1.2]],
        }
        with patch("mempalace.searcher.get_collection", return_value=mock_col):
            search("anything", fake_palace_path)
        captured = capsys.readouterr()
        assert "mempalace repair" in captured.err
        assert "cosine" in captured.err.lower()

    def test_search_does_not_warn_when_palace_is_correctly_configured(
        self, fake_palace_path, capsys
    ):
        mock_col = MagicMock()
        mock_col.metadata = {"hnsw:space": "cosine"}
        mock_col.query.return_value = {
            "documents": [["some drawer content"]],
            "metadatas": [[{"source_file": "a.md", "wing": "w", "room": "r"}]],
            "distances": [[0.3]],
        }
        with patch("mempalace.searcher.get_collection", return_value=mock_col):
            search("anything", fake_palace_path)
        captured = capsys.readouterr()
        assert "mempalace repair" not in captured.err

    def test_search_handles_none_metadata_without_crash(self, fake_palace_path, capsys):
        """ChromaDB can return `None` entries in the metadatas list when a
        drawer has no metadata. The CLI print path must not crash on them
        mid-render — it used to raise `AttributeError: 'NoneType' object has
        no attribute 'get'` after printing earlier results."""
        mock_col = MagicMock()
        mock_col.query.return_value = {
            "documents": [["first doc", "second doc"]],
            "metadatas": [[{"source_file": "a.md", "wing": "w", "room": "r"}, None]],
            "distances": [[0.1, 0.2]],
        }
        with patch("mempalace.searcher.get_collection", return_value=mock_col):
            search("anything", fake_palace_path)
        captured = capsys.readouterr()
        assert "[1]" in captured.out
        assert "[2]" in captured.out
        # Second result renders with fallback '?' values instead of crashing
        assert "second doc" in captured.out

    def test_search_handles_none_document_without_crash(self, fake_palace_path, capsys):
        mock_col = MagicMock()
        mock_col.metadata = {"hnsw:space": "cosine"}
        mock_col.query.return_value = {
            "documents": [["first doc", None]],
            "metadatas": [[{"source_file": "a.md", "wing": "w", "room": "r"}, None]],
            "distances": [[0.1, 0.2]],
        }
        with patch("mempalace.searcher.get_collection", return_value=mock_col):
            search("anything", fake_palace_path)
        captured = capsys.readouterr()
        assert "[1]" in captured.out
        assert "[2]" in captured.out

    def test_search_routes_to_bm25_when_hnsw_diverged(self, fake_palace_path, capsys):
        """Regression: `mempalace search` on a diverged HNSW segment must not
        segfault ChromaDB's Rust bindings.

        The MCP path gates this via ``_vector_disabled`` (#1222); the CLI
        path was missing the gate, so any query into a diverged palace
        exited 139 (SIGBUS) at ``chromadb/api/rust.py:_query`` with zero
        diagnostic output. This test verifies the CLI now probes
        ``hnsw_capacity_status`` and routes to the BM25-only sqlite
        fallback instead of calling ``col.query()`` against a segment
        that would crash it.
        """
        bm25_result = {
            "query": "anything",
            "filters": {},
            "total_before_filter": 1,
            "results": [
                {
                    "text": "diary entry that matches the query",
                    "wing": "wing_test",
                    "room": "diary",
                    "source_file": "test.jsonl",
                    "bm25_score": 1.5,
                    "distance": None,
                }
            ],
            "fallback": "bm25_only_via_sqlite",
            "fallback_reason": "vector_search_disabled",
        }
        with (
            patch("mempalace.searcher.resolve_backend_name", return_value="chroma"),
            patch(
                "mempalace.backends.chroma.hnsw_capacity_status",
                return_value={"diverged": True, "message": "test divergence"},
            ),
            patch("mempalace.searcher._bm25_only_via_sqlite", return_value=bm25_result),
            patch("mempalace.searcher.get_collection") as mock_get_collection,
        ):
            search("anything", fake_palace_path)
        captured = capsys.readouterr()
        # Routed to BM25 before opening Chroma at all. Client construction and
        # identity enforcement can touch the same damaged native index, so a
        # query-only guard is insufficient.
        mock_get_collection.assert_not_called()
        # User got actionable output, not a silent crash.
        assert "mempalace repair" in captured.out
        assert "diary entry that matches" in captured.out

    def test_search_proceeds_to_vector_when_hnsw_healthy(self, fake_palace_path, capsys):
        """Paired guard: when HNSW is healthy, the divergence probe must NOT
        short-circuit to BM25 — vector search proceeds normally.

        Prevents a regression where the gate accidentally always fires.
        """
        mock_col = MagicMock()
        mock_col.metadata = {"hnsw:space": "cosine"}
        mock_col.query.return_value = {
            "documents": [["a matching doc"]],
            "metadatas": [[{"source_file": "a.md", "wing": "w", "room": "r"}]],
            "distances": [[0.1]],
        }
        with (
            patch("mempalace.searcher.resolve_backend_name", return_value="chroma"),
            patch(
                "mempalace.backends.chroma.hnsw_capacity_status",
                return_value={"diverged": False, "status": "ok"},
            ),
            patch("mempalace.searcher._bm25_only_via_sqlite") as mock_bm25,
            patch("mempalace.searcher.get_collection", return_value=mock_col),
        ):
            search("anything", fake_palace_path)
        captured = capsys.readouterr()
        # Vector path ran.
        mock_col.query.assert_called_once()
        # BM25 fallback was NOT invoked.
        mock_bm25.assert_not_called()
        assert "a matching doc" in captured.out

    def test_search_forwards_date_window_to_bm25_fallback_when_hnsw_diverged(
        self, fake_palace_path
    ):
        """A `--since`/`--before` window must survive the diverged-index detour.

        The fence returns BM25-only results before the vector path runs, and
        that fallback reads drawers straight from sqlite. Unless the window
        travels with it, the CLI answers a wider question than the caller
        asked — silently, with no notice that the filter was dropped. A
        degraded index may cost ranking quality; it must never cost the
        filter.
        """
        seen = {}

        def _spy_bm25(**kwargs):
            seen.update(kwargs)
            return {"query": "anything", "filters": {}, "total_before_filter": 0, "results": []}

        with (
            patch("mempalace.searcher.resolve_backend_name", return_value="chroma"),
            patch(
                "mempalace.backends.chroma.hnsw_capacity_status",
                return_value={"diverged": True, "message": "test divergence"},
            ),
            patch("mempalace.searcher._bm25_only_via_sqlite", side_effect=_spy_bm25),
        ):
            search("anything", fake_palace_path, since="2026-01-01", before="2026-02-01")

        assert seen["since_dt"] == datetime(2026, 1, 1)
        assert seen["before_dt"] == datetime(2026, 2, 1)

    def test_search_rejects_inverted_window_before_probing_a_diverged_index(self, fake_palace_path):
        """An inverted window is a caller error, so it must raise the same way
        whether the index is healthy or diverged. If the fence ran first it
        would swallow the mistake and answer with unfiltered BM25 results."""
        with (
            patch("mempalace.searcher.resolve_backend_name", return_value="chroma"),
            patch(
                "mempalace.backends.chroma.hnsw_capacity_status",
                return_value={"diverged": True, "message": "test divergence"},
            ),
            patch("mempalace.searcher._bm25_only_via_sqlite") as mock_bm25,
        ):
            with pytest.raises(SearchError, match="must be earlier than"):
                search("anything", fake_palace_path, since="2026-02-01", before="2026-01-01")
        mock_bm25.assert_not_called()

    def test_search_does_not_run_chroma_probe_for_other_backends(self, fake_palace_path, capsys):
        """The HNSW guard is Chroma-specific and must not fence other backends."""
        mock_col = MagicMock()
        mock_col.query.return_value = {
            "documents": [["backend-native result"]],
            "metadatas": [[{"source_file": "native.md", "wing": "w", "room": "r"}]],
            "distances": [[0.1]],
        }
        with (
            patch("mempalace.searcher.resolve_backend_name", return_value="sqlite_exact"),
            patch("mempalace.backends.chroma.hnsw_capacity_status") as mock_probe,
            patch("mempalace.searcher.get_collection", return_value=mock_col),
        ):
            search("anything", fake_palace_path)

        mock_probe.assert_not_called()
        mock_col.query.assert_called_once()
        assert "backend-native result" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "resolution_error",
        [BackendMismatchError("mixed backend artifacts"), KeyError("unknown_backend")],
    )
    def test_search_delegates_backend_resolution_errors_to_open_diagnostic(
        self, fake_palace_path, resolution_error
    ):
        """The early HNSW fence must not replace normal CLI diagnostics."""
        with (
            patch("mempalace.searcher.resolve_backend_name", side_effect=resolution_error),
            patch("mempalace.searcher._hnsw_capacity_diverged") as mock_probe,
            patch("mempalace.searcher._open_collection_or_explain", return_value=None) as mock_open,
        ):
            with pytest.raises(SearchError):
                search("anything", fake_palace_path)

        mock_probe.assert_not_called()
        mock_open.assert_called_once_with(fake_palace_path, opener=get_collection)


# ── since/before date window (#463) ────────────────────────────────────


class TestSearchMemoriesDateFilter:
    """search_memories accepts since/before ISO bounds filtered on filed_at.

    Window semantics mirror list_drawers (#1128): since inclusive, before
    exclusive, wall-clock naive comparison, undated drawers excluded while
    a bound is active. Seeded filed_at values: aaa=01-01, bbb=01-02,
    ccc=01-03, ddd=01-04 (see conftest seeded_collection).
    """

    BROAD = "authentication database frontend sprint planning"

    def test_since_narrows_to_newer_drawers(self, palace_path, seeded_collection):
        result = search_memories(self.BROAD, palace_path, n_results=10, since="2026-01-03")
        assert result["results"], "expected in-window hits"
        assert all(r["created_at"] >= "2026-01-03" for r in result["results"])

    def test_before_narrows_to_older_drawers(self, palace_path, seeded_collection):
        result = search_memories(self.BROAD, palace_path, n_results=10, before="2026-01-02")
        assert result["results"]
        assert all(r["created_at"] < "2026-01-02" for r in result["results"])

    def test_window_both_bounds(self, palace_path, seeded_collection):
        result = search_memories(
            self.BROAD, palace_path, n_results=10, since="2026-01-02", before="2026-01-04"
        )
        got = sorted(r["created_at"][:10] for r in result["results"])
        assert got == ["2026-01-02", "2026-01-03"]

    def test_since_boundary_inclusive(self, palace_path, seeded_collection):
        result = search_memories(self.BROAD, palace_path, n_results=10, since="2026-01-04")
        assert [r["created_at"][:10] for r in result["results"]] == ["2026-01-04"]

    def test_before_boundary_exclusive(self, palace_path, seeded_collection):
        result = search_memories(self.BROAD, palace_path, n_results=10, before="2026-01-04")
        assert "2026-01-04" not in [r["created_at"][:10] for r in result["results"]]
        assert len(result["results"]) == 3

    def test_invalid_since_returns_error(self, palace_path, seeded_collection):
        result = search_memories(self.BROAD, palace_path, since="next tuesday")
        assert "error" in result
        assert "since" in result["error"]

    def test_inverted_window_returns_error(self, palace_path, seeded_collection):
        result = search_memories(self.BROAD, palace_path, since="2026-01-04", before="2026-01-01")
        assert "error" in result
        assert "must be earlier than" in result["error"]

    def test_undated_drawer_excluded_while_bound_active(self, palace_path, seeded_collection):
        seeded_collection.upsert(
            ids=["undated1"],
            documents=["Undated planning note about authentication frontend database."],
            metadatas=[{"wing": "notes", "room": "planning", "source_file": "undated.md"}],
        )
        result = search_memories(self.BROAD, palace_path, n_results=10, since="2026-01-01")
        assert "undated.md" not in [r["source_file"] for r in result["results"]]
        # ...but with no bound it is searchable as before.
        result = search_memories(self.BROAD, palace_path, n_results=10)
        assert "undated.md" in [r["source_file"] for r in result["results"]]

    def test_aware_filed_at_matches_wall_clock(self, palace_path, seeded_collection):
        # diary_ingest stamps aware UTC; the window compares wall-clock fields.
        seeded_collection.upsert(
            ids=["aware1"],
            documents=["Aware planning entry about database sprint authentication."],
            metadatas=[
                {
                    "wing": "notes",
                    "room": "planning",
                    "source_file": "aware.md",
                    "filed_at": "2026-01-05T12:00:00+00:00",
                }
            ],
        )
        result = search_memories(self.BROAD, palace_path, n_results=10, since="2026-01-05")
        assert [r["source_file"] for r in result["results"]] == ["aware.md"]

    def test_filters_envelope_echoes_window(self, palace_path, seeded_collection):
        result = search_memories(self.BROAD, palace_path, since="2026-01-02", before="2026-01-03")
        assert result["filters"]["since"] == "2026-01-02"
        assert result["filters"]["before"] == "2026-01-03"

    def test_no_window_keeps_filters_none(self, palace_path, seeded_collection):
        result = search_memories("authentication", palace_path)
        assert result["filters"]["since"] is None
        assert result["filters"]["before"] is None

    def test_window_composes_with_wing_filter(self, palace_path, seeded_collection):
        result = search_memories(
            self.BROAD, palace_path, n_results=10, wing="project", since="2026-01-02"
        )
        got = {(r["wing"], r["created_at"][:10]) for r in result["results"]}
        assert got == {("project", "2026-01-02"), ("project", "2026-01-03")}

    def test_max_distance_zero_results_stay_empty_not_error(self, palace_path, seeded_collection):
        result = search_memories("zebra quantum blockchain", palace_path, since="2027-01-01")
        assert "error" not in result
        assert result["results"] == []

    def test_date_window_widens_candidate_pool(self, palace_path):
        # 12 near-duplicate drawers embed closest to the query; the one
        # in-window drawer is textually farther, so the historical 3x pool
        # (n_results=2 -> 6) would never contain it. The widened window
        # pool must recover it: recall is the design requirement.
        from mempalace.palace import get_collection

        col = get_collection(palace_path, create=True)
        ids, docs, metas = [], [], []
        for i in range(12):
            ids.append(f"near{i}")
            docs.append(f"Weekly budget meeting notes revision {i} about spending review.")
            metas.append(
                {
                    "wing": "fin",
                    "room": "budget",
                    "source_file": f"near{i}.md",
                    "filed_at": "2026-02-01T00:00:00",
                }
            )
        ids.append("target")
        docs.append("Quarterly offsite retrospective and travel logistics summary.")
        metas.append(
            {
                "wing": "fin",
                "room": "budget",
                "source_file": "target.md",
                "filed_at": "2026-03-01T00:00:00",
            }
        )
        col.upsert(ids=ids, documents=docs, metadatas=metas)
        result = search_memories(
            "budget meeting spending review",
            palace_path,
            n_results=2,
            since="2026-02-15",
        )
        assert [r["source_file"] for r in result["results"]] == ["target.md"]

    def test_bm25_fallback_respects_window(self, palace_path, seeded_collection):
        result = search_memories(
            "authentication database frontend sprint planning tokens",
            palace_path,
            n_results=10,
            vector_disabled=True,
            since="2026-01-02",
            before="2026-01-04",
        )
        assert result.get("fallback") == "bm25_only_via_sqlite"
        assert result["results"], "expected in-window bm25 hits"
        assert sorted(r["created_at"][:10] for r in result["results"]) == [
            "2026-01-02",
            "2026-01-03",
        ]
        assert result["filters"]["since"] == "2026-01-02"

    def test_bm25_fallback_includes_bare_date_filed_at_on_since_boundary(
        self, palace_path, seeded_collection
    ):
        # A bare-date filed_at equal to the since day is in-window
        # (since is inclusive, parsed as midnight). The SQL prefilter must
        # not drop it before the authoritative Python check: lexicographic
        # "2026-01-02" < "2026-01-02T00:00:00", so a full-isoformat lower
        # bound would exclude it at the SQL layer where Python can't
        # recover it (review finding on the #463 change).
        seeded_collection.upsert(
            ids=["bare1", "space1", "zulu1"],
            documents=[
                "Bare-date drawer about the database sprint planning.",
                "Space-separated drawer about the database sprint planning.",
                "Zulu drawer about the database sprint planning.",
            ],
            metadatas=[
                {
                    "wing": "notes",
                    "room": "planning",
                    "source_file": "bare.md",
                    "filed_at": "2026-01-02",
                },
                {
                    "wing": "notes",
                    "room": "planning",
                    "source_file": "space.md",
                    # sqlite CURRENT_TIMESTAMP style: space separator sorts
                    # before "T" and a full-isoformat SQL bound would drop
                    # the whole boundary day.
                    "filed_at": "2026-01-02 09:30:00",
                },
                {
                    "wing": "notes",
                    "room": "planning",
                    "source_file": "zulu.md",
                    # "Z" sorts after a fractional bound; the day-granular
                    # upper prefilter must keep it for the Python check.
                    "filed_at": "2026-01-02T10:00:00Z",
                },
            ],
        )
        result = search_memories(
            "database sprint planning",
            palace_path,
            n_results=10,
            vector_disabled=True,
            since="2026-01-02",
            before="2026-01-02T10:00:00.500000",
        )
        assert result.get("fallback") == "bm25_only_via_sqlite"
        got = [r["source_file"] for r in result["results"]]
        assert "bare.md" in got
        assert "space.md" in got
        assert "zulu.md" in got

    def test_bm25_fallback_invalid_since_errors(self, palace_path, seeded_collection):
        result = search_memories(
            "authentication", palace_path, vector_disabled=True, since="garbage"
        )
        assert "error" in result
        assert "since" in result["error"]

    def test_pool_truncated_flag_set_when_widened_pool_full(self, palace_path):
        # n_results=1 -> widened pool = 15; seed 16 in-window drawers so the
        # backend returns a full pool and the honesty flag must fire.
        from mempalace.palace import get_collection

        col = get_collection(palace_path, create=True)
        ids, docs, metas = [], [], []
        for i in range(16):
            ids.append(f"flag{i}")
            docs.append(f"standup summary entry number {i} about deploy status.")
            metas.append(
                {
                    "wing": "ops",
                    "room": "standup",
                    "source_file": f"flag{i}.md",
                    "filed_at": "2026-05-01T00:00:00",
                }
            )
        col.upsert(ids=ids, documents=docs, metadatas=metas)
        result = search_memories(
            "standup deploy status", palace_path, n_results=1, since="2026-04-01"
        )
        assert result.get("date_filter_pool_truncated") is True
        assert result["total_before_filter"] >= 15

    def test_pool_truncated_flag_absent_on_small_corpus(self, palace_path, seeded_collection):
        result = search_memories(
            "authentication database", palace_path, n_results=5, since="2026-01-01"
        )
        assert "date_filter_pool_truncated" not in result

    def test_bm25_fallback_survives_calendar_ceiling_before(self, palace_path, seeded_collection):
        # before="9999-12-31" is a plausible open-ended sentinel; the
        # day-granular SQL prefilter must not overflow past datetime.max
        # on the resilience path (it degrades to no SQL narrowing and the
        # Python filter decides).
        result = search_memories(
            "authentication database frontend sprint planning",
            palace_path,
            n_results=10,
            vector_disabled=True,
            since="2026-01-01",
            before="9999-12-31",
        )
        assert "error" not in result
        assert result.get("fallback") == "bm25_only_via_sqlite"
        assert len(result["results"]) == 4

    def test_bm25_fallback_pool_truncated_flag(self, palace_path, seeded_collection):
        # Direct call with a tiny max_candidates: the FTS page comes back
        # full under an active window -> the same honesty flag as the
        # vector path; without a window the key stays absent.
        from datetime import datetime

        from mempalace.searcher import _bm25_only_via_sqlite

        truncated = _bm25_only_via_sqlite(
            "authentication database frontend sprint",
            palace_path,
            n_results=2,
            max_candidates=2,
            since_dt=datetime(2026, 1, 1),
            before_dt=None,
        )
        assert truncated.get("date_filter_pool_truncated") is True
        unwindowed = _bm25_only_via_sqlite(
            "authentication database frontend sprint",
            palace_path,
            n_results=2,
            max_candidates=2,
        )
        assert "date_filter_pool_truncated" not in unwindowed

    def test_union_strategy_respects_window(self, palace_path, seeded_collection):
        # The out-of-window drawer has the strongest lexical signal for the
        # query; union mode must not smuggle it past the window.
        seeded_collection.upsert(
            ids=["lex1"],
            documents=["passkeys passkeys passkeys rollout checklist."],
            metadatas=[
                {
                    "wing": "notes",
                    "room": "planning",
                    "source_file": "lex.md",
                    "filed_at": "2026-01-01T00:00:00",
                }
            ],
        )
        result = search_memories(
            "passkeys rollout",
            palace_path,
            n_results=5,
            candidate_strategy="union",
            since="2026-01-02",
        )
        assert "error" not in result
        assert "lex.md" not in [r["source_file"] for r in result["results"]]


class TestCliSearchDateFilter:
    """The printing CLI path accepts the same since/before window."""

    def test_cli_search_since_filters_output(self, palace_path, seeded_collection, capsys):
        search(
            "authentication database frontend sprint planning",
            palace_path,
            n_results=10,
            since="2026-01-04",
        )
        out = capsys.readouterr().out
        assert "sprint.md" in out
        assert "auth.py" not in out
        assert "db.py" not in out

    def test_cli_search_reranks_full_window_pool_before_trim(self, fake_palace_path, capsys):
        # Regression for the review finding: under an active window the CLI
        # must hybrid-re-rank ALL in-window survivors and trim to n_results
        # AFTER the re-rank. A BM25-strong drawer sitting deep in the
        # vector ordering (position 25 of 30) must still surface in the
        # printed top-2; trimming before the re-rank would cut it at
        # position n_results and it could never appear.
        mock_col = MagicMock()
        mock_col.metadata = {"hnsw:space": "cosine"}
        docs, metas, dists = [], [], []
        for i in range(30):
            text = "unrelated filler paragraph number {}".format(i)
            if i == 25:
                text = "quixotic zephyr baseline report"  # exact query tokens
            docs.append(text)
            metas.append(
                {
                    "wing": "w",
                    "room": "r",
                    "source_file": "doc{}.md".format(i),
                    "filed_at": "2026-01-10T00:00:00",
                }
            )
            dists.append(0.30 + i * 0.01)  # strictly increasing vector distance
        mock_col.query.return_value = {
            "documents": [docs],
            "metadatas": [metas],
            "distances": [dists],
        }
        with patch("mempalace.searcher.get_collection", return_value=mock_col):
            search(
                "quixotic zephyr baseline",
                fake_palace_path,
                n_results=2,
                since="2026-01-01",
            )
        out = capsys.readouterr().out
        assert "doc25.md" in out
        # The widened pool was requested from the backend, not just n_results.
        assert mock_col.query.call_args.kwargs["n_results"] > 2

    def test_cli_search_invalid_since_raises_search_error(
        self, palace_path, seeded_collection, capsys
    ):
        import pytest

        with pytest.raises(SearchError, match="since"):
            search("anything", palace_path, since="garbage")

    def test_cli_search_inverted_window_raises(self, palace_path, seeded_collection):
        import pytest

        with pytest.raises(SearchError, match="must be earlier than"):
            search("anything", palace_path, since="2026-01-04", before="2026-01-01")
