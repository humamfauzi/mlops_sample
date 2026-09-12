"""Tests for scripts/registry.py.

The prune command deletes trained model pickles from the only copy of the
registry, so its selection rule and its safety guard both need locking down:
it must never remove a blob the server loads, and it must refuse to produce an
output that would break serving.
"""
import pathlib
import sqlite3
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import registry  # noqa: E402

MIB = 1024 * 1024


def blob(size):
    return b"x" * size


@pytest.fixture
def make_registry(tmp_path):
    """Build a small registry with a controllable run/tag/blob layout."""
    counter = {"n": 0}

    def _build(published_with_best=True):
        counter["n"] += 1
        path = tmp_path / f"registry{counter['n']}.db"
        conn = sqlite3.connect(path)
        conn.executescript(
            """
            CREATE TABLE runs (id INTEGER PRIMARY KEY, name TEXT, parent_id INTEGER, experiment_id TEXT);
            CREATE TABLE tags (id INTEGER PRIMARY KEY, run_id INTEGER, key TEXT, value TEXT);
            CREATE TABLE metrics (id INTEGER PRIMARY KEY, run_id INTEGER, key TEXT, value FLOAT);
            CREATE TABLE properties (id INTEGER PRIMARY KEY, run_id INTEGER, key TEXT, value TEXT);
            CREATE TABLE objects (id INTEGER PRIMARY KEY, run_id INTEGER, type TEXT, url TEXT);
            CREATE TABLE experiments (id TEXT PRIMARY KEY, name TEXT);
            CREATE TABLE blobs (id INTEGER PRIMARY KEY, run_id INTEGER, intent TEXT, type TEXT, data BLOB);
            CREATE UNIQUE INDEX idx_blobs_runid_intent ON blobs(run_id, intent);
            """
        )
        conn.commit()
        conn.close()
        return str(path)

    return _build


def add_run(db, name, parent=None):
    with sqlite3.connect(db) as conn:
        cur = conn.execute(
            "INSERT INTO runs (name, parent_id, experiment_id) VALUES (?, ?, 'exp')",
            (name, parent),
        )
        return cur.lastrowid


def tag(db, run_id, key, value):
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO tags (run_id, key, value) VALUES (?, ?, ?)", (run_id, key, value))


def add_model(db, run_id, name, size):
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO blobs (run_id, intent, type, data) VALUES (?, ?, 'pkl', ?)",
            (run_id, f"model/{name}", blob(size)),
        )


def add_transform(db, run_id, name="log-00-col.pkl"):
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO blobs (run_id, intent, type, data) VALUES (?, ?, 'pkl', ?)",
            (run_id, f"transformation_object/{name}", b"t"),
        )


def intents(db):
    with sqlite3.connect(db) as conn:
        return {r[0] for r in conn.execute("SELECT intent FROM blobs")}


def build_published_champion(db):
    """A published parent whose best child holds a loadable model."""
    parent = add_run(db, "PARENT")
    tag(db, parent, "status.deployment", "published")
    best = add_run(db, "BEST01", parent=parent)
    tag(db, best, "level", "best")
    add_model(db, best, "BEST01", 1000)
    add_transform(db, best)
    return parent, best


class TestReportAndVerify:
    def test_verify_passes_on_a_healthy_registry(self, make_registry):
        db = make_registry()
        build_published_champion(db)

        assert registry.main(["verify", db]) == 0

    def test_verify_fails_when_a_published_model_is_missing(self, make_registry):
        db = make_registry()
        parent, best = build_published_champion(db)
        with sqlite3.connect(db) as conn:
            conn.execute("DELETE FROM blobs")

        assert registry.main(["verify", db]) == 1

    def test_report_runs(self, make_registry, capsys):
        db = make_registry()
        build_published_champion(db)

        assert registry.main(["report", db]) == 0
        assert "model" in capsys.readouterr().out


class TestSnapshot:
    def test_snapshot_is_a_usable_copy(self, make_registry):
        db = make_registry()
        build_published_champion(db)
        dest = str(pathlib.Path(db).parent / "snap.db")

        assert registry.main(["snapshot", db, dest]) == 0
        assert registry.main(["verify", dest]) == 0

    def test_snapshot_refuses_to_overwrite_by_default(self, make_registry):
        db = make_registry()
        dest = str(pathlib.Path(db).parent / "snap.db")
        pathlib.Path(dest).write_text("existing")

        with pytest.raises(SystemExit, match="refusing to overwrite"):
            registry.main(["snapshot", db, dest])

    def test_snapshot_force_overwrites(self, make_registry):
        db = make_registry()
        build_published_champion(db)
        dest = str(pathlib.Path(db).parent / "snap.db")
        pathlib.Path(dest).write_text("existing")

        assert registry.main(["snapshot", db, dest, "--force"]) == 0
        assert registry.main(["verify", dest]) == 0


class TestPruneSelection:
    def test_drops_models_from_non_best_children(self, make_registry):
        db = make_registry()
        parent, best = build_published_champion(db)
        loser = add_run(db, "LOSER1", parent=parent)
        add_model(db, loser, "LOSER1", 5 * MIB)
        dest = str(pathlib.Path(db).parent / "out.db")

        assert registry.main(["prune", db, dest]) == 0
        assert "model/BEST01" in intents(dest)
        assert "model/LOSER1" not in intents(dest)

    def test_keeps_transformation_blobs(self, make_registry):
        db = make_registry()
        build_published_champion(db)
        dest = str(pathlib.Path(db).parent / "out.db")

        registry.main(["prune", db, dest])

        assert any(i.startswith("transformation_object/") for i in intents(dest))

    def test_keeps_models_of_a_run_with_no_best_child(self, make_registry):
        # A run that failed part way produced no best child; its models are the
        # only record of what it managed to train.
        db = make_registry()
        build_published_champion(db)
        orphan_parent = add_run(db, "ORPHAN")
        only_child = add_run(db, "ONLY01", parent=orphan_parent)
        add_model(db, only_child, "ONLY01", 2 * MIB)
        dest = str(pathlib.Path(db).parent / "out.db")

        registry.main(["prune", db, dest])

        assert "model/ONLY01" in intents(dest)

    def test_include_incomplete_drops_those_too(self, make_registry):
        db = make_registry()
        build_published_champion(db)
        orphan_parent = add_run(db, "ORPHAN")
        only_child = add_run(db, "ONLY01", parent=orphan_parent)
        add_model(db, only_child, "ONLY01", 2 * MIB)
        dest = str(pathlib.Path(db).parent / "out.db")

        registry.main(["prune", db, dest, "--include-incomplete"])

        assert "model/ONLY01" not in intents(dest)

    def test_every_published_model_survives(self, make_registry):
        db = make_registry()
        build_published_champion(db)
        for i in range(5):
            parent = add_run(db, f"SON{i:03d}")
            loser = add_run(db, f"LOS{i:03d}", parent=parent)
            add_model(db, loser, f"LOS{i:03d}", MIB)
        dest = str(pathlib.Path(db).parent / "out.db")

        registry.main(["prune", db, dest])

        assert registry.main(["verify", dest]) == 0

    def test_dry_run_writes_nothing(self, make_registry):
        db = make_registry()
        build_published_champion(db)
        parent = add_run(db, "P2")
        loser = add_run(db, "LOSER2", parent=parent)
        add_model(db, loser, "LOSER2", MIB)
        dest = pathlib.Path(db).parent / "out.db"

        registry.main(["prune", db, str(dest), "--dry-run"])

        assert not dest.exists()

    def test_source_is_never_modified(self, make_registry):
        db = make_registry()
        parent, best = build_published_champion(db)
        loser = add_run(db, "LOSER3", parent=parent)
        add_model(db, loser, "LOSER3", MIB)
        before = intents(db)
        dest = str(pathlib.Path(db).parent / "out.db")

        registry.main(["prune", db, dest])

        assert intents(db) == before

    def test_refuses_to_produce_an_output_that_breaks_serving(self, make_registry):
        # A published parent whose best child has no model at all. Pruning must
        # not quietly emit a registry the server cannot start from.
        db = make_registry()
        parent = add_run(db, "PARENT")
        tag(db, parent, "status.deployment", "published")
        best = add_run(db, "BEST01", parent=parent)
        tag(db, best, "level", "best")
        # deliberately no model blob for the best child
        other = add_run(db, "OTHER1", parent=parent)
        add_model(db, other, "OTHER1", MIB)
        dest = str(pathlib.Path(db).parent / "out.db")

        assert registry.main(["verify", db]) == 1

    def test_prune_reports_failure_when_the_invariant_is_already_broken(
        self, make_registry, capsys
    ):
        db = make_registry()
        parent = add_run(db, "PARENT")
        tag(db, parent, "status.deployment", "published")
        best = add_run(db, "BEST01", parent=parent)
        tag(db, best, "level", "best")  # best child has no blob
        other = add_run(db, "OTHER1", parent=parent)
        add_model(db, other, "OTHER1", MIB)
        dest = str(pathlib.Path(db).parent / "out.db")

        assert registry.main(["prune", db, dest]) == 1
        assert "WOULD BREAK SERVING" in capsys.readouterr().out
