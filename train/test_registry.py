"""Tests for the SQLite registry: nomination, lookup, experiments and blobs.

Covers F-06, F-07 and F-18 from REPOSITORY_MAP.md:

  F-06  the champion/challenger comparison relied on unspecified row order
  F-07  model lookup by the six-character run name was not scoped to a run, so
        a name collision would silently resolve to another run's model
  F-18  runs referenced an experiment id with no row, blobs had no uniqueness,
        and an unused audit_logs table was created on every migration

F-06 and F-07 were latent rather than active -- no collision has occurred and
every published run currently has exactly one test-scored child -- so the tests
below construct the conditions explicitly.
"""
import sqlite3

import pytest

from repositories.repo import Facade
from repositories.sqlite import ObjectStorage, SQLiteRepository
from repositories.struct import ModelObject

EXPERIMENT = "experiment_test"
INTENT = "some_intent"


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "registry.db")


@pytest.fixture
def repo(db_path):
    return SQLiteRepository(name=db_path, migrate=True)


@pytest.fixture
def store(db_path, repo):
    """The artifact half of the same database."""
    return ObjectStorage(name=db_path, migrate=True)


def rows(db_path, query, args=()):
    with sqlite3.connect(db_path) as conn:
        return conn.execute(query, args).fetchall()


def make_published_run(repo, name, intent=INTENT, experiment=EXPERIMENT):
    """A parent run tagged as the published champion for `intent`."""
    run_id = repo.new_run(name, experiment)
    repo.new_property(run_id, "name.intent", intent)
    repo.upsert_tag(run_id, "status.deployment", "published")
    return run_id


def add_child_with_score(repo, parent_id, name, score, metric="validation.test.mae"):
    child_id = repo.new_child_run(name, parent_id, EXPERIMENT)
    repo.new_metric(child_id, metric, score)
    return child_id


class TestSelectPreviouslyPublished:
    def test_returns_none_when_nothing_is_published(self, repo):
        run_id = repo.new_run("AAAAAA", EXPERIMENT)
        repo.new_property(run_id, "name.intent", INTENT)
        add_child_with_score(repo, run_id, "AAAAA1", 0.5)

        assert repo.select_previously_published(EXPERIMENT, INTENT, "mae") == (None, float("inf"))

    def test_returns_the_published_score(self, repo):
        run_id = make_published_run(repo, "AAAAAA")
        add_child_with_score(repo, run_id, "AAAAA1", 0.42)

        selected_id, score = repo.select_previously_published(EXPERIMENT, INTENT, "mae")

        assert selected_id == run_id
        assert score == pytest.approx(0.42)

    def test_picks_the_best_child_regardless_of_insertion_order(self, repo):
        # F-06. A published run with several test-scored children used to return
        # whichever row SQLite happened to hand back first -- here 0.9, the
        # worst of the three. The comparison must be against the run's *best*
        # child, deterministically.
        run_id = make_published_run(repo, "AAAAAA")
        for i, score in enumerate([0.90, 0.50, 0.70]):
            add_child_with_score(repo, run_id, f"AAAAA{i}", score)

        selected_id, score = repo.select_previously_published(EXPERIMENT, INTENT, "mae")

        assert selected_id == run_id
        assert score == pytest.approx(0.50)

    def test_is_stable_across_repeated_calls(self, repo):
        run_id = make_published_run(repo, "AAAAAA")
        for i, score in enumerate([0.9, 0.5, 0.7, 0.6]):
            add_child_with_score(repo, run_id, f"AAAAA{i}", score)

        results = [repo.select_previously_published(EXPERIMENT, INTENT, "mae") for _ in range(5)]

        assert len(set(results)) == 1

    def test_only_the_matching_intent_competes(self, repo):
        # Two intents are independent competitions; nominating into one must not
        # see the other's champion.
        mine = make_published_run(repo, "AAAAAA", intent="intent_a")
        add_child_with_score(repo, mine, "AAAAA1", 0.80)
        other = make_published_run(repo, "BBBBBB", intent="intent_b")
        add_child_with_score(repo, other, "BBBBB1", 0.10)

        selected_id, score = repo.select_previously_published(EXPERIMENT, "intent_a", "mae")

        assert selected_id == mine
        assert score == pytest.approx(0.80)

    def test_only_the_matching_experiment_competes(self, repo):
        mine = make_published_run(repo, "AAAAAA", experiment="experiment_a")
        add_child_with_score(repo, mine, "AAAAA1", 0.80)
        other = make_published_run(repo, "BBBBBB", experiment="experiment_b")
        add_child_with_score(repo, other, "BBBB1", 0.10)

        selected_id, _ = repo.select_previously_published("experiment_a", INTENT, "mae")

        assert selected_id == mine

    def test_ignores_retracted_and_inferior_runs(self, repo):
        retracted = repo.new_run("AAAAAA", EXPERIMENT)
        repo.new_property(retracted, "name.intent", INTENT)
        repo.upsert_tag(retracted, "status.deployment", "retracted")
        add_child_with_score(repo, retracted, "AAAAA1", 0.10)

        assert repo.select_previously_published(EXPERIMENT, INTENT, "mae") == (None, float("inf"))


class TestModelRunLookup:
    def test_finds_a_child_within_its_parent(self, repo):
        parent = repo.new_run("AAAAAA", EXPERIMENT)
        child = add_child_with_score(repo, parent, "CHILD1", 0.5)

        found_id, found_parent = repo.get_model_run_id("CHILD1", parent)

        assert (found_id, found_parent) == (child, parent)

    def test_raises_when_the_name_is_not_under_that_parent(self, repo):
        # F-07. An unscoped lookup would happily return the other run's model.
        parent_a = repo.new_run("AAAAAA", EXPERIMENT)
        child_a = add_child_with_score(repo, parent_a, "SHARED", 0.5)
        parent_b = repo.new_run("BBBBBB", EXPERIMENT)

        with pytest.raises(ValueError, match="not found under parent run"):
            repo.get_model_run_id("SHARED", parent_b)

        # and it really is reachable under the parent it belongs to
        assert repo.get_model_run_id("SHARED", parent_a)[0] == child_a

    def test_colliding_names_resolve_to_the_right_run(self, repo):
        # The actual collision scenario: two runs each generated a child named
        # "SAMENAME". Scoping is what keeps them apart.
        parent_a = repo.new_run("AAAAAA", EXPERIMENT)
        child_a = add_child_with_score(repo, parent_a, "SAMENA", 0.1)
        parent_b = repo.new_run("BBBBBB", EXPERIMENT)
        child_b = repo.new_child_run("SAMENA", parent_b, EXPERIMENT)
        repo.new_metric(child_b, "validation.test.mae", 0.2)

        assert repo.get_model_run_id("SAMENA", parent_b)[0] == child_b
        assert repo.get_model_run_id("SAMENA", parent_a)[0] == child_a

    def test_raises_for_an_unknown_name(self, repo):
        parent = repo.new_run("AAAAAA", EXPERIMENT)

        with pytest.raises(ValueError, match="not found"):
            repo.get_model_run_id("NOPE00", parent)


class TestRunNameUniqueness:
    def test_run_name_exists(self, repo):
        repo.new_run("AAAAAA", EXPERIMENT)

        assert repo.run_name_exists("AAAAAA") is True
        assert repo.run_name_exists("ZZZZZZ") is False

    def test_generate_run_id_avoids_existing_names(self, repo, monkeypatch):
        # Force a collision on the first candidate so the retry path is exercised
        # rather than merely asserted to be reachable. random.choice is called
        # once per character, so the first six calls yield "AAAAAA" (taken) and
        # the next six yield "BBBBBB" (free).
        import repositories.repo as repo_module

        repo.new_child_run("AAAAAA", repo.new_run("PARENT", EXPERIMENT), EXPERIMENT)
        calls = {"n": 0}

        def fake_choice(_seq):
            calls["n"] += 1
            return "A" if calls["n"] <= 6 else "B"

        monkeypatch.setattr(repo_module.random, "choice", fake_choice)
        facade = Facade(EXPERIMENT, repo, repo)

        assert facade.generate_run_id() == "BBBBBB"
        assert calls["n"] == 12  # two full candidates were drawn

    def test_generate_run_id_gives_up_with_a_clear_error(self, repo, monkeypatch):
        import repositories.repo as repo_module

        monkeypatch.setattr(repo_module.random, "choice", lambda seq: "A")
        repo.new_run("AAAAAA", EXPERIMENT)

        facade = Facade(EXPERIMENT, repo, repo)

        with pytest.raises(RuntimeError, match="unique run id"):
            facade.generate_run_id(attempts=5)

    def test_generated_ids_look_like_run_names(self, repo):
        facade = Facade(EXPERIMENT, repo, repo)

        generated = {facade.generate_run_id() for _ in range(20)}

        assert all(len(g) == 6 for g in generated)
        # the malformed alphabet had a duplicated '8' and no '0'
        assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890" for g in generated for c in g)


class TestExperimentRegistration:
    """F-18: runs referenced an experiment_id that never existed as a row.

    The experiments table stayed empty while 91 runs pointed at
    'experiment_001', so a typo in EXPERIMENT_ID would silently start a
    parallel universe of runs with nothing to catch it.
    """

    def test_ensure_experiment_registers_the_id(self, repo, db_path):
        repo.ensure_experiment("exp_a")

        assert rows(db_path, "SELECT id FROM experiments") == [("exp_a",)]

    def test_ensure_experiment_is_idempotent(self, repo, db_path):
        repo.ensure_experiment("exp_a")
        repo.ensure_experiment("exp_a")
        repo.ensure_experiment("exp_a")

        assert rows(db_path, "SELECT id FROM experiments") == [("exp_a",)]

    def test_creating_a_run_registers_its_experiment(self, repo, db_path):
        # Going through the facade is what the training pipeline does.
        Facade("exp_via_facade", repo, repo).new_run("RUN001")

        assert ("exp_via_facade",) in rows(db_path, "SELECT id FROM experiments")

    def test_every_run_experiment_has_a_row(self, db_path):
        facade = Facade("exp_a", SQLiteRepository(name=db_path, migrate=True), None)
        facade.new_run("RUN001")
        facade.new_run("RUN002")

        known = {r[0] for r in rows(db_path, "SELECT id FROM experiments")}
        referenced = {r[0] for r in rows(db_path, "SELECT DISTINCT experiment_id FROM runs")}

        assert referenced <= known

    def test_audit_logs_table_is_gone(self, repo, db_path):
        # It was created on every migrate and never written to.
        tables = {r[0] for r in rows(db_path, "SELECT name FROM sqlite_master WHERE type='table'")}

        assert "audit_logs" not in tables


class TestBlobStorage:
    """F-18: blobs were keyed by (run_id, intent) with no uniqueness."""

    def test_saving_the_same_artifact_twice_stores_one_row(self, store, db_path):
        store.save_model(1, ModelObject(filename="MODEL1", object={"v": 1}))
        store.save_model(1, ModelObject(filename="MODEL1", object={"v": 2}))

        assert rows(db_path, "SELECT COUNT(*) FROM blobs") == [(1,)]

    def test_the_latest_value_wins(self, store):
        store.save_model(1, ModelObject(filename="MODEL1", object={"v": 1}))
        store.save_model(1, ModelObject(filename="MODEL1", object={"v": 2}))

        assert store.load_model(1, "MODEL1").object == {"v": 2}

    def test_distinct_artifacts_coexist(self, store, db_path):
        store.save_model(1, ModelObject(filename="MODEL1", object={"v": 1}))
        store.save_model(1, ModelObject(filename="MODEL2", object={"v": 2}))
        store.save_model(2, ModelObject(filename="MODEL1", object={"v": 3}))

        assert rows(db_path, "SELECT COUNT(*) FROM blobs") == [(3,)]

    def test_unique_index_exists(self, store, db_path):
        indexes = {r[0] for r in rows(db_path, "SELECT name FROM sqlite_master WHERE type='index'")}

        assert "idx_blobs_runid_intent" in indexes

    def test_transformation_objects_report_what_they_wrote(self, store):
        from repositories.struct import TransformationObject

        written = store.save_transformation_object(1, [
            TransformationObject(filename="ohe-00-a.pkl", object={"a": 1}),
            TransformationObject(filename="log-01-b.pkl", object={"b": 2}),
        ])

        assert written == ["ohe-00-a.pkl", "log-01-b.pkl"]
