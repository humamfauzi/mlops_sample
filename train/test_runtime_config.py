"""Tests for the shared runtime configuration.

The point of ``runtime_config`` is that the training pipeline and the inference
server resolve their storage settings through one code path. Before it existed
the trainer read a ``repository`` block from each train_config while the server
assembled the same structure from flat environment variables, so the two could
silently disagree about which database they were using.

These tests lock in both the resolution rules and the cross-runtime agreement.
"""
import json

import pytest

import runtime_config


@pytest.fixture
def clean_env(monkeypatch):
    """Remove anything that could leak repository settings into a test."""
    for var in tuple(runtime_config.ENV_OVERRIDES) + ("MLOPS_CONFIG",):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def write_config(path, **overrides):
    cfg = {
        "experiment_id": "exp_from_file",
        "column_reference": "sample",
        "data": {"type": "sqlite", "properties": {"name": "from_file.db", "migrate": False}},
        "object": {"type": "sqlite", "properties": {"name": "from_file.db", "migrate": False}},
    }
    cfg.update(overrides)
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


class TestResolution:
    def test_explicit_path_is_used(self, clean_env, tmp_path):
        cfg_file = write_config(tmp_path / "runtime.json")

        cfg = runtime_config.load(path=str(cfg_file))

        assert cfg["experiment_id"] == "exp_from_file"
        assert cfg["data"]["properties"]["name"] == "from_file.db"

    def test_mlops_config_env_var_is_used(self, clean_env, tmp_path):
        cfg_file = write_config(tmp_path / "runtime.json", experiment_id="exp_from_env")
        clean_env.setenv("MLOPS_CONFIG", str(cfg_file))

        assert runtime_config.load()["experiment_id"] == "exp_from_env"

    def test_repo_default_config_is_found(self, clean_env):
        # config/runtime.json ships with the repository, so with no override the
        # loader must locate it rather than falling back to bare defaults.
        cfg = runtime_config.load()

        assert cfg["data"]["type"] == "sqlite"
        assert cfg["object"]["type"] == "sqlite"
        assert "runtime.json" in runtime_config.source()

    def test_environment_overrides_the_file(self, clean_env, tmp_path):
        cfg_file = write_config(tmp_path / "runtime.json")
        clean_env.setenv("REPOSITORY_DATA_PATH", "override.db")
        clean_env.setenv("EXPERIMENT_ID", "exp_overridden")

        cfg = runtime_config.load(path=str(cfg_file))

        assert cfg["experiment_id"] == "exp_overridden"
        assert cfg["data"]["properties"]["name"] == "override.db"
        # untouched keys still come from the file
        assert cfg["object"]["properties"]["name"] == "from_file.db"

    def test_unknown_backend_is_rejected(self, clean_env, tmp_path):
        cfg_file = write_config(
            tmp_path / "runtime.json", data={"type": "postgres", "properties": {}}
        )

        with pytest.raises(ValueError, match="unsupported data backend"):
            runtime_config.load(path=str(cfg_file))

    def test_s3_backend_is_no_longer_accepted(self, clean_env, tmp_path):
        # S3 was retired with the rest of the legacy stack; a config still
        # asking for it should fail loudly rather than at first write.
        cfg_file = write_config(
            tmp_path / "runtime.json", object={"type": "s3", "properties": {}}
        )

        with pytest.raises(ValueError, match="unsupported object backend"):
            runtime_config.load(path=str(cfg_file))

    def test_malformed_json_reports_the_file(self, clean_env, tmp_path):
        bad = tmp_path / "runtime.json"
        bad.write_text("{not json", encoding="utf-8")

        with pytest.raises(ValueError, match="not valid JSON"):
            runtime_config.load(path=str(bad))

    def test_describe_reports_source_and_paths(self, clean_env, tmp_path):
        cfg_file = write_config(tmp_path / "runtime.json")

        described = runtime_config.describe(path=str(cfg_file))

        assert described["source"] == str(cfg_file)
        assert described["data_path"] == "from_file.db"
        assert described["object_backend"] == "sqlite"

    def test_describe_lists_environment_overrides(self, clean_env, tmp_path):
        # A deployment can set every repository value from the environment,
        # leaving the config file irrelevant. Reporting only the file would
        # hide that, so describe() must name the variables that won.
        cfg_file = write_config(tmp_path / "runtime.json")
        clean_env.setenv("REPOSITORY_DATA_PATH", "override.db")
        clean_env.setenv("EXPERIMENT_ID", "exp_overridden")

        described = runtime_config.describe(path=str(cfg_file))

        assert described["environment_overrides"] == {
            "REPOSITORY_DATA_PATH": "override.db",
            "EXPERIMENT_ID": "exp_overridden",
        }
        assert described["data_path"] == "override.db"

    def test_describe_has_no_overrides_when_env_is_clean(self, clean_env, tmp_path):
        cfg_file = write_config(tmp_path / "runtime.json")

        described = runtime_config.describe(path=str(cfg_file))

        assert described["environment_overrides"] == {}


class TestCrossRuntimeAgreement:
    """The regression this whole module exists to prevent."""

    def test_trainer_and_server_resolve_identically(self, clean_env, tmp_path, monkeypatch):
        cfg_file = write_config(tmp_path / "runtime.json", experiment_id="exp_shared")
        clean_env.setenv("MLOPS_CONFIG", str(cfg_file))
        # Run away from the repository root so the checked-in .env does not
        # layer an override on top of the config file under test.
        monkeypatch.chdir(tmp_path)

        from server.main import load_settings
        from train.scenario_manager import InstructionFactory, ScenarioManager

        server_cfg = load_settings()

        instruction = InstructionFactory.parse_instruction(
            {"name": "n", "description": "d", "instructions": []}
        )
        trainer_cfg = ScenarioManager(instruction)._resolve_repository()

        assert trainer_cfg == server_cfg
        assert trainer_cfg["experiment_id"] == "exp_shared"

    def test_both_runtimes_honour_the_same_environment_override(self, clean_env, tmp_path, monkeypatch):
        cfg_file = write_config(tmp_path / "runtime.json")
        clean_env.setenv("MLOPS_CONFIG", str(cfg_file))
        clean_env.setenv("REPOSITORY_OBJECT_PATH", "shared_override.db")
        monkeypatch.chdir(tmp_path)

        from server.main import load_settings
        from train.scenario_manager import InstructionFactory, ScenarioManager

        server_cfg = load_settings()
        instruction = InstructionFactory.parse_instruction(
            {"name": "n", "description": "d", "instructions": []}
        )
        trainer_cfg = ScenarioManager(instruction)._resolve_repository()

        assert trainer_cfg == server_cfg
        assert trainer_cfg["object"]["properties"]["name"] == "shared_override.db"

    def test_checked_in_env_still_overrides_for_both(self, clean_env, tmp_path, monkeypatch):
        # Deployment reality: .env carries per-environment values. Whichever
        # runtime reads it must land on the same answer.
        cfg_file = write_config(tmp_path / "runtime.json", experiment_id="from_file")
        env_file = tmp_path / ".env"
        env_file.write_text("EXPERIMENT_ID=from_dotenv\n", encoding="utf-8")
        clean_env.setenv("MLOPS_CONFIG", str(cfg_file))
        monkeypatch.chdir(tmp_path)

        from server.main import load_settings
        from train.scenario_manager import InstructionFactory, ScenarioManager

        server_cfg = load_settings()
        instruction = InstructionFactory.parse_instruction(
            {"name": "n", "description": "d", "instructions": []}
        )
        trainer_cfg = ScenarioManager(instruction)._resolve_repository()

        assert trainer_cfg == server_cfg
        assert trainer_cfg["experiment_id"] == "from_dotenv"


class TestTrainConfigPrecedence:
    def test_absent_repository_uses_shared_config(self, clean_env, tmp_path):
        cfg_file = write_config(tmp_path / "runtime.json", experiment_id="exp_shared")
        clean_env.setenv("MLOPS_CONFIG", str(cfg_file))

        from train.scenario_manager import InstructionFactory, ScenarioManager

        instruction = InstructionFactory.parse_instruction(
            {"name": "n", "description": "d", "instructions": []}
        )
        assert instruction.repository is None
        assert ScenarioManager(instruction)._resolve_repository()["experiment_id"] == "exp_shared"

    def test_explicit_repository_overrides_shared_config(self, clean_env, tmp_path):
        cfg_file = write_config(tmp_path / "runtime.json", experiment_id="exp_shared")
        clean_env.setenv("MLOPS_CONFIG", str(cfg_file))

        from train.scenario_manager import InstructionFactory, ScenarioManager

        explicit = {
            "experiment_id": "exp_explicit",
            "data": {"type": "sqlite", "properties": {"name": "explicit.db"}},
            "object": {"type": "disk", "properties": {"root": "temp/x"}},
        }
        instruction = InstructionFactory.parse_instruction(
            {"name": "n", "description": "d", "instructions": [], "repository": explicit}
        )

        assert ScenarioManager(instruction)._resolve_repository() == explicit

    def test_empty_repository_still_selects_noop(self, clean_env):
        # The test suite depends on this: "repository": {} must keep every unit
        # test off the real database.
        from train.scenario_manager import InstructionFactory, ScenarioManager

        instruction = InstructionFactory.parse_instruction(
            {"name": "n", "description": "d", "instructions": [], "repository": {}}
        )

        assert ScenarioManager(instruction)._resolve_repository() == {}
