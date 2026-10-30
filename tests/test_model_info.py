"""S4.6 -- ``modules/model_info.py``.

The registry is the Chatterbox counterpart of VoxCPM's: a static config table,
a live registry, filesystem scanning, and two feature flags that must degrade
to ``False`` rather than break node import.

The scan tests are the interesting ones. A Chatterbox pack is not a
``config.json`` with an ``architecture`` key -- it is a tokenizer plus
safetensors -- so detection is file-based and the "rejects one missing weights"
case is the one that has teeth.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ComfyUI_Chatterbox.modules import model_info
from ComfyUI_Chatterbox.modules.constants import CHATTERBOX_FILES_TO_DOWNLOAD, CHATTERBOX_REPO_ID
from ComfyUI_Chatterbox.modules.model_info import (
    AVAILABLE_CHATTERBOX_MODELS,
    CHATTERBOX_WEIGHT_FILES,
    MODEL_CONFIGS,
    TEXT_NORMALIZATION_AVAILABLE,
    TURBO_AVAILABLE,
    describe_model_dir,
    is_chatterbox_model_dir,
    read_model_config,
    refresh_feature_flags,
    refresh_local_models,
    register_local_model,
    scan_chatterbox_models,
    scan_custom_model_path,
    unregister_local_model,
)

REQUIRED_OFFICIAL_MODELS = (
    "chatterbox",
    "T3_v1",
    "T3_v2",
    "T3_v3",
    "chatterbox-turbo",
    "chatterbox-nano",
)


def write_pack(parent, name: str, weights=("t3_cfg.safetensors",), tokenizer=True) -> str:
    pack = parent / name
    pack.mkdir(parents=True, exist_ok=True)
    if tokenizer:
        (pack / "tokenizer.json").write_text('{"model": {}}', encoding="utf-8")
    for weight in weights:
        (pack / weight).write_bytes(b"weights")
    return str(pack)


@pytest.fixture(autouse=True)
def restore_registry():
    """The registry is module state; leave only the official entries behind."""
    yield
    for name in list(AVAILABLE_CHATTERBOX_MODELS):
        if AVAILABLE_CHATTERBOX_MODELS[name].get("type") == "local":
            AVAILABLE_CHATTERBOX_MODELS.pop(name)


class TestOfficialRegistry:
    @pytest.mark.parametrize("name", REQUIRED_OFFICIAL_MODELS)
    def test_every_documented_model_is_present(self, name: str) -> None:
        assert name in MODEL_CONFIGS

    @pytest.mark.parametrize("name", REQUIRED_OFFICIAL_MODELS)
    def test_official_entries_carry_the_full_metadata(self, name: str) -> None:
        config = MODEL_CONFIGS[name]
        assert config["repo_id"], f"{name} has no repo_id; it could never be downloaded"
        assert config["architecture"] in model_info.VALID_CHATTERBOX_ARCHITECTURES
        assert config["sample_rate"] == 24000
        assert config["size_gb"] > 0

    def test_the_default_pack_uses_the_frozen_repo(self) -> None:
        assert MODEL_CONFIGS["chatterbox"]["repo_id"] == CHATTERBOX_REPO_ID

    def test_the_default_pack_downloads_the_frozen_file_list(self) -> None:
        assert MODEL_CONFIGS["chatterbox"]["files"] == CHATTERBOX_FILES_TO_DOWNLOAD

    def test_turbo_and_nano_come_from_their_own_repositories(self) -> None:
        assert MODEL_CONFIGS["chatterbox-turbo"]["repo_id"] == "ResembleAI/chatterbox-turbo"
        assert MODEL_CONFIGS["chatterbox-nano"]["repo_id"] == "ResembleAI/chatterbox-nano"

    def test_the_multilingual_variants_are_distinct_checkpoints(self) -> None:
        files = [MODEL_CONFIGS[name]["files"][0] for name in ("T3_v1", "T3_v2", "T3_v3")]
        assert len(set(files)) == 3, files
        assert all(name.endswith(".safetensors") for name in files)

    @pytest.mark.parametrize("name", REQUIRED_OFFICIAL_MODELS)
    def test_available_registry_marks_them_official(self, name: str) -> None:
        entry = AVAILABLE_CHATTERBOX_MODELS[name]
        assert entry["type"] == "official"
        assert entry["repo_id"] == MODEL_CONFIGS[name]["repo_id"]

    def test_every_official_file_list_names_real_weights(self) -> None:
        for name, config in MODEL_CONFIGS.items():
            known = [
                path
                for path in config["files"]
                if path in CHATTERBOX_WEIGHT_FILES or path == "tokenizer.json"
            ]
            assert known, f"{name} lists no recognised weight file: {config['files']}"


class TestWhatIsAPack:
    """The registry holds two kinds of entry: downloadable packs, and the T3
    checkpoints that live inside the classic repository and are picked out of
    a pack directory rather than downloaded as one. The dropdown may only
    offer the former."""

    def test_the_t3_checkpoints_are_marked_as_not_packs(self) -> None:
        for name in ("T3_v1", "T3_v2", "T3_v3"):
            assert MODEL_CONFIGS[name].get("is_pack") is False, (
                f"{name} is a checkpoint inside a pack, not a pack: it must not "
                "be offered as a downloadable pack"
            )

    def test_every_other_entry_is_a_pack(self) -> None:
        for name, config in MODEL_CONFIGS.items():
            if name in ("T3_v1", "T3_v2", "T3_v3"):
                continue
            assert config.get("is_pack", True) is True, f"{name} is not marked as a pack"

    def test_official_pack_names_lists_every_downloadable_pack(self) -> None:
        assert model_info.official_pack_names() == [
            name
            for name, config in MODEL_CONFIGS.items()
            if config.get("is_pack", True)
        ]

    def test_the_pack_list_carries_no_checkpoint(self) -> None:
        names = model_info.official_pack_names()
        for name in ("T3_v1", "T3_v2", "T3_v3"):
            assert name not in names

    def test_every_offered_pack_can_be_downloaded(self) -> None:
        for name in model_info.official_pack_names():
            assert MODEL_CONFIGS[name].get("repo_id"), (
                f"the offered pack '{name}' has no repo_id, so it can never be "
                "downloaded"
            )


class TestLocalRegistry:
    def test_a_registered_local_model_carries_its_path(self, tmp_path) -> None:
        pack = write_pack(tmp_path, "mine")
        register_local_model("mine", pack)

        entry = AVAILABLE_CHATTERBOX_MODELS["mine"]
        assert entry["type"] == "local"
        assert entry["path"] == pack
        assert entry["repo_id"] == ""

    def test_local_entries_are_separated_from_official_ones(self, tmp_path) -> None:
        register_local_model("mine", write_pack(tmp_path, "mine"))
        assert AVAILABLE_CHATTERBOX_MODELS["mine"]["type"] == "local"
        assert AVAILABLE_CHATTERBOX_MODELS["chatterbox-turbo"]["type"] == "official"

    def test_an_official_entry_is_never_unregistered(self) -> None:
        assert unregister_local_model("chatterbox-turbo") is False
        assert "chatterbox-turbo" in AVAILABLE_CHATTERBOX_MODELS

    def test_a_local_entry_can_be_unregistered(self, tmp_path) -> None:
        register_local_model("mine", write_pack(tmp_path, "mine"))
        assert unregister_local_model("mine") is True
        assert "mine" not in AVAILABLE_CHATTERBOX_MODELS

    def test_refresh_local_models_mirrors_the_models_root(self, tmp_path, monkeypatch) -> None:
        import folder_paths

        root = tmp_path / "tts" / "chatterbox"
        write_pack(root, "pack_a")
        write_pack(root, "pack_b")
        monkeypatch.setattr(folder_paths, "get_folder_paths", lambda name: [str(tmp_path / "tts")])

        found = refresh_local_models()
        names = {entry["name"] for entry in found}
        assert names == {"pack_a", "pack_b"}
        assert AVAILABLE_CHATTERBOX_MODELS["pack_a"]["path"] == str(root / "pack_a")

    def test_refresh_local_models_drops_packs_that_disappeared(
        self, tmp_path, monkeypatch
    ) -> None:
        import folder_paths
        import shutil

        from ComfyUI_Chatterbox.modules import model_discovery

        root = tmp_path / "tts" / "chatterbox"
        write_pack(root, "pack_a")
        write_pack(root, "pack_b")
        monkeypatch.setattr(folder_paths, "get_folder_paths", lambda name: [str(tmp_path / "tts")])
        model_discovery.invalidate_cache()

        refresh_local_models()
        assert "pack_a" in AVAILABLE_CHATTERBOX_MODELS

        shutil.rmtree(root / "pack_b")
        model_discovery.invalidate_cache()
        refresh_local_models()

        assert "pack_b" not in AVAILABLE_CHATTERBOX_MODELS
        assert "pack_a" in AVAILABLE_CHATTERBOX_MODELS


class TestReadModelConfig:
    def test_reads_a_config_json(self, tmp_path) -> None:
        (tmp_path / "config.json").write_text(
            json.dumps({"architecture": "chatterbox_turbo"}), encoding="utf-8"
        )
        assert read_model_config(str(tmp_path)) == {"architecture": "chatterbox_turbo"}

    def test_returns_none_when_there_is_no_config(self, tmp_path) -> None:
        assert read_model_config(str(tmp_path)) is None

    def test_returns_none_for_corrupt_json(self, tmp_path) -> None:
        (tmp_path / "config.json").write_text("{oops", encoding="utf-8")
        assert read_model_config(str(tmp_path)) is None

    def test_returns_none_for_a_missing_directory(self, tmp_path) -> None:
        assert read_model_config(str(tmp_path / "nope")) is None


class TestDescribeModelDir:
    def test_infers_the_architecture_from_the_weights(self, tmp_path) -> None:
        pack = write_pack(tmp_path, "pack", weights=("t3_turbo_v1.safetensors",))
        assert describe_model_dir(pack)["architecture"] == "chatterbox_turbo"

    def test_infers_nano(self, tmp_path) -> None:
        pack = write_pack(tmp_path, "pack", weights=("t3_nano_v1.safetensors",))
        assert describe_model_dir(pack)["architecture"] == "chatterbox_nano"

    def test_infers_multilingual(self, tmp_path) -> None:
        pack = write_pack(tmp_path, "pack", weights=("t3_mtl23ls_v3.safetensors",))
        assert describe_model_dir(pack)["architecture"] == "chatterbox_multilingual"

    def test_an_explicit_config_wins_over_the_filenames(self, tmp_path) -> None:
        pack = Path(write_pack(tmp_path, "pack", weights=("t3_nano_v1.safetensors",)))
        (pack / "config.json").write_text(
            json.dumps({"architecture": "chatterbox_turbo"}), encoding="utf-8"
        )
        assert describe_model_dir(str(pack))["architecture"] == "chatterbox_turbo"

    def test_an_unrecognisable_directory_reports_unknown(self, tmp_path) -> None:
        pack = write_pack(tmp_path, "pack", weights=("something.bin",))
        assert describe_model_dir(pack)["architecture"] == "unknown"

    def test_carries_name_and_path(self, tmp_path) -> None:
        pack = write_pack(tmp_path, "pack")
        description = describe_model_dir(pack)
        assert description["name"] == "pack"
        assert description["path"] == pack


class TestIsChatterboxModelDir:
    def test_accepts_a_complete_pack(self, tmp_path) -> None:
        pack = write_pack(tmp_path, "pack", weights=("t3_cfg.safetensors", "ve.safetensors"))
        assert is_chatterbox_model_dir(pack) is True

    @pytest.mark.parametrize(
        "weights,tokenizer,reason",
        [
            ((), True, "no weights at all"),
            (("t3_cfg.safetensors",), False, "no tokenizer"),
            ((), False, "neither weights nor tokenizer"),
        ],
    )
    def test_rejects_an_incomplete_pack(self, tmp_path, weights, tokenizer, reason) -> None:
        pack = write_pack(tmp_path, "pack", weights=weights, tokenizer=tokenizer)
        assert is_chatterbox_model_dir(pack) is False, reason

    def test_rejects_weights_without_a_tokenizer(self, tmp_path) -> None:
        pack = write_pack(tmp_path, "pack", weights=("t3_cfg.safetensors",), tokenizer=False)
        assert is_chatterbox_model_dir(pack) is False

    def test_rejects_a_foreign_architecture(self, tmp_path) -> None:
        pack = Path(write_pack(tmp_path, "pack"))
        (pack / "config.json").write_text(
            json.dumps({"architecture": "whisper"}), encoding="utf-8"
        )
        assert is_chatterbox_model_dir(str(pack)) is False

    def test_rejects_an_empty_directory(self, tmp_path) -> None:
        (tmp_path / "empty").mkdir()
        assert is_chatterbox_model_dir(str(tmp_path / "empty")) is False

    def test_rejects_a_file(self, tmp_path) -> None:
        target = tmp_path / "a-file"
        target.write_text("x", encoding="utf-8")
        assert is_chatterbox_model_dir(str(target)) is False

    def test_rejects_a_missing_path(self, tmp_path) -> None:
        assert is_chatterbox_model_dir(str(tmp_path / "nope")) is False
        assert is_chatterbox_model_dir("") is False

    def test_accepts_a_pack_whose_weights_name_is_unknown(self, tmp_path) -> None:
        """A community checkpoint with a renamed blob is still usable, as long
        as the tokenizer is there and at least one safetensors exists."""
        pack = write_pack(tmp_path, "pack", weights=("community_weights.safetensors",))
        assert is_chatterbox_model_dir(pack) is True


class TestScanChatterboxModels:
    def test_finds_a_valid_directory(self, tmp_path) -> None:
        pack = write_pack(tmp_path, "good")
        models = scan_chatterbox_models(str(tmp_path))
        assert [entry["name"] for entry in models] == ["good"]
        assert models[0]["path"] == pack
        assert models[0]["architecture"] == "chatterbox_tts"

    def test_rejects_one_missing_weights(self, tmp_path) -> None:
        write_pack(tmp_path, "good")
        write_pack(tmp_path, "half", weights=(), tokenizer=True)

        names = [entry["name"] for entry in scan_chatterbox_models(str(tmp_path))]
        assert names == ["good"], f"an incomplete pack was offered as a model: {names}"

    def test_ignores_loose_files(self, tmp_path) -> None:
        write_pack(tmp_path, "good")
        (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
        (tmp_path / "ve.safetensors").write_bytes(b"x")

        assert [entry["name"] for entry in scan_chatterbox_models(str(tmp_path))] == ["good"]

    def test_results_are_sorted(self, tmp_path) -> None:
        for name in ("zeta", "alpha", "mu"):
            write_pack(tmp_path, name)
        names = [entry["name"] for entry in scan_chatterbox_models(str(tmp_path))]
        assert names == sorted(names)

    def test_a_missing_directory_yields_an_empty_list(self, tmp_path) -> None:
        assert scan_chatterbox_models(str(tmp_path / "nope")) == []

    def test_an_empty_root_yields_an_empty_list(self, tmp_path) -> None:
        assert scan_chatterbox_models(str(tmp_path)) == []

    def test_every_entry_has_the_documented_keys(self, tmp_path) -> None:
        write_pack(tmp_path, "good")
        entry = scan_chatterbox_models(str(tmp_path))[0]
        assert {"name", "architecture", "path"} <= set(entry)


class TestScanCustomModelPath:
    def test_scans_children_of_a_parent_directory(self, tmp_path) -> None:
        write_pack(tmp_path, "good")
        assert [entry["name"] for entry in scan_custom_model_path(str(tmp_path))] == ["good"]

    def test_accepts_a_path_pointing_straight_at_a_pack(self, tmp_path) -> None:
        pack = write_pack(tmp_path, "good")
        models = scan_custom_model_path(pack)
        assert len(models) == 1
        assert models[0]["name"] == "good"
        assert models[0]["path"] == pack

    def test_a_missing_path_yields_an_empty_list(self, tmp_path) -> None:
        assert scan_custom_model_path(str(tmp_path / "nope")) == []

    def test_a_blank_path_yields_an_empty_list(self) -> None:
        assert scan_custom_model_path("") == []


class TestFeatureFlags:
    def test_flags_are_booleans_at_import(self) -> None:
        assert isinstance(TEXT_NORMALIZATION_AVAILABLE, bool)
        assert isinstance(TURBO_AVAILABLE, bool)

    def test_refresh_recomputes_them(self, monkeypatch) -> None:
        monkeypatch.setattr(model_info, "_probe_text_normalization", lambda: True)
        monkeypatch.setattr(model_info, "_probe_turbo", lambda: False)
        refresh_feature_flags()
        assert model_info.TEXT_NORMALIZATION_AVAILABLE is True
        assert model_info.TURBO_AVAILABLE is False

    def test_text_normalization_degrades_to_false_when_no_segmenter(self, monkeypatch) -> None:
        import importlib.util

        real_find_spec = importlib.util.find_spec

        def nothing_installed(name, *args, **kwargs):
            if name in ("pykakasi", "spacy_pkuseg"):
                return None
            return real_find_spec(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, "find_spec", nothing_installed)
        assert model_info._probe_text_normalization() is False

    def test_text_normalization_is_true_when_a_segmenter_is_present(self, monkeypatch) -> None:
        import importlib.util

        real_find_spec = importlib.util.find_spec

        def kk_present(name, *args, **kwargs):
            if name == "pykakasi":
                return object()
            return real_find_spec(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, "find_spec", kk_present)
        assert model_info._probe_text_normalization() is True

    def test_a_mocked_optional_package_does_not_raise(self, monkeypatch) -> None:
        import importlib.util

        def exploding(name, *args, **kwargs):
            raise ValueError("__spec__ is None")

        monkeypatch.setattr(importlib.util, "find_spec", exploding)
        assert model_info._probe_text_normalization() is False

    def test_turbo_is_probed_on_disk_not_by_import(self, tmp_path) -> None:
        """Importing ``chatterbox.tts_turbo`` would load torch and transformers;
        the probe must stay a file check."""
        assert model_info._probe_turbo() is True, (
            "the vendored tts_turbo.py entry point should be found on disk"
        )

    def test_turbo_is_false_without_the_entry_point(self, monkeypatch) -> None:
        monkeypatch.setattr(os.path, "isfile", lambda path: False)
        assert model_info._probe_turbo() is False


class TestModuleHygiene:
    def test_no_comfy_api_import(self) -> None:
        with open(model_info.__file__, "r", encoding="utf-8") as handle:
            assert "import comfy_api" not in handle.read()

    def test_no_module_scope_comfy_imports(self) -> None:
        import ast

        with open(model_info.__file__, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in ("folder_paths", "server"), alias.name
                    assert not alias.name.startswith("comfy")
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("comfy")
                assert node.module not in ("folder_paths", "server")

    def test_the_vendored_package_is_not_imported(self) -> None:
        """``import chatterbox`` at module scope would load the model stack."""
        import ast

        with open(model_info.__file__, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("chatterbox"), alias.name

# ---------------------------------------------------------------------------
# The Turbo family shares one folder (2026-10-06)
# ---------------------------------------------------------------------------


class TestTurboFamilyFolder:
    def test_both_members_download_into_one_folder(self) -> None:
        assert MODEL_CONFIGS["chatterbox-turbo"]["folder"] == "chatterbox-turbo"
        assert MODEL_CONFIGS["chatterbox-nano"]["folder"] == "chatterbox-turbo"
        assert (
            MODEL_CONFIGS["chatterbox-nano"]["repo_id"]
            == "ResembleAI/chatterbox-nano"
        )

    def test_pack_dir_entries_splits_a_family_folder(self, tmp_path) -> None:
        family = tmp_path / "chatterbox-turbo"
        family.mkdir()
        for name in ("t3_turbo_v1.safetensors", "t3_nano_v1.safetensors"):
            (family / name).write_bytes(b"\x00" * 4)

        assert model_info.pack_dir_entries(str(family)) == [
            ("chatterbox-turbo", "chatterbox_turbo"),
            ("chatterbox-nano", "chatterbox_nano"),
        ]

    def test_a_single_variant_folder_offers_itself_once(self, tmp_path) -> None:
        pack = write_pack(tmp_path, "mine", weights=("t3_cfg.safetensors",))

        assert model_info.pack_dir_entries(pack) == [("mine", "chatterbox_tts")]

    def test_pack_dir_entries_answers_nothing_for_a_missing_dir(self, tmp_path) -> None:
        assert model_info.pack_dir_entries(str(tmp_path / "nope")) == []

    def test_register_accepts_an_explicit_architecture(self, tmp_path) -> None:
        pack = write_pack(
            tmp_path,
            "pack",
            weights=("t3_turbo_v1.safetensors", "t3_nano_v1.safetensors"),
        )

        register_local_model("pack-nano", pack, architecture="chatterbox_nano")

        assert (
            AVAILABLE_CHATTERBOX_MODELS["pack-nano"]["architecture"]
            == "chatterbox_nano"
        )
        assert unregister_local_model("pack-nano") is True

    def test_refresh_registers_both_family_members(self, tmp_path, monkeypatch) -> None:
        family = tmp_path / "family"
        family.mkdir()
        for name in ("t3_turbo_v1.safetensors", "t3_nano_v1.safetensors"):
            (family / name).write_bytes(b"\x00" * 4)
        registry = {k: dict(v) for k, v in AVAILABLE_CHATTERBOX_MODELS.items()}
        monkeypatch.setattr(model_info, "AVAILABLE_CHATTERBOX_MODELS", registry)
        import ComfyUI_Chatterbox.modules.model_discovery as discovery

        monkeypatch.setattr(
            discovery, "get_model_pack_entries", lambda: [("family", str(family))]
        )

        local = refresh_local_models()

        assert {entry["name"] for entry in local} == {"family", "family-nano"}
        assert registry["family"]["architecture"] == "chatterbox_turbo"
        assert registry["family-nano"]["architecture"] == "chatterbox_nano"
        assert registry["family-nano"]["path"] == str(family)


# ---------------------------------------------------------------------------
# The upstream pack names (2026-10-23)
# ---------------------------------------------------------------------------


class TestPackNameAliases:
    """A saved workflow carries the name it was saved with, so the old
    spelling has to keep resolving -- but it is never offered in the
    dropdown, and one table is the only place that knows about it."""

    def test_the_aliases_point_at_the_packs_they_replaced(self) -> None:
        """Two renames, two aliases: the Turbo-family member upstream
        renamed, and the classic pack this repository stopped calling by an
        internal id when the dropdown learned the upstream name."""
        assert set(model_info.PACK_ALIASES.values()) == {
            "chatterbox-nano",
            "chatterbox",
        }

    def test_every_alias_points_at_a_real_pack(self) -> None:
        for alias, canonical in model_info.PACK_ALIASES.items():
            assert canonical in MODEL_CONFIGS, (
                f"the alias '{alias}' points at '{canonical}', which is not a pack"
            )
            assert canonical not in model_info.PACK_ALIASES, (
                f"the alias '{alias}' points at '{canonical}', which is itself an alias"
            )

    def test_an_alias_is_never_offered(self) -> None:
        for alias in model_info.PACK_ALIASES:
            assert alias not in model_info.official_pack_names()
            assert alias not in AVAILABLE_CHATTERBOX_MODELS

    def test_canonicalising_returns_the_current_name(self) -> None:
        for alias, canonical in model_info.PACK_ALIASES.items():
            assert model_info.canonical_pack_name(alias) == canonical
            assert model_info.canonical_pack_name(canonical) == canonical

    def test_an_unknown_name_is_its_own_canonical_name(self) -> None:
        assert model_info.canonical_pack_name("my-pack") == "my-pack"
        assert model_info.canonical_pack_name("") == ""


class TestUpstreamFileLists:
    """The registry asks for the files upstream publishes. A file that is not
    there makes the pack report itself as never downloaded."""

    def test_the_turbo_entry_lists_the_files_upstream_publishes(self) -> None:
        files = MODEL_CONFIGS["chatterbox-turbo"]["files"]
        assert "tokenizer.json" not in files, (
            "upstream's Turbo repo ships the GPT2 tokenizer layout "
            "(vocab.json + merges.txt), not a tokenizer.json"
        )
        assert {
            "vocab.json",
            "merges.txt",
            "tokenizer_config.json",
            "added_tokens.json",
            "special_tokens_map.json",
        } <= set(files)

    def test_both_members_agree_on_the_shared_files(self) -> None:
        """The two upstream repos publish the same file list apart from their
        own t3 checkpoint."""
        turbo = set(MODEL_CONFIGS["chatterbox-turbo"]["files"])
        nano = set(MODEL_CONFIGS["chatterbox-nano"]["files"])
        assert turbo - nano == {"t3_turbo_v1.safetensors", "t3_turbo_v1.yaml"}
        assert nano - turbo == {"t3_nano_v1.safetensors", "t3_nano_v1.yaml"}

    def test_no_turbo_family_entry_asks_for_a_tokenizer_json(self) -> None:
        for name in ("chatterbox-turbo", "chatterbox-nano"):
            assert "tokenizer.json" not in MODEL_CONFIGS[name]["files"]


class TestTheOfficialNanoEntrySurvivesTheLocalScan:
    """The shared Turbo folder holds both members, so the scan mints the
    Nano name for it -- but the official Nano entry already knows that folder
    and the repository it comes from, and a local copy would strip both."""

    @staticmethod
    def _registry_copy(monkeypatch) -> dict:
        registry = {k: dict(v) for k, v in AVAILABLE_CHATTERBOX_MODELS.items()}
        monkeypatch.setattr(model_info, "AVAILABLE_CHATTERBOX_MODELS", registry)
        return registry

    @staticmethod
    def _family_folder(tmp_path) -> Path:
        family = tmp_path / "chatterbox-turbo"
        family.mkdir()
        for name in ("t3_turbo_v1.safetensors", "t3_nano_v1.safetensors"):
            (family / name).write_bytes(b"\x00" * 4)
        return family

    def test_the_shared_folder_keeps_the_official_nano_entry(
        self, tmp_path, monkeypatch
    ) -> None:
        import ComfyUI_Chatterbox.modules.model_discovery as discovery

        family = self._family_folder(tmp_path)
        registry = self._registry_copy(monkeypatch)
        monkeypatch.setattr(
            discovery, "get_model_pack_entries", lambda: [("chatterbox-turbo", str(family))]
        )

        refresh_local_models()

        entry = registry["chatterbox-nano"]
        assert entry["type"] == "official"
        assert entry["repo_id"] == "ResembleAI/chatterbox-nano"
        assert entry["folder"] == "chatterbox-turbo"

    def test_the_shared_folder_still_registers_its_own_name(
        self, tmp_path, monkeypatch
    ) -> None:
        import ComfyUI_Chatterbox.modules.model_discovery as discovery

        family = self._family_folder(tmp_path)
        registry = self._registry_copy(monkeypatch)
        monkeypatch.setattr(
            discovery, "get_model_pack_entries", lambda: [("chatterbox-turbo", str(family))]
        )

        refresh_local_models()

        entry = registry["chatterbox-turbo"]
        assert entry["type"] == "local"
        assert entry["path"] == str(family)

    def test_a_directory_of_the_canonical_name_still_registers_locally(
        self, tmp_path, monkeypatch
    ) -> None:
        """A user's own folder is data, not vocabulary: it keeps its name and
        its local entry, exactly as every other directory does."""
        import ComfyUI_Chatterbox.modules.model_discovery as discovery

        pack = tmp_path / "chatterbox-nano"
        pack.mkdir()
        (pack / "t3_nano_v1.safetensors").write_bytes(b"\x00" * 4)
        registry = self._registry_copy(monkeypatch)
        monkeypatch.setattr(
            discovery, "get_model_pack_entries", lambda: [("chatterbox-nano", str(pack))]
        )

        refresh_local_models()

        entry = registry["chatterbox-nano"]
        assert entry["type"] == "local"
        assert entry["path"] == str(pack)
