"""S4.7 -- ``modules/model_discovery.py``.

This module is a cache. The node schemas rebuild their pack combo on every
frontend refresh, and the naive implementation ``listdir``s the models root
each time. So the behaviour under test is mostly about *not* touching the disk:
two calls against an unchanged tree must cost exactly one ``os.scandir``, and
touching the tree must cost a second one.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ComfyUI_Chatterbox.modules import model_discovery
from ComfyUI_Chatterbox.modules.constants import (
    CHATTERBOX_MODEL_SUBDIR,
    DEFAULT_MODEL_PACK_NAME,
)
from ComfyUI_Chatterbox.modules.model_discovery import (
    get_chatterbox_model_pack_names,
    get_model_pack_entries,
    get_model_pack_path,
    get_models_root,
    invalidate_cache,
)


#: Every official pack the dropdown must offer, in the registry's order.
#:
#: Written out rather than read back from the registry on purpose: a test
#: that derives its expectation from the code under test proves nothing. A
#: pack dropped from the table must fail here, not quietly follow the drop.
OFFICIAL_PACK_NAMES = (
    "chatterbox",
    "chatterbox-multilingual-zh-cmn",
    "chatterbox-multilingual-es-mx-latam",
    "chatterbox-multilingual-es-es",
    "chatterbox-multilingual-pt-br",
    "chatterbox-multilingual-pt-pt",
    "chatterbox-multilingual-hi",
    "chatterbox-turbo",
    "chatterbox-nano",
    "chatterbox-flash",
)

#: The official names other than the default, i.e. the ones appended after
#: the local packs in a merged list.
ADDITIONAL_OFFICIAL_PACK_NAMES = OFFICIAL_PACK_NAMES[1:]


@pytest.fixture(autouse=True)
def clean_cache():
    invalidate_cache()
    yield
    invalidate_cache()


@pytest.fixture()
def models_root(tmp_path, monkeypatch):
    """A models root at ``<tmp>/models/tts/chatterbox``, wired into folder_paths."""
    import folder_paths

    root = tmp_path / "models" / "tts" / "chatterbox"
    root.mkdir(parents=True)
    monkeypatch.setattr(
        folder_paths, "get_folder_paths", lambda name: [str(tmp_path / "models" / "tts")]
    )
    return root


def touch_root(root: Path) -> None:
    """Change the root's mtime the way adding a pack would.

    A real write bumps ``st_mtime_ns`` by well over a millisecond; sleeping
    keeps this honest on filesystems with coarse timestamps.
    """
    import time

    time.sleep(0.01)
    (root / "touched.txt").write_text("x", encoding="utf-8")


class TestModelsRoot:
    def test_resolves_under_the_tts_folder(self, models_root) -> None:
        assert get_models_root() == str(models_root)

    def test_falls_back_to_models_dir(self, tmp_path, monkeypatch) -> None:
        import folder_paths

        def unregistered(name):
            raise KeyError(name)

        monkeypatch.setattr(folder_paths, "get_folder_paths", unregistered)
        monkeypatch.setattr(folder_paths, "models_dir", str(tmp_path / "models"))
        assert get_models_root() == os.path.join(str(tmp_path / "models"), CHATTERBOX_MODEL_SUBDIR)

    def test_falls_back_to_a_relative_path_without_comfy(self, monkeypatch) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "folder_paths", None)
        assert get_models_root() == os.path.join("models", CHATTERBOX_MODEL_SUBDIR)


class TestPackNames:
    def test_lists_every_subdirectory(self, models_root) -> None:
        (models_root / "alpha").mkdir()
        (models_root / "beta").mkdir()
        assert get_chatterbox_model_pack_names() == [
            DEFAULT_MODEL_PACK_NAME,
            "alpha",
            "beta",
            *ADDITIONAL_OFFICIAL_PACK_NAMES,
        ]

    def test_ignores_loose_files(self, models_root) -> None:
        (models_root / "alpha").mkdir()
        (models_root / "stray.txt").write_text("x", encoding="utf-8")
        assert get_chatterbox_model_pack_names() == [
            DEFAULT_MODEL_PACK_NAME,
            "alpha",
            *ADDITIONAL_OFFICIAL_PACK_NAMES,
        ]

    def test_a_partially_downloaded_pack_is_still_offered(self, models_root) -> None:
        """The pack must stay selectable, or the download can never finish it."""
        half = models_root / "half"
        half.mkdir()
        (half / "ve.safetensors.incomplete").write_bytes(b"x")
        assert get_chatterbox_model_pack_names() == [
            DEFAULT_MODEL_PACK_NAME,
            "half",
            *ADDITIONAL_OFFICIAL_PACK_NAMES,
        ]

    def test_the_default_pack_is_hoisted_to_index_zero(self, models_root) -> None:
        for name in ("aaa", "zzz", DEFAULT_MODEL_PACK_NAME, "mmm"):
            (models_root / name).mkdir()
        names = get_chatterbox_model_pack_names()
        assert names[0] == DEFAULT_MODEL_PACK_NAME
        assert names[1:4] == ["aaa", "mmm", "zzz"]
        assert sorted(names) == sorted(
            ["aaa", "zzz", "mmm", *OFFICIAL_PACK_NAMES]
        )

    def test_an_empty_root_still_offers_the_default(self, models_root) -> None:
        """An empty root offers every official pack, the default first."""
        assert get_chatterbox_model_pack_names() == list(OFFICIAL_PACK_NAMES)

    def test_a_missing_root_still_offers_the_default(self, tmp_path, monkeypatch) -> None:
        import folder_paths

        monkeypatch.setattr(folder_paths, "get_folder_paths", lambda name: [str(tmp_path / "gone")])
        assert get_chatterbox_model_pack_names() == list(OFFICIAL_PACK_NAMES)

    def test_the_names_are_the_frozen_default(self) -> None:
        assert DEFAULT_MODEL_PACK_NAME == "chatterbox"


# ---------------------------------------------------------------------------
# The dropdown offers what can be downloaded, not only what is on disk
# ---------------------------------------------------------------------------


class TestDownloadablePacksAreOffered:
    """A fresh install has no models folder, and a dropdown built from the
    filesystem alone then offers nothing to pick -- and a pack that cannot be
    picked cannot be downloaded. The registry knows every official pack, so
    the names list must carry them whether or not the folder exists yet."""

    def test_an_empty_root_offers_every_official_pack(self, models_root) -> None:
        names = get_chatterbox_model_pack_names()
        for name in OFFICIAL_PACK_NAMES:
            assert name in names, f"the downloadable pack '{name}' is not offered"

    def test_the_official_packs_are_offered_in_registry_order(self, models_root) -> None:
        """The order is the registry's, so the list reads like the table."""
        assert get_chatterbox_model_pack_names() == list(OFFICIAL_PACK_NAMES)

    def test_a_missing_root_offers_the_official_packs_too(
        self, tmp_path, monkeypatch
    ) -> None:
        import folder_paths

        monkeypatch.setattr(
            folder_paths, "get_folder_paths", lambda name: [str(tmp_path / "gone")]
        )
        names = get_chatterbox_model_pack_names()
        assert all(name in names for name in OFFICIAL_PACK_NAMES)

    def test_local_packs_and_official_packs_are_merged(self, models_root) -> None:
        """A community pack on disk and the official ones that are not."""
        (models_root / "my-voice").mkdir()
        names = get_chatterbox_model_pack_names()

        assert "my-voice" in names
        for name in OFFICIAL_PACK_NAMES:
            assert name in names
        assert names.index("my-voice") < names.index("chatterbox-flash")

    def test_the_merge_holds_no_duplicates(self, models_root) -> None:
        (models_root / "my-voice").mkdir()
        names = get_chatterbox_model_pack_names()

        assert len(names) == len(set(names))

    def test_a_local_directory_named_after_an_official_pack_is_not_duplicated(
        self, models_root
    ) -> None:
        """The same pack, downloaded and unpacked in place, is one entry."""
        (models_root / "chatterbox-flash").mkdir()
        names = get_chatterbox_model_pack_names()

        assert names.count("chatterbox-flash") == 1
        assert len(names) == len(set(names))

    def test_both_turbo_family_names_are_offered(self, models_root) -> None:
        """Turbo and Nano are two dropdown entries over one folder."""
        names = get_chatterbox_model_pack_names()

        assert "chatterbox-turbo" in names
        assert "chatterbox-nano" in names

    def test_no_alias_is_offered(self, models_root) -> None:
        """An old spelling keeps resolving, but it is never put in front of
        the user: the dropdown lists the current name only."""
        from ComfyUI_Chatterbox.modules.model_info import PACK_ALIASES

        names = get_chatterbox_model_pack_names()

        for alias in PACK_ALIASES:
            assert alias not in names, (
                f"the alias '{alias}' is offered in the dropdown; it is a "
                "resolution-time name, not a pack of its own"
            )

    def test_the_t3_checkpoints_are_not_offered(self, models_root) -> None:
        """``T3_v1``/``T3_v2``/``T3_v3`` are checkpoints inside the classic
        repository, not packs of their own: there is no directory to download
        them into, and the multilingual loader picks one out of a pack."""
        names = get_chatterbox_model_pack_names()

        for name in ("T3_v1", "T3_v2", "T3_v3"):
            assert name not in names

    def test_the_list_is_never_empty(self, tmp_path, monkeypatch) -> None:
        import folder_paths

        monkeypatch.setattr(
            folder_paths, "get_folder_paths", lambda name: [str(tmp_path / "gone")]
        )
        assert get_chatterbox_model_pack_names()

    def test_the_entries_are_untouched_by_the_merge(self, models_root) -> None:
        """``get_model_pack_entries`` is the *path* contract: the registry
        registers local packs from it. An official name with no directory
        behind it must never leak in there."""
        (models_root / "my-voice").mkdir()
        entries = get_model_pack_entries()

        assert entries == [("my-voice", str(models_root / "my-voice"))]
        assert not {name for name, _path in entries} & set(OFFICIAL_PACK_NAMES)


class TestPackEntries:
    def test_entries_pair_names_with_paths(self, models_root) -> None:
        (models_root / "alpha").mkdir()
        entries = get_model_pack_entries()
        assert entries == [("alpha", str(models_root / "alpha"))]

    def test_the_returned_list_is_a_copy(self, models_root) -> None:
        (models_root / "alpha").mkdir()
        first = get_model_pack_entries()
        first.append(("bogus", "/nowhere"))
        assert get_model_pack_entries() == [("alpha", str(models_root / "alpha"))]

    def test_a_missing_root_yields_nothing(self, tmp_path, monkeypatch) -> None:
        import folder_paths

        monkeypatch.setattr(folder_paths, "get_folder_paths", lambda name: [str(tmp_path / "gone")])
        assert get_model_pack_entries() == []


class TestPackPath:
    def test_resolves_a_name(self, models_root) -> None:
        assert get_model_pack_path("alpha") == str(models_root / "alpha")

    @pytest.mark.parametrize("blank", ["", None])
    def test_a_blank_name_returns_none(self, blank) -> None:
        assert get_model_pack_path(blank) is None

    def test_a_blank_name_does_not_resolve_to_the_models_root(self, models_root) -> None:
        """A saved workflow can carry an empty combo value; resolving it to the
        models root would load the wrong directory."""
        assert get_model_pack_path("") != str(models_root)


class TestCaching:
    """The whole point of the module."""

    @staticmethod
    def _count_scandir(monkeypatch) -> list[int]:
        calls = []
        real_scandir = os.scandir

        def counting_scandir(path=".", *args, **kwargs):
            calls.append(1)
            return real_scandir(path, *args, **kwargs)

        monkeypatch.setattr(model_discovery.os, "scandir", counting_scandir)
        return calls

    def test_two_calls_on_an_unchanged_tree_scan_once(self, models_root, monkeypatch) -> None:
        (models_root / "alpha").mkdir()
        (models_root / "beta").mkdir()
        calls = self._count_scandir(monkeypatch)

        first = get_chatterbox_model_pack_names()
        second = get_chatterbox_model_pack_names()

        assert first == second == [
            DEFAULT_MODEL_PACK_NAME,
            "alpha",
            "beta",
            *ADDITIONAL_OFFICIAL_PACK_NAMES,
        ]
        assert len(calls) == 1, (
            f"the models root was scanned {len(calls)} times for an unchanged tree; "
            "the cache is not working"
        )

    def test_touching_the_root_invalidates_the_cache(self, models_root, monkeypatch) -> None:
        (models_root / "alpha").mkdir()
        calls = self._count_scandir(monkeypatch)

        assert get_chatterbox_model_pack_names() == [
            DEFAULT_MODEL_PACK_NAME,
            "alpha",
            *ADDITIONAL_OFFICIAL_PACK_NAMES,
        ]
        assert len(calls) == 1

        (models_root / "beta").mkdir()
        touch_root(models_root)

        assert get_chatterbox_model_pack_names() == [
            DEFAULT_MODEL_PACK_NAME,
            "alpha",
            "beta",
            *ADDITIONAL_OFFICIAL_PACK_NAMES,
        ]
        assert len(calls) == 2

    def test_the_cache_key_is_the_root_mtime(self, models_root) -> None:
        first = model_discovery._root_key(str(models_root))
        touch_root(models_root)
        assert model_discovery._root_key(str(models_root)) != first

    def test_a_missing_root_is_cached_too(self, tmp_path, monkeypatch) -> None:
        import folder_paths

        monkeypatch.setattr(folder_paths, "get_folder_paths", lambda name: [str(tmp_path / "gone")])
        calls = self._count_scandir(monkeypatch)

        assert get_chatterbox_model_pack_names() == list(OFFICIAL_PACK_NAMES)
        assert get_chatterbox_model_pack_names() == list(OFFICIAL_PACK_NAMES)
        assert len(calls) == 1, (
            "a models root that does not exist must be cached too, otherwise "
            "every schema rebuild pays a failed directory open"
        )

    def test_invalidate_cache_forces_a_rescan(self, models_root, monkeypatch) -> None:
        (models_root / "alpha").mkdir()
        calls = self._count_scandir(monkeypatch)

        get_chatterbox_model_pack_names()
        invalidate_cache()
        get_chatterbox_model_pack_names()

        assert len(calls) == 2

    def test_changing_the_root_resolves_different_trees(self, tmp_path, monkeypatch) -> None:
        import folder_paths

        first_root = tmp_path / "one" / "chatterbox"
        second_root = tmp_path / "two" / "chatterbox"
        first_root.mkdir(parents=True)
        second_root.mkdir(parents=True)
        (first_root / "from_one").mkdir()
        (second_root / "from_two").mkdir()

        roots = {"value": str(tmp_path / "one")}
        monkeypatch.setattr(folder_paths, "get_folder_paths", lambda name: [roots["value"]])
        assert get_chatterbox_model_pack_names() == [
            DEFAULT_MODEL_PACK_NAME,
            "from_one",
            *ADDITIONAL_OFFICIAL_PACK_NAMES,
        ]

        roots["value"] = str(tmp_path / "two")
        assert get_chatterbox_model_pack_names() == [
            DEFAULT_MODEL_PACK_NAME,
            "from_two",
            *ADDITIONAL_OFFICIAL_PACK_NAMES,
        ]


class TestFrozenConstants:
    """``modules/constants.py`` against the recorded snapshot.

    ``tests/test_move_parity.py`` checks the snapshot against a hardcoded copy
    of the values. This checks the snapshot against the module that actually
    ships, so the two cannot drift apart unnoticed.
    """

    @pytest.fixture(scope="class")
    def recorded(self) -> dict:
        from conftest import ROOT_DIR

        import json

        with open(
            os.path.join(ROOT_DIR, "tests", "frozen_widget_snapshot.json"),
            "r",
            encoding="utf-8",
        ) as handle:
            return json.load(handle)["constants"]

    def test_the_module_subdir_matches_the_snapshot(self, recorded: dict) -> None:
        assert CHATTERBOX_MODEL_SUBDIR == os.path.join(*recorded["CHATTERBOX_MODEL_SUBDIR"]["parts"])
        assert recorded["CHATTERBOX_MODEL_SUBDIR"]["value"] == "tts/chatterbox"

    def test_the_repo_id_matches_the_snapshot(self, recorded: dict) -> None:
        from ComfyUI_Chatterbox.modules.constants import CHATTERBOX_REPO_ID

        assert CHATTERBOX_REPO_ID == recorded["CHATTERBOX_REPO_ID"]["value"]
        assert CHATTERBOX_REPO_ID == "".join(recorded["CHATTERBOX_REPO_ID"]["parts"])

    def test_the_file_list_matches_the_snapshot(self, recorded: dict) -> None:
        from ComfyUI_Chatterbox.modules.constants import CHATTERBOX_FILES_TO_DOWNLOAD

        assert CHATTERBOX_FILES_TO_DOWNLOAD == recorded["CHATTERBOX_FILES_TO_DOWNLOAD"]["value"]
        assert CHATTERBOX_FILES_TO_DOWNLOAD == list(
            recorded["CHATTERBOX_FILES_TO_DOWNLOAD"]["parts"]
        )

    def test_the_default_pack_name_matches_the_snapshot(self, recorded: dict) -> None:
        assert DEFAULT_MODEL_PACK_NAME == recorded["DEFAULT_MODEL_PACK_NAME"]["value"]
        assert DEFAULT_MODEL_PACK_NAME == "".join(recorded["DEFAULT_MODEL_PACK_NAME"]["parts"])


class TestModuleHygiene:
    def test_no_comfy_api_import(self) -> None:
        with open(model_discovery.__file__, "r", encoding="utf-8") as handle:
            assert "import comfy_api" not in handle.read()

    def test_the_scan_uses_scandir_not_listdir(self) -> None:
        """``os.listdir`` would defeat the point of counting directory reads,
        and cannot distinguish a file from a subdirectory in one pass."""
        import ast

        with open(model_discovery.__file__, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "scandir" in called
        assert "listdir" not in called

# ---------------------------------------------------------------------------
# Turbo-family folder: one directory, two dropdown entries
# ---------------------------------------------------------------------------


class TestTurboFamilyFolderIsOfferedTwice:
    def test_a_folder_with_both_t3_checkpoints_is_offered_twice(self, models_root) -> None:
        family = models_root / "chatterbox-turbo"
        family.mkdir()
        for name in ("t3_turbo_v1.safetensors", "t3_nano_v1.safetensors", "ve.safetensors"):
            (family / name).write_bytes(b"\x00" * 4)

        entries = get_model_pack_entries()

        assert [name for name, _path in entries] == [
            "chatterbox-turbo",
            "chatterbox-nano",
        ]
        assert entries[0][1] == entries[1][1] == str(family)

    def test_the_synthesized_name_yields_to_a_real_directory(self, models_root) -> None:
        """A directory that owns the canonical name owns it: the name the
        scan mints must not take it over."""
        family = models_root / "chatterbox-turbo"
        family.mkdir()
        for name in ("t3_turbo_v1.safetensors", "t3_nano_v1.safetensors"):
            (family / name).write_bytes(b"\x00" * 4)
        real = models_root / "chatterbox-nano"
        real.mkdir()

        paths = dict(get_model_pack_entries())

        assert paths["chatterbox-turbo"] == str(family)
        assert paths["chatterbox-nano"] == str(real)

    def test_a_single_variant_folder_is_offered_once(self, models_root) -> None:
        pack = models_root / "chatterbox-turbo"
        pack.mkdir()
        (pack / "t3_turbo_v1.safetensors").write_bytes(b"\x00" * 4)

        assert [name for name, _path in get_model_pack_entries()] == ["chatterbox-turbo"]

    def test_the_shared_folder_is_offered_under_the_canonical_name(
        self, models_root
    ) -> None:
        """One folder, two entries -- and the Nano one carries the name
        upstream gives it."""
        family = models_root / "chatterbox-turbo"
        family.mkdir()
        for name in ("t3_turbo_v1.safetensors", "t3_nano_v1.safetensors"):
            (family / name).write_bytes(b"\x00" * 4)

        names = get_chatterbox_model_pack_names()

        assert names.count("chatterbox-nano") == 1
        assert len(names) == len(set(names))
        assert "chatterbox-turbo" in names

    def test_resolve_pack_dir_honours_the_entry_folder(self, models_root, monkeypatch) -> None:
        """The Nano member downloads into the shared Turbo folder."""
        from ComfyUI_Chatterbox.modules import model_info
        from ComfyUI_Chatterbox.modules.loader import ChatterboxLoader

        registry = dict(model_info.AVAILABLE_CHATTERBOX_MODELS)
        registry["chatterbox-nano"] = {
            "type": "official",
            "repo_id": "ResembleAI/chatterbox-nano",
            "folder": "chatterbox-turbo",
        }
        monkeypatch.setattr(
            "ComfyUI_Chatterbox.modules.loader.AVAILABLE_CHATTERBOX_MODELS", registry
        )

        resolved = ChatterboxLoader.resolve_pack_dir("chatterbox-nano")

        assert resolved == str(models_root / "chatterbox-turbo")


class TestTheOfficialNanoEntrySurvivesTheSharedFolder:
    """The shared folder holds both members, so the scan mints the Nano name
    for it -- but the official Nano entry already knows that folder and the
    repository it comes from, and a local copy would strip both."""

    def test_the_scan_keeps_the_official_nano_entry(self, models_root, monkeypatch) -> None:
        from ComfyUI_Chatterbox.modules import model_info

        family = models_root / "chatterbox-turbo"
        family.mkdir()
        for name in ("t3_turbo_v1.safetensors", "t3_nano_v1.safetensors"):
            (family / name).write_bytes(b"\x00" * 4)

        registry = {k: dict(v) for k, v in model_info.AVAILABLE_CHATTERBOX_MODELS.items()}
        monkeypatch.setattr(model_info, "AVAILABLE_CHATTERBOX_MODELS", registry)

        model_info.refresh_local_models()

        entry = registry["chatterbox-nano"]
        assert entry["type"] == "official"
        assert entry["repo_id"] == "ResembleAI/chatterbox-nano"
        assert entry["folder"] == "chatterbox-turbo"


# ---------------------------------------------------------------------------
# An offered pack that is not on disk yet resolves, and downloads
# ---------------------------------------------------------------------------

class TestAnAbsentOfficialPackResolves:
    """The offered name has to mean something: it must resolve to the
    directory the pack will be downloaded into, and the registry entry has to
    name the repository the download comes from."""

    @pytest.mark.parametrize(
        "name,folder",
        [
            ("chatterbox", "chatterbox"),
            ("chatterbox-multilingual-zh-cmn", "chatterbox-multilingual-zh-cmn"),
            ("chatterbox-multilingual-es-mx-latam", "chatterbox-multilingual-es-mx-latam"),
            ("chatterbox-multilingual-es-es", "chatterbox-multilingual-es-es"),
            ("chatterbox-multilingual-pt-br", "chatterbox-multilingual-pt-br"),
            ("chatterbox-multilingual-pt-pt", "chatterbox-multilingual-pt-pt"),
            ("chatterbox-multilingual-hi", "chatterbox-multilingual-hi"),
            ("chatterbox-turbo", "chatterbox-turbo"),
            # Nano is a dropdown entry over Turbo's folder.
            ("chatterbox-nano", "chatterbox-turbo"),
            ("chatterbox-flash", "chatterbox-flash"),
        ],
    )
    def test_it_resolves_to_its_download_directory(
        self, models_root, name: str, folder: str
    ) -> None:
        from ComfyUI_Chatterbox.modules.loader import ChatterboxLoader

        resolved = ChatterboxLoader.resolve_pack_dir(name)

        assert resolved == str(models_root / folder)
        assert not os.path.exists(resolved), (
            "the pack was supposed to be absent -- the test would prove "
            "nothing if it were already on disk"
        )

    @pytest.mark.parametrize("name", OFFICIAL_PACK_NAMES)
    def test_its_registry_entry_names_a_repository(self, name: str) -> None:
        """Without a ``repo_id`` the download cannot run at all."""
        from ComfyUI_Chatterbox.modules.model_info import MODEL_CONFIGS

        assert MODEL_CONFIGS[name].get("repo_id"), (
            f"the offered pack '{name}' has no repo_id, so it can never be "
            "downloaded"
        )


class TestAnAbsentPackIsDownloaded:
    """Picking a pack that is not on disk yet must download it -- from the
    registry's repository and file list -- instead of failing with a
    'missing files' error."""

    @staticmethod
    def _fake_downloader(downloads: list):
        def fake_download(model_name, repo_id, local_dir, client_id=None, allow_patterns=None):
            downloads.append(
                {
                    "model_name": model_name,
                    "repo_id": repo_id,
                    "local_dir": local_dir,
                    "allow_patterns": list(allow_patterns or []),
                }
            )
            os.makedirs(local_dir, exist_ok=True)
            for name in allow_patterns or []:
                with open(os.path.join(local_dir, name), "wb") as handle:
                    handle.write(b"\x00" * 4)

        return fake_download

    @staticmethod
    def _fake_shared_fetch(repo_id, filename, local_dir):
        os.makedirs(local_dir, exist_ok=True)
        path = os.path.join(local_dir, filename)
        with open(path, "wb") as handle:
            handle.write(b"\x00" * 4)
        return path

    def test_an_absent_slp_pack_is_downloaded(self, models_root, monkeypatch) -> None:
        """The multilingual node's Single Language Packs are offered whether
        or not they are on disk; selecting one has to fetch it."""
        import torch

        from ComfyUI_Chatterbox.modules.loader import ChatterboxLoader

        downloads: list = []
        monkeypatch.setattr(
            ChatterboxLoader,
            "download_pack",
            staticmethod(self._fake_downloader(downloads)),
        )
        monkeypatch.setattr(
            "huggingface_hub.hf_hub_download", self._fake_shared_fetch
        )

        pack_dir = ChatterboxLoader.resolve_pack_dir("chatterbox-multilingual-zh-cmn")
        model = ChatterboxLoader.load_multilingual_model(
            "chatterbox-multilingual-zh-cmn",
            "t3_mtl23ls_v3.safetensors",
            torch.device("cpu"),
            model_path=pack_dir,
        )

        assert downloads, "the absent pack was never downloaded"
        call = downloads[0]
        assert call["repo_id"] == "ResembleAI/Chatterbox-Multilingual-zh-cmn"
        assert call["local_dir"] == pack_dir
        assert "t3_zh_cmn.safetensors" in call["allow_patterns"], (
            "the pack's own T3 checkpoint must be part of the fetch"
        )
        assert model is not None

    def test_the_multilingual_checkpoint_is_fetched_for_the_classic_pack(
        self, models_root, monkeypatch
    ) -> None:
        """The classic repository also holds the multilingual T3 weights, but
        the classic entry's file list names the monolingual set. Without the
        resolved checkpoint in the fetch the pack would download and still be
        missing it -- a 'missing files' error the user cannot act on."""
        import torch

        from ComfyUI_Chatterbox.modules.constants import CHATTERBOX_REPO_ID
        from ComfyUI_Chatterbox.modules.loader import ChatterboxLoader

        downloads: list = []
        monkeypatch.setattr(
            ChatterboxLoader,
            "download_pack",
            staticmethod(self._fake_downloader(downloads)),
        )
        monkeypatch.setattr(
            "huggingface_hub.hf_hub_download", self._fake_shared_fetch
        )

        ChatterboxLoader.load_multilingual_model(
            DEFAULT_MODEL_PACK_NAME,
            "t3_mtl23ls_v3.safetensors",
            torch.device("cpu"),
            model_path=str(models_root / DEFAULT_MODEL_PACK_NAME),
        )

        assert downloads, "the absent pack was never downloaded"
        call = downloads[0]
        assert call["repo_id"] == CHATTERBOX_REPO_ID
        assert "t3_mtl23ls_v3.safetensors" in call["allow_patterns"], (
            "the multilingual checkpoint the loader asked for was not fetched"
        )
