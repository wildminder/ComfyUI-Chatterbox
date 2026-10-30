"""S4.4 -- ``modules/api_routes.py``.

Seven routes, registered on a fake aiohttp router and invoked as coroutines.
The point of every test here is that the handlers answer *something a frontend
can act on*: the registry lists the official entries, a cancellation that has
nothing to cancel says so instead of failing, a directory missing weights is
rejected with a message naming what is missing, and clearing the settings both
reloads :class:`ChatterboxSettings` and pushes the result to the clients.

No test touches the network and none of them reads the real models tree: the
injected ``folder_paths`` stand-in and ``tmp_path`` are the only filesystem.
"""

from __future__ import annotations

import ast
import asyncio
import copy
import json
import os

import pytest

from ComfyUI_Chatterbox.modules import (
    api_routes,
    config_broadcast,
    downloader,
    model_discovery,
    model_info,
)
from ComfyUI_Chatterbox.modules import settings as settings_module
from ComfyUI_Chatterbox.modules.api_routes import (
    API_PREFIX,
    DOWNLOAD_EVENT_CHANNEL,
    DOWNLOAD_EVENT_TYPES,
    register_routes,
)
from ComfyUI_Chatterbox.modules.constants import DEFAULT_MODEL_PACK_NAME
from ComfyUI_Chatterbox.modules.folder_registration import TTS_FOLDER_KEY
from ComfyUI_Chatterbox.modules.model_info import MODEL_CONFIGS

# ── Doubles ──────────────────────────────────────────────────────────

EXPECTED_ROUTES: tuple[tuple[str, str], ...] = (
    ("GET", f"{API_PREFIX}/models"),
    ("POST", f"{API_PREFIX}/cancel_download"),
    ("GET", f"{API_PREFIX}/download_status"),
    ("GET", f"{API_PREFIX}/tts_search_paths"),
    ("POST", f"{API_PREFIX}/validate_directory"),
    ("POST", f"{API_PREFIX}/register_model_path"),
    ("DELETE", f"{API_PREFIX}/settings"),
)


class FakeRouter:
    """Records ``(method, path, handler)`` the way an aiohttp router would."""

    def __init__(self) -> None:
        self.registered: list[tuple[str, str, object]] = []

    def _decorator(self, method: str, path: str):
        def decorate(handler):
            self.registered.append((method, path, handler))
            return handler

        return decorate

    def get(self, path: str):
        return self._decorator("GET", path)

    def post(self, path: str):
        return self._decorator("POST", path)

    def delete(self, path: str):
        return self._decorator("DELETE", path)

    @property
    def pairs(self) -> list[tuple[str, str]]:
        return [(method, path) for method, path, _handler in self.registered]

    def handler(self, method: str, path: str):
        """The handler registered for exactly this method+path pair."""
        for registered_method, registered_path, handler in self.registered:
            if registered_method == method and registered_path == path:
                return handler
        raise AssertionError(
            f"no {method} route registered for {path!r}; got {self.pairs}"
        )


class FakeResponse:
    """What a fake ``web.json_response`` hands back to a handler."""

    def __init__(self, payload, status: int = 200) -> None:
        self.payload = payload
        self.status = status
        self.text = json.dumps(payload)

    def json(self):
        return self.payload


class FakeWeb:
    """Stands in for ``aiohttp.web`` so no response object is fabricated."""

    def json_response(self, payload, status: int = 200) -> FakeResponse:  # noqa: N802
        return FakeResponse(payload, status)


class FakeRequest:
    """The two things a handler touches: a JSON body and a query string."""

    def __init__(self, body=None, query=None) -> None:
        self._body = body if body is not None else {}
        self.query = dict(query or {})

    async def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class FakeFolderPaths:
    """``folder_paths`` stand-in with the four attributes the routes use."""

    def __init__(self, models_dir, tts_roots=(), user_directory=None) -> None:
        self.models_dir = str(models_dir)
        self._paths: dict[str, list[str]] = {TTS_FOLDER_KEY: list(tts_roots)}
        self._user_directory = str(user_directory) if user_directory else None
        self.added: list[tuple[str, str]] = []

    def get_folder_paths(self, key: str) -> list[str]:
        if key not in self._paths:
            raise ValueError(f"Folder {key!r} is not registered with ComfyUI")
        return list(self._paths[key])

    def add_model_folder_path(self, key: str, path: str) -> None:
        self.added.append((key, path))
        self._paths.setdefault(key, []).append(path)

    def get_user_directory(self):
        return self._user_directory


class FakeDownloadManager:
    def __init__(self, cancel_result: bool, statuses: dict | None = None) -> None:
        self._cancel_result = cancel_result
        self._statuses = statuses if statuses is not None else {}
        self.cancelled: list[str] = []

    def cancel_download(self, model_name: str) -> bool:
        self.cancelled.append(model_name)
        return self._cancel_result

    def get_all_statuses(self) -> dict:
        return dict(self._statuses)


def run(coro):
    """Invoke a handler as a coroutine (``pytest-asyncio`` is not installed)."""
    return asyncio.run(coro)


def make_pack(parent, name: str, weights=("t3_cfg.safetensors",)):
    """Create a minimal but valid Chatterbox pack directory."""
    pack = parent / name
    pack.mkdir(parents=True, exist_ok=True)
    (pack / "tokenizer.json").write_text("{}", encoding="utf-8")
    for weight in weights:
        (pack / weight).write_bytes(b"")
    return pack


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def hermetic(tmp_path, monkeypatch):
    """Keep the registry, the settings singleton and the models root contained."""
    saved_registry = copy.deepcopy(model_info.AVAILABLE_CHATTERBOX_MODELS)
    # A models root that does not exist: `refresh_local_models()` during route
    # registration must not read the developer's real install.
    monkeypatch.setattr(
        model_discovery, "get_models_root", lambda: str(tmp_path / "absent-models-root")
    )
    monkeypatch.setattr(
        model_discovery,
        "get_chatterbox_model_pack_names",
        lambda: [DEFAULT_MODEL_PACK_NAME],
    )
    settings_module._settings_instance = None
    yield
    model_info.AVAILABLE_CHATTERBOX_MODELS.clear()
    model_info.AVAILABLE_CHATTERBOX_MODELS.update(saved_registry)
    settings_module._settings_instance = None


@pytest.fixture
def fake_web(monkeypatch) -> FakeWeb:
    web = FakeWeb()
    monkeypatch.setattr(api_routes, "_resolve_web", lambda: web)
    return web


@pytest.fixture
def routes(fake_web, tmp_path) -> FakeRouter:
    """A router with all seven routes registered against a stock folder_paths."""
    router = FakeRouter()
    register_routes(router, FakeFolderPaths(tmp_path / "models"), str(tmp_path))
    return router


@pytest.fixture
def broadcasts(monkeypatch) -> list:
    """Records every ``send_config_to_client()`` call made by a handler."""
    sent: list = []

    def spy(client_id=None):
        sent.append(client_id)

    monkeypatch.setattr(config_broadcast, "send_config_to_client", spy)
    return sent


# ── Route table ──────────────────────────────────────────────────────

class TestRouteRegistration:
    def test_exactly_the_seven_documented_routes_are_registered(self, routes) -> None:
        assert set(routes.pairs) == set(EXPECTED_ROUTES)
        assert len(routes.registered) == 7

    @pytest.mark.parametrize("method,path", EXPECTED_ROUTES)
    def test_each_route_is_registered_on_its_own_method_and_path(
        self, routes, method, path
    ) -> None:
        handler = routes.handler(method, path)
        assert asyncio.iscoroutinefunction(handler), (
            f"{method} {path} did not register an async handler"
        )
        assert handler.__name__.startswith("chatterbox_")

    def test_no_voxcpm_route_survives_the_port(self, routes) -> None:
        assert not [pair for pair in routes.pairs if "voxcpm" in pair[1]]

    def test_registration_primes_the_registry_once(self, monkeypatch, fake_web, tmp_path) -> None:
        """B5's handoff: local packs must exist before the first /models poll."""
        calls: list[str] = []
        monkeypatch.setattr(
            model_info, "refresh_feature_flags", lambda: calls.append("flags")
        )
        monkeypatch.setattr(
            model_info, "refresh_local_models", lambda: calls.append("local")
        )

        register_routes(FakeRouter(), FakeFolderPaths(tmp_path / "models"), str(tmp_path))

        assert calls == ["flags", "local"], (
            "route registration must refresh the feature flags and the local "
            "model registry exactly once"
        )

    def test_a_failing_local_scan_does_not_break_registration(
        self, monkeypatch, fake_web, tmp_path
    ) -> None:
        def boom():
            raise OSError("models root is a file")

        monkeypatch.setattr(model_info, "refresh_local_models", boom)
        router = FakeRouter()
        register_routes(router, FakeFolderPaths(tmp_path / "models"), str(tmp_path))
        assert set(router.pairs) == set(EXPECTED_ROUTES)


# ── GET /chatterbox/models ───────────────────────────────────────────

class TestModelsRoute:
    def test_lists_every_official_entry_from_model_configs(self, routes) -> None:
        response = run(routes.handler("GET", f"{API_PREFIX}/models")(FakeRequest()))

        assert response.status == 200
        names = {entry["name"] for entry in response.payload["models"]}
        assert names == set(MODEL_CONFIGS)
        assert response.payload["model_count"] == len(MODEL_CONFIGS)

        default_voice = next(
            entry
            for entry in response.payload["models"]
            if entry["name"] == "chatterbox"
        )
        assert default_voice["type"] == "official"
        assert default_voice["repo_id"] == MODEL_CONFIGS["chatterbox"]["repo_id"]
        assert default_voice["architecture"] == "chatterbox_tts"
        assert default_voice["sample_rate"] == 24000
        assert default_voice["is_downloaded"] is False

    def test_reflects_the_feature_flags(self, routes, monkeypatch) -> None:
        monkeypatch.setattr(model_info, "TEXT_NORMALIZATION_AVAILABLE", True)
        monkeypatch.setattr(model_info, "TURBO_AVAILABLE", False)

        response = run(routes.handler("GET", f"{API_PREFIX}/models")(FakeRequest()))

        assert response.payload["normalization_available"] is True
        assert response.payload["turbo_available"] is False

    def test_embeds_the_settings_so_a_late_client_still_sees_them(
        self, routes, tmp_path, monkeypatch
    ) -> None:
        settings_file = tmp_path / "user" / "default" / "comfy.settings.json"
        settings_file.parent.mkdir(parents=True)
        settings_file.write_text(
            json.dumps({"chatterbox.use_custom_path": True}), encoding="utf-8"
        )
        monkeypatch.setattr(
            settings_module.ChatterboxSettings,
            "_get_comfy_settings_path",
            staticmethod(lambda: str(settings_file)),
        )
        model_info.AVAILABLE_CHATTERBOX_MODELS.clear()
        model_info.AVAILABLE_CHATTERBOX_MODELS.update(
            {"local_pack": {"type": "local", "path": str(tmp_path / "pack")}}
        )

        response = run(routes.handler("GET", f"{API_PREFIX}/models")(FakeRequest()))

        assert response.payload["settings"]["use_custom_path"] is True
        assert response.payload["model_packs"] == [DEFAULT_MODEL_PACK_NAME]
        assert response.payload["models"][0]["type"] == "local"
        assert response.payload["models"][0]["path"] == str(tmp_path / "pack")

    def test_no_scan_block_unless_a_path_is_supplied(self, routes, tmp_path) -> None:
        make_pack(tmp_path / "elsewhere", "chatterbox-turbo", ("t3_turbo_v1.safetensors",))

        without = run(routes.handler("GET", f"{API_PREFIX}/models")(FakeRequest()))
        assert "scan" not in without.payload

        with_path = run(
            routes.handler("GET", f"{API_PREFIX}/models")(
                FakeRequest(query={"path": str(tmp_path / "elsewhere")})
            )
        )
        assert with_path.payload["scan"]["count"] == 1
        assert with_path.payload["scan"]["models"][0]["name"] == "chatterbox-turbo"


# ── POST /chatterbox/cancel_download ─────────────────────────────────

class TestCancelDownload:
    def _install_manager(self, monkeypatch, manager: FakeDownloadManager) -> None:
        monkeypatch.setattr(downloader, "get_download_manager", lambda: manager)

    def test_with_an_active_download_it_reports_success(self, routes, monkeypatch) -> None:
        manager = FakeDownloadManager(cancel_result=True)
        self._install_manager(monkeypatch, manager)

        response = run(
            routes.handler("POST", f"{API_PREFIX}/cancel_download")(
                FakeRequest({"model_name": "resembleai_default_voice"})
            )
        )

        assert response.status == 200
        assert response.payload == {
            "success": True,
            "message": "Download cancelled for 'resembleai_default_voice'",
        }
        assert manager.cancelled == ["resembleai_default_voice"]

    def test_without_an_active_download_it_is_a_no_op_not_a_failure(
        self, routes, monkeypatch
    ) -> None:
        manager = FakeDownloadManager(cancel_result=False)
        self._install_manager(monkeypatch, manager)

        response = run(
            routes.handler("POST", f"{API_PREFIX}/cancel_download")(
                FakeRequest({"model_name": "chatterbox-turbo"})
            )
        )

        assert response.status == 200, "cancelling nothing must not be a 500"
        assert response.payload == {
            "success": False,
            "message": "No active download for 'chatterbox-turbo'",
        }
        assert manager.cancelled == ["chatterbox-turbo"]

    def test_a_missing_model_name_is_a_400(self, routes) -> None:
        response = run(
            routes.handler("POST", f"{API_PREFIX}/cancel_download")(FakeRequest({}))
        )

        assert response.status == 400
        assert response.payload == {"error": "model_name is required"}

    def test_a_malformed_body_is_a_400(self, routes) -> None:
        response = run(
            routes.handler("POST", f"{API_PREFIX}/cancel_download")(
                FakeRequest(json.JSONDecodeError("bad", "{", 0))
            )
        )

        assert response.status == 400
        assert "Invalid JSON body" in response.payload["error"]


# ── GET /chatterbox/download_status ──────────────────────────────────

class TestDownloadStatus:
    def test_returns_the_downloads_mapping(self, routes, monkeypatch) -> None:
        statuses = {
            "resembleai_default_voice": {
                "model_name": "resembleai_default_voice",
                "status": "downloading",
                "progress_percentage": 42.0,
            }
        }
        monkeypatch.setattr(
            downloader,
            "get_download_manager",
            lambda: FakeDownloadManager(cancel_result=False, statuses=statuses),
        )

        response = run(routes.handler("GET", f"{API_PREFIX}/download_status")(FakeRequest()))

        assert response.status == 200
        assert list(response.payload) == ["downloads"]
        assert response.payload["downloads"]["resembleai_default_voice"]["status"] == (
            "downloading"
        )

    def test_is_empty_while_nothing_downloads(self, routes, monkeypatch) -> None:
        monkeypatch.setattr(
            downloader,
            "get_download_manager",
            lambda: FakeDownloadManager(cancel_result=False),
        )

        response = run(routes.handler("GET", f"{API_PREFIX}/download_status")(FakeRequest()))

        assert response.payload == {"downloads": {}}


# ── GET /chatterbox/tts_search_paths ─────────────────────────────────

class TestTtsSearchPaths:
    def test_reports_the_packs_under_the_default_root(self, routes, fake_web, tmp_path) -> None:
        tts_root = tmp_path / "models" / TTS_FOLDER_KEY
        pack = make_pack(tts_root / "chatterbox", "chatterbox-nano", ("t3_nano_v1.safetensors",))
        router = FakeRouter()
        register_routes(
            router, FakeFolderPaths(tmp_path / "models", [str(tts_root)]), str(tmp_path)
        )

        response = run(
            router.handler("GET", f"{API_PREFIX}/tts_search_paths")(FakeRequest())
        )

        assert response.status == 200
        entry = response.payload["paths"][0]
        assert entry["path"] == str(tts_root)
        assert entry["is_default"] is True
        assert entry["chatterbox_subdir"] == str(tts_root / "chatterbox")
        assert [model["name"] for model in entry["models"]] == ["chatterbox-nano"]
        assert entry["models"][0]["architecture"] == "chatterbox_nano"
        assert entry["models"][0]["path"] == str(pack)

    def test_an_unregistered_tts_key_degrades_to_no_paths(self, routes, fake_web, tmp_path) -> None:
        router = FakeRouter()
        folder_paths = FakeFolderPaths(tmp_path / "models")
        folder_paths._paths.pop(TTS_FOLDER_KEY)  # a fresh ComfyUI with no 'tts'
        register_routes(router, folder_paths, str(tmp_path))

        response = run(
            router.handler("GET", f"{API_PREFIX}/tts_search_paths")(FakeRequest())
        )

        assert response.status == 200
        assert response.payload == {"paths": []}


# ── POST /chatterbox/validate_directory ──────────────────────────────

class TestValidateDirectory:
    def _validate(self, router, path):
        return run(
            router.handler("POST", f"{API_PREFIX}/validate_directory")(
                FakeRequest({"path": path})
            )
        )

    def test_rejects_a_registered_directory_missing_weights_with_a_reason(
        self, routes, fake_web, tmp_path
    ) -> None:
        empty = tmp_path / "models" / TTS_FOLDER_KEY / "chatterbox"
        empty.mkdir(parents=True)
        router = FakeRouter()
        register_routes(
            router, FakeFolderPaths(tmp_path / "models", [str(empty)]), str(tmp_path)
        )

        response = self._validate(router, str(empty))

        assert response.status == 200
        assert response.payload["valid"] is False
        assert response.payload["exists"] is True
        assert response.payload["is_registered"] is True
        assert response.payload["models"] == []
        assert "tokenizer.json" in response.payload["error"]
        assert "safetensors" in response.payload["error"]

    def test_names_the_missing_weights_only_when_the_tokenizer_is_there(
        self, routes, fake_web, tmp_path
    ) -> None:
        root = tmp_path / "models" / TTS_FOLDER_KEY
        pack = make_pack(root, "weights_only")
        (pack / "tokenizer.json").unlink()
        router = FakeRouter()
        register_routes(router, FakeFolderPaths(tmp_path / "models", [str(root)]), str(tmp_path))

        response = self._validate(router, str(root))

        assert response.payload["valid"] is False
        assert "safetensors" in response.payload["error"]
        assert "t3_cfg.safetensors" in response.payload["error"]

    def test_a_directory_holding_a_pack_is_valid(self, routes, fake_web, tmp_path) -> None:
        root = tmp_path / "models" / TTS_FOLDER_KEY
        make_pack(root, "resembleai_default_voice")
        router = FakeRouter()
        register_routes(router, FakeFolderPaths(tmp_path / "models", [str(root)]), str(tmp_path))

        response = self._validate(router, str(root))

        assert response.payload["valid"] is True
        assert response.payload["error"] is None
        assert response.payload["registered_parent"] == str(root)

    def test_a_pack_itself_is_valid_even_outside_a_registered_root(
        self, routes, fake_web, tmp_path
    ) -> None:
        pack = make_pack(tmp_path / "downloads", "tts_turbo", ("t3_turbo_v1.safetensors",))
        router = FakeRouter()
        register_routes(router, FakeFolderPaths(tmp_path / "models"), str(tmp_path))

        response = self._validate(router, str(pack))

        assert response.payload["valid"] is True
        assert response.payload["is_registered"] is False
        assert response.payload["is_direct_model"] is True
        assert response.payload["direct_model_info"]["architecture"] == "chatterbox_turbo"

    def test_a_missing_directory_is_reported_not_raised(self, routes, tmp_path) -> None:
        response = self._validate(routes, str(tmp_path / "definitely" / "not" / "here"))

        assert response.status == 200
        assert response.payload["valid"] is False
        assert response.payload["exists"] is False
        assert "does not exist" in response.payload["error"]

    def test_a_relative_path_is_a_400(self, routes) -> None:
        response = self._validate(routes, "models/tts/chatterbox")

        assert response.status == 400
        assert response.payload == {"error": "Path must be absolute"}

    def test_a_missing_path_key_is_a_400(self, routes) -> None:
        response = run(
            routes.handler("POST", f"{API_PREFIX}/validate_directory")(FakeRequest({}))
        )

        assert response.status == 400
        assert response.payload == {"error": "path is required"}


# ── POST /chatterbox/register_model_path ─────────────────────────────

class TestRegisterModelPath:
    def _register(self, router, path):
        return run(
            router.handler("POST", f"{API_PREFIX}/register_model_path")(
                FakeRequest({"path": path})
            )
        )

    def _router(self, fake_web, tmp_path, folder_paths):
        router = FakeRouter()
        register_routes(router, folder_paths, str(tmp_path))
        return router

    def test_a_direct_model_directory_returns_a_one_element_list(
        self, routes, fake_web, tmp_path, monkeypatch
    ) -> None:
        pack = make_pack(tmp_path / "downloads", "tts_nano", ("t3_nano_v1.safetensors",))
        original_scan = model_info.scan_custom_model_path
        scans: list[str] = []

        def spy(path):
            scans.append(path)
            return original_scan(path)

        monkeypatch.setattr(model_info, "scan_custom_model_path", spy)
        router = self._router(fake_web, tmp_path, FakeFolderPaths(tmp_path / "models"))

        response = self._register(router, str(pack))

        assert response.status == 200
        assert response.payload["success"] is True
        assert response.payload["already_registered"] is False
        assert response.payload["model_count"] == 1
        assert response.payload["models"][0]["name"] == "tts_nano"
        assert response.payload["models"][0]["path"] == str(pack)
        assert response.payload["warning"] is None
        assert scans == [str(pack)], "scan_custom_model_path must be the scan entry point"

    def test_a_parent_directory_returns_the_discovered_packs(
        self, routes, fake_web, tmp_path
    ) -> None:
        parent = tmp_path / "downloads"
        make_pack(parent, "pack_a")
        make_pack(parent, "pack_b", ("t3_mtl23ls_v2.safetensors",))
        (parent / "not_a_model").mkdir()
        folder_paths = FakeFolderPaths(tmp_path / "models")
        router = self._router(fake_web, tmp_path, folder_paths)

        response = self._register(router, str(parent))

        assert response.status == 200
        assert [model["name"] for model in response.payload["models"]] == [
            "pack_a",
            "pack_b",
        ]
        assert response.payload["model_count"] == 2
        assert folder_paths.added == [(TTS_FOLDER_KEY, str(parent))]
        assert str(parent) in folder_paths.get_folder_paths(TTS_FOLDER_KEY)

    def test_the_chatterbox_subdirectory_convention_is_scanned(
        self, routes, fake_web, tmp_path
    ) -> None:
        root = tmp_path / "downloads"
        make_pack(root / "chatterbox", "resembleai_default_voice")
        router = self._router(fake_web, tmp_path, FakeFolderPaths(tmp_path / "models"))

        response = self._register(router, str(root))

        assert [model["name"] for model in response.payload["models"]] == [
            "chatterbox"
        ]

    def test_discovered_packs_land_in_the_registry(
        self, routes, fake_web, tmp_path
    ) -> None:
        parent = tmp_path / "downloads"
        pack = make_pack(parent, "my_local_pack")
        router = self._router(fake_web, tmp_path, FakeFolderPaths(tmp_path / "models"))

        response = self._register(router, str(parent))

        entry = model_info.AVAILABLE_CHATTERBOX_MODELS["my_local_pack"]
        assert entry["type"] == "local"
        assert entry["path"] == str(pack)
        assert response.payload["model_count"] == 1

    def test_an_official_entry_is_never_demoted_to_local(
        self, routes, fake_web, tmp_path
    ) -> None:
        parent = tmp_path / "downloads"
        make_pack(parent, "chatterbox-turbo", ("t3_turbo_v1.safetensors",))
        router = self._router(fake_web, tmp_path, FakeFolderPaths(tmp_path / "models"))

        self._register(router, str(parent))

        assert (
            model_info.AVAILABLE_CHATTERBOX_MODELS["chatterbox-turbo"]["type"]
            == "official"
        )

    def test_an_already_registered_path_is_not_added_twice(
        self, routes, fake_web, tmp_path
    ) -> None:
        parent = tmp_path / "downloads"
        make_pack(parent, "pack_a")
        folder_paths = FakeFolderPaths(tmp_path / "models", [str(parent)])
        router = self._router(fake_web, tmp_path, folder_paths)

        response = self._register(router, str(parent))

        assert response.payload["already_registered"] is True
        assert folder_paths.added == []
        assert [model["name"] for model in response.payload["models"]] == ["pack_a"]

    def test_an_empty_directory_still_registers_but_warns(
        self, routes, fake_web, tmp_path
    ) -> None:
        empty = tmp_path / "downloads"
        empty.mkdir()
        router = self._router(fake_web, tmp_path, FakeFolderPaths(tmp_path / "models"))

        response = self._register(router, str(empty))

        assert response.status == 200
        assert response.payload["success"] is True
        assert response.payload["model_count"] == 0
        assert "No Chatterbox models found" in response.payload["warning"]

    def test_registration_broadcasts_the_config(
        self, routes, fake_web, tmp_path, broadcasts
    ) -> None:
        parent = tmp_path / "downloads"
        make_pack(parent, "pack_a")
        router = self._router(fake_web, tmp_path, FakeFolderPaths(tmp_path / "models"))
        broadcasts.clear()

        self._register(router, str(parent))

        assert broadcasts == [None], "the frontend must learn about the new root"

    def test_a_relative_path_is_a_400(self, routes) -> None:
        response = self._register(routes, "relative/models")

        assert response.status == 400
        assert response.payload == {"error": "Path must be absolute"}

    def test_a_missing_directory_is_a_400(self, routes, tmp_path) -> None:
        response = self._register(routes, str(tmp_path / "definitely" / "not" / "here"))

        assert response.status == 400
        assert "does not exist" in response.payload["error"]


# ── DELETE /chatterbox/settings ──────────────────────────────────────

class TestClearSettings:
    def _settings_file(self, tmp_path):
        path = tmp_path / "user" / "default" / "comfy.settings.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "chatterbox.use_custom_path": True,
                    "chatterbox.custom_model_path": "/models",
                    "other.node.setting": "keep me",
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_removes_only_the_chatterbox_keys_and_broadcasts(
        self, fake_web, tmp_path, monkeypatch, broadcasts
    ) -> None:
        settings_file = self._settings_file(tmp_path)
        monkeypatch.setattr(
            settings_module.ChatterboxSettings,
            "_get_comfy_settings_path",
            staticmethod(lambda: str(settings_file)),
        )
        folder_paths = FakeFolderPaths(
            tmp_path / "models", user_directory=tmp_path / "user"
        )
        router = FakeRouter()
        register_routes(router, folder_paths, str(tmp_path))
        broadcasts.clear()

        response = run(router.handler("DELETE", f"{API_PREFIX}/settings")(FakeRequest()))

        assert response.status == 200
        assert response.payload["success"] is True
        assert sorted(response.payload["removed"]) == [
            "chatterbox.custom_model_path",
            "chatterbox.use_custom_path",
        ]
        assert json.loads(settings_file.read_text(encoding="utf-8")) == {
            "other.node.setting": "keep me"
        }
        assert broadcasts == [None], "the cleared settings must reach the clients"

    def test_calls_through_to_the_chatterbox_settings_singleton(
        self, fake_web, tmp_path, monkeypatch
    ) -> None:
        settings_file = self._settings_file(tmp_path)
        monkeypatch.setattr(
            settings_module.ChatterboxSettings,
            "_get_comfy_settings_path",
            staticmethod(lambda: str(settings_file)),
        )
        router = FakeRouter()
        register_routes(
            router,
            FakeFolderPaths(tmp_path / "models", user_directory=tmp_path / "user"),
            str(tmp_path),
        )

        response = run(router.handler("DELETE", f"{API_PREFIX}/settings")(FakeRequest()))

        singleton = settings_module.get_settings()
        assert singleton._settings == {}, "the singleton must have reloaded the file"
        assert response.payload["settings"] == singleton.to_dict()
        assert response.payload["settings"]["use_custom_path"] is False

    def test_a_missing_settings_file_is_a_successful_no_op(
        self, fake_web, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            settings_module.ChatterboxSettings,
            "_get_comfy_settings_path",
            staticmethod(lambda: str(tmp_path / "nope" / "comfy.settings.json")),
        )
        router = FakeRouter()
        register_routes(
            router,
            FakeFolderPaths(tmp_path / "models", user_directory=tmp_path / "user"),
            str(tmp_path),
        )

        response = run(router.handler("DELETE", f"{API_PREFIX}/settings")(FakeRequest()))

        assert response.status == 200
        assert response.payload["removed"] == []

    def test_an_unresolvable_user_directory_is_a_500(
        self, fake_web, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            settings_module.ChatterboxSettings,
            "_get_comfy_settings_path",
            staticmethod(lambda: None),
        )
        router = FakeRouter()
        register_routes(router, FakeFolderPaths(tmp_path / "models"), str(tmp_path))

        response = run(router.handler("DELETE", f"{API_PREFIX}/settings")(FakeRequest()))

        assert response.status == 500
        assert response.payload == {"error": "User directory not found"}


# ── Cross-cutting contracts ──────────────────────────────────────────

class TestWireContract:
    def test_the_progress_channel_is_the_chatterbox_one(self) -> None:
        assert DOWNLOAD_EVENT_CHANNEL == "chatterbox.download_progress"
        assert DOWNLOAD_EVENT_CHANNEL == downloader.PROGRESS_CHANNEL

    def test_cancelled_is_a_first_class_event_type(self) -> None:
        assert DOWNLOAD_EVENT_TYPES == ("progress", "complete", "error", "cancelled")
        assert "cancelled" in DOWNLOAD_EVENT_TYPES
        assert "cancelled" != "error"
        assert downloader.DownloadStatus.CANCELLED.value == "cancelled"
        assert downloader.DownloadStatus.CANCELLED.value != downloader.DownloadStatus.ERROR.value

    def test_the_module_never_imports_comfy_api(self) -> None:
        source = api_routes.__file__
        with open(source, "r", encoding="utf-8") as handle:
            text = handle.read()
        assert "comfy_api" not in text

        tree = ast.parse(text)
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        assert not [name for name in imported if "comfy_api" in name]

    def test_register_config_on_connect_exists_nowhere_in_the_package(
        self, repo_root
    ) -> None:
        """Plan §3.1.2: the borrowed no-op must not come back, anywhere.

        Checked on the AST so a docstring is free to *mention* the rejected
        function; only executable code is searched.
        """
        offenders: list[str] = []
        for directory, _dirnames, filenames in os.walk(os.path.join(repo_root, "modules")):
            for filename in sorted(filenames):
                if not filename.endswith(".py"):
                    continue
                path = os.path.join(directory, filename)
                with open(path, "r", encoding="utf-8") as handle:
                    tree = ast.parse(handle.read())
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name) and node.id == "register_config_on_connect":
                        offenders.append(path)
                    elif isinstance(node, ast.Attribute):
                        if node.attr == "register_config_on_connect":
                            offenders.append(path)
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if node.name == "register_config_on_connect":
                            offenders.append(path)
                    elif isinstance(node, ast.alias):
                        if (node.asname or node.name) == "register_config_on_connect":
                            offenders.append(path)
        assert offenders == [], (
            "register_config_on_connect has been reintroduced in: "
            f"{sorted(set(offenders))}"
        )
        assert not hasattr(config_broadcast, "register_config_on_connect")