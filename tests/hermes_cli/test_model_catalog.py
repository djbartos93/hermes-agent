"""Tests for hermes_cli.model_catalog — remote manifest fetch + cache + fallback."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Isolate HERMES_HOME + reset any module-level catalog cache per test."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    # Force a fresh catalog module state for each test.
    import importlib
    from hermes_cli import model_catalog
    importlib.reload(model_catalog)
    yield home
    model_catalog.reset_cache()


def _valid_manifest() -> dict:
    return {
        "version": 1,
        "updated_at": "2026-04-25T22:00:00Z",
        "metadata": {"source": "test"},
        "providers": {
            "openrouter": {
                "metadata": {"display_name": "OpenRouter"},
                "models": [
                    {"id": "anthropic/claude-opus-4.7", "description": "recommended"},
                    {"id": "openai/gpt-5.4", "description": ""},
                    {"id": "openrouter/elephant-alpha", "description": "free"},
                ],
            },
            "nous": {
                "metadata": {"display_name": "Nous Portal"},
                "models": [
                    {"id": "anthropic/claude-opus-4.7"},
                    {"id": "moonshotai/kimi-k2.6"},
                ],
            },
        },
    }


class TestValidation:
    def test_accepts_well_formed_manifest(self, isolated_home):
        from hermes_cli.model_catalog import _validate_manifest
        assert _validate_manifest(_valid_manifest()) is True

    def test_rejects_non_dict(self, isolated_home):
        from hermes_cli.model_catalog import _validate_manifest
        assert _validate_manifest("string") is False
        assert _validate_manifest([]) is False
        assert _validate_manifest(None) is False

    def test_rejects_missing_version(self, isolated_home):
        from hermes_cli.model_catalog import _validate_manifest
        m = _valid_manifest()
        del m["version"]
        assert _validate_manifest(m) is False

    def test_rejects_future_version(self, isolated_home):
        from hermes_cli.model_catalog import _validate_manifest
        m = _valid_manifest()
        m["version"] = 999
        assert _validate_manifest(m) is False

    def test_rejects_missing_providers(self, isolated_home):
        from hermes_cli.model_catalog import _validate_manifest
        m = _valid_manifest()
        del m["providers"]
        assert _validate_manifest(m) is False

    def test_rejects_malformed_model_entry(self, isolated_home):
        from hermes_cli.model_catalog import _validate_manifest
        m = _valid_manifest()
        m["providers"]["openrouter"]["models"][0] = {"id": ""}  # empty id
        assert _validate_manifest(m) is False

    def test_rejects_non_string_model_id(self, isolated_home):
        from hermes_cli.model_catalog import _validate_manifest
        m = _valid_manifest()
        m["providers"]["openrouter"]["models"][0] = {"id": 42}
        assert _validate_manifest(m) is False


class TestFetchSuccess:
    def test_fetch_and_cache_writes_disk(self, isolated_home):
        from hermes_cli import model_catalog
        manifest = _valid_manifest()
        with patch.object(
            model_catalog, "_fetch_manifest", return_value=manifest
        ) as fetch:
            result = model_catalog.get_catalog(force_refresh=True)

        assert result == manifest
        assert fetch.called

        cache_file = model_catalog._cache_path()
        assert cache_file.exists()
        with open(cache_file) as fh:
            assert json.load(fh) == manifest

    def test_second_call_uses_in_process_cache(self, isolated_home):
        from hermes_cli import model_catalog
        manifest = _valid_manifest()
        with patch.object(
            model_catalog, "_fetch_manifest", return_value=manifest
        ) as fetch:
            model_catalog.get_catalog(force_refresh=True)
            model_catalog.get_catalog()  # should not hit network again
        assert fetch.call_count == 1

    def test_force_refresh_always_refetches(self, isolated_home):
        from hermes_cli import model_catalog
        manifest = _valid_manifest()
        with patch.object(
            model_catalog, "_fetch_manifest", return_value=manifest
        ) as fetch:
            model_catalog.get_catalog(force_refresh=True)
            model_catalog.get_catalog(force_refresh=True)
        assert fetch.call_count == 2


class TestFetchFailure:
    def test_network_failure_returns_empty_when_no_cache(self, isolated_home):
        from hermes_cli import model_catalog
        with patch.object(model_catalog, "_fetch_manifest", return_value=None):
            result = model_catalog.get_catalog(force_refresh=True)
        assert result == {}

    def test_network_failure_falls_back_to_disk_cache(self, isolated_home):
        from hermes_cli import model_catalog
        # Prime disk cache with a fresh copy.
        manifest = _valid_manifest()
        with patch.object(model_catalog, "_fetch_manifest", return_value=manifest):
            model_catalog.get_catalog(force_refresh=True)

        # Now wipe in-process cache and simulate network failure on refetch.
        model_catalog.reset_cache()
        with patch.object(model_catalog, "_fetch_manifest", return_value=None):
            result = model_catalog.get_catalog(force_refresh=True)

        assert result == manifest

    def test_fetch_failure_falls_back_to_stale_cache(self, isolated_home):
        from hermes_cli import model_catalog
        manifest = _valid_manifest()
        # Write stale cache directly (mtime in the past).
        cache = model_catalog._cache_path()
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "w") as fh:
            json.dump(manifest, fh)
        old = time.time() - 30 * 24 * 3600  # 30 days ago
        import os as _os
        _os.utime(cache, (old, old))

        with patch.object(model_catalog, "_fetch_manifest", return_value=None):
            result = model_catalog.get_catalog()

        # Stale cache is better than nothing.
        assert result == manifest


class TestCuratedAccessors:
    def test_openrouter_returns_tuples(self, isolated_home):
        from hermes_cli import model_catalog
        with patch.object(
            model_catalog, "_fetch_manifest", return_value=_valid_manifest()
        ):
            result = model_catalog.get_curated_openrouter_models()
        assert result == [
            ("anthropic/claude-opus-4.7", "recommended"),
            ("openai/gpt-5.4", ""),
            ("openrouter/elephant-alpha", "free"),
        ]

    def test_nous_returns_ids(self, isolated_home):
        from hermes_cli import model_catalog
        with patch.object(
            model_catalog, "_fetch_manifest", return_value=_valid_manifest()
        ):
            result = model_catalog.get_curated_nous_models()
        assert result == ["anthropic/claude-opus-4.7", "moonshotai/kimi-k2.6"]

    def test_openrouter_returns_none_when_catalog_empty(self, isolated_home):
        from hermes_cli import model_catalog
        with patch.object(model_catalog, "_fetch_manifest", return_value=None):
            assert model_catalog.get_curated_openrouter_models() is None

    def test_nous_returns_none_when_catalog_empty(self, isolated_home):
        from hermes_cli import model_catalog
        with patch.object(model_catalog, "_fetch_manifest", return_value=None):
            assert model_catalog.get_curated_nous_models() is None


class TestDisabled:
    def test_disabled_config_short_circuits(self, isolated_home):
        from hermes_cli import model_catalog
        with patch.object(
            model_catalog,
            "_load_catalog_config",
            return_value={
                "enabled": False,
                "url": "http://ignored",
                "ttl_hours": 24.0,
                "providers": {},
            },
        ):
            with patch.object(model_catalog, "_fetch_manifest") as fetch:
                result = model_catalog.get_catalog()
        assert result == {}
        fetch.assert_not_called()


class TestProviderOverride:
    def test_override_url_takes_precedence(self, isolated_home):
        from hermes_cli import model_catalog

        override_payload = {
            "version": 1,
            "providers": {
                "openrouter": {
                    "models": [
                        {"id": "override/model", "description": "custom"},
                    ]
                }
            },
        }

        def fake_fetch(url, timeout):
            if "override" in url:
                return override_payload
            return _valid_manifest()

        with patch.object(
            model_catalog,
            "_load_catalog_config",
            return_value={
                "enabled": True,
                "url": "http://master",
                "ttl_hours": 24.0,
                "providers": {"openrouter": {"url": "http://override"}},
            },
        ):
            with patch.object(model_catalog, "_fetch_manifest", side_effect=fake_fetch):
                result = model_catalog.get_curated_openrouter_models()

        assert result == [("override/model", "custom")]


class TestIntegrationWithModelsModule:
    """Exercise the fallback paths via the real callers in hermes_cli.models."""

    def test_curated_nous_ids_falls_back_to_hardcoded_on_empty_catalog(
        self, isolated_home
    ):
        from hermes_cli import model_catalog
        from hermes_cli.models import get_curated_nous_model_ids, _PROVIDER_MODELS

        with patch.object(model_catalog, "_fetch_manifest", return_value=None):
            result = get_curated_nous_model_ids()

        assert result == list(_PROVIDER_MODELS["nous"])

    def test_curated_nous_ids_prefers_manifest(self, isolated_home):
        from hermes_cli import model_catalog
        from hermes_cli.models import get_curated_nous_model_ids

        with patch.object(
            model_catalog, "_fetch_manifest", return_value=_valid_manifest()
        ):
            result = get_curated_nous_model_ids()

        assert result == ["anthropic/claude-opus-4.7", "moonshotai/kimi-k2.6"]


class TestExtraModels:
    """User-defined extras file at ``$HERMES_HOME/extra_models.yaml``."""

    def _write_extras(self, home, contents: str) -> None:
        (home / "extra_models.yaml").write_text(contents)

    def test_missing_file_returns_empty_dict(self, isolated_home):
        from hermes_cli.model_catalog import load_extra_models
        assert load_extra_models() == {}

    def test_loads_simple_string_lists(self, isolated_home):
        from hermes_cli.model_catalog import load_extra_models
        self._write_extras(
            isolated_home,
            "openrouter:\n  - x-ai/grok-4\n  - moonshotai/kimi-k2-0905\n"
            "nous:\n  - Hermes-4-405B\n",
        )
        assert load_extra_models() == {
            "openrouter": ["x-ai/grok-4", "moonshotai/kimi-k2-0905"],
            "nous": ["Hermes-4-405B"],
        }

    def test_accepts_dict_entries_with_id(self, isolated_home):
        from hermes_cli.model_catalog import load_extra_models
        self._write_extras(
            isolated_home,
            "openrouter:\n"
            "  - id: x-ai/grok-4\n"
            "  - some-string/model\n",
        )
        assert load_extra_models() == {
            "openrouter": ["x-ai/grok-4", "some-string/model"],
        }

    def test_skips_malformed_entries(self, isolated_home):
        from hermes_cli.model_catalog import load_extra_models
        self._write_extras(
            isolated_home,
            "openrouter:\n  - 42\n  - null\n  - {}\n  - x-ai/grok-4\n",
        )
        assert load_extra_models() == {"openrouter": ["x-ai/grok-4"]}

    def test_invalid_yaml_returns_empty(self, isolated_home):
        from hermes_cli.model_catalog import load_extra_models
        self._write_extras(isolated_home, "openrouter: [unclosed\n")
        assert load_extra_models() == {}

    def test_top_level_non_dict_returns_empty(self, isolated_home):
        from hermes_cli.model_catalog import load_extra_models
        self._write_extras(isolated_home, "- just\n- a\n- list\n")
        assert load_extra_models() == {}

    def test_mtime_invalidation_picks_up_edits(self, isolated_home):
        import os
        from hermes_cli.model_catalog import load_extra_models

        self._write_extras(isolated_home, "openrouter:\n  - x-ai/grok-4\n")
        assert load_extra_models() == {"openrouter": ["x-ai/grok-4"]}

        # Bump mtime forward so the cache invalidates regardless of FS resolution.
        path = isolated_home / "extra_models.yaml"
        path.write_text("openrouter:\n  - moonshotai/kimi-k2-0905\n")
        future = time.time() + 10
        os.utime(path, (future, future))

        assert load_extra_models() == {"openrouter": ["moonshotai/kimi-k2-0905"]}

    def test_openrouter_picker_unions_extras(self, isolated_home):
        """End-to-end: extras should appear in fetch_openrouter_models output."""
        from hermes_cli import models, model_catalog

        self._write_extras(isolated_home, "openrouter:\n  - aux/extra-model\n")

        # Reset the openrouter cache so this test runs clean.
        models._openrouter_catalog_cache = None
        models._openrouter_catalog_extras_mtime = 0.0

        # Fake the live /v1/models to include both a curated id and the extra.
        live_payload = {
            "data": [
                {
                    "id": "moonshotai/kimi-k2.6",
                    "supported_parameters": ["tools"],
                    "pricing": {"prompt": "0", "completion": "0"},
                },
                {
                    "id": "aux/extra-model",
                    "supported_parameters": ["tools"],
                    "pricing": {"prompt": "1", "completion": "1"},
                },
            ]
        }

        class _FakeResp:
            def __init__(self, payload):
                self._payload = payload
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def read(self):
                return json.dumps(self._payload).encode()

        with patch.object(
            model_catalog, "_fetch_manifest", return_value=_valid_manifest()
        ):
            with patch(
                "urllib.request.urlopen",
                return_value=_FakeResp(live_payload),
            ):
                result = models.fetch_openrouter_models(force_refresh=True)

        ids = [mid for mid, _ in result]
        assert "aux/extra-model" in ids

    def test_nous_curated_ids_unions_extras(self, isolated_home):
        from hermes_cli import models, model_catalog

        self._write_extras(isolated_home, "nous:\n  - Hermes-4-Custom\n")

        with patch.object(
            model_catalog, "_fetch_manifest", return_value=_valid_manifest()
        ):
            result = models.get_curated_nous_model_ids()

        assert "Hermes-4-Custom" in result
        # Curated ids still present.
        assert "anthropic/claude-opus-4.7" in result

    def test_extras_dedupe_against_curated(self, isolated_home):
        from hermes_cli import models, model_catalog

        # "moonshotai/kimi-k2.6" already present in the in-repo fallback.
        self._write_extras(isolated_home, "nous:\n  - moonshotai/kimi-k2.6\n")

        with patch.object(model_catalog, "_fetch_manifest", return_value=None):
            result = models.get_curated_nous_model_ids()

        assert result.count("moonshotai/kimi-k2.6") == 1

    def test_reset_cache_clears_extras(self, isolated_home):
        from hermes_cli import model_catalog

        self._write_extras(isolated_home, "openrouter:\n  - x-ai/grok-4\n")
        model_catalog.load_extra_models()
        assert model_catalog._extra_models_cache is not None

        model_catalog.reset_cache()
        assert model_catalog._extra_models_cache is None
        assert model_catalog._extra_models_cache_mtime == 0.0
