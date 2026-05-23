from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(relative_path: str, module_name: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EntrypointSmokeTests(unittest.TestCase):
    def test_build_cache_cli_calls_main(self) -> None:
        module = load_module("audit/build_cache.py", "test_build_cache_entrypoint")

        with patch.object(module, "main") as main_mock:
            module.cli()

        main_mock.assert_called_once_with()

    def test_export_to_json_cli_calls_export(self) -> None:
        module = load_module("web/export_to_json.py", "test_export_to_json_entrypoint")

        with patch.object(module, "export") as export_mock:
            module.cli()

        export_mock.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()