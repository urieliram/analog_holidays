from __future__ import annotations

from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
PARENT_DIR = REPO_ROOT.parent

if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))


class AnalogPackageSmokeTests(unittest.TestCase):
    def test_analog_package_exports_core_models(self) -> None:
        import analog_holidays.analog as analog_pkg

        self.assertTrue(hasattr(analog_pkg, "AnalogKNN"))
        self.assertTrue(hasattr(analog_pkg, "AnalogSpecialDays"))
        self.assertTrue(hasattr(analog_pkg, "_REGRESSORS"))

    def test_analog_holidays_module_uses_repo_audit_data_path(self) -> None:
        import analog_holidays.analog.analog_holidays as analog_holidays_module

        self.assertEqual(
            analog_holidays_module.DEFAULT_SOURCE_PATH,
            REPO_ROOT / "audit" / "data",
        )


if __name__ == "__main__":
    unittest.main()