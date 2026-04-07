import importlib.util
import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "vistext" / "data" / "json2ttl" / "json_to_ttl_converter_v2.py"


def load_converter_module():
    spec = importlib.util.spec_from_file_location("json_to_ttl_converter_v2", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class JsonToTTLConverterV2RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_converter_module()
        cls.converter = cls.module.JSONToTTLConverterV2()
        cls.labels_dir = REPO_ROOT / "vistext" / "data" / "labels"

    def _ttl_for(self, img_id: int) -> str:
        with (self.labels_dir / f"{img_id}.json").open("r", encoding="utf-8") as handle:
            return self.converter.convert_json_to_ttl(json.load(handle))

    def test_numeric_prefix_label_is_not_split(self):
        ttl = self._ttl_for(706)
        self.assertIn(':yValue "80 years and older"', ttl)
        self.assertNotIn(':xValue "80" ;\n    :yValue "years and older"', ttl)

    def test_numeric_suffix_label_is_not_split(self):
        ttl = self._ttl_for(1283)
        self.assertIn(':yValue "BBC Radio 2"', ttl)
        self.assertIn(':yValue "BBC Radio 4"', ttl)
        self.assertIn(':yValue "BBC Radio 5 live"', ttl)
        self.assertNotIn(':xValue "5" ;\n    :yValue "live"', ttl)

    def test_scenegraph_repairs_damaged_bar_labels(self):
        ttl = self._ttl_for(193)
        self.assertIn(':xValue "Romelu Lukaku"', ttl)
        self.assertIn(':xValue "Ashley Young"', ttl)
        self.assertNotIn(':xValue "omelu Lukaku"', ttl)

    def test_year_mapping_guard_is_preserved(self):
        ttl = self._ttl_for(1358)
        self.assertIn(':xValue "10.15" ;\n    :yValue "2025*"', ttl)
        self.assertIn(':xValue "8.38" ;\n    :yValue "2015"', ttl)


if __name__ == "__main__":
    unittest.main()
