import hashlib
import tempfile
import unittest
from pathlib import Path
from src.verify_frozen_v3 import verify_manifest


class FrozenV3ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "original.py").write_text("print('frozen')\n")
        digest = hashlib.sha256((self.root / "original.py").read_bytes()).hexdigest()
        self.manifest = {"scenarios": [
            {"scenario_id": f"ORIGINAL-{i:02}", "implementation_path": "original.py",
             "implementation_sha256": digest, "parameters": {"threshold": i},
             "lookback": "7D", "entity_key": "sender"} for i in range(11)
        ]}

    def test_complete_manifest(self):
        self.assertEqual(verify_manifest(self.manifest, self.root)["scenario_count"], 11)

    def test_four_or_twelve_scenarios_rejected(self):
        for count in (4, 12):
            with self.assertRaises(ValueError):
                verify_manifest({"scenarios": self.manifest["scenarios"][:count]
                                 + ([self.manifest["scenarios"][0]] if count == 12 else [])}, self.root)

    def test_changed_code_rejected(self):
        (self.root / "original.py").write_text("print('changed')\n")
        with self.assertRaises(ValueError):
            verify_manifest(self.manifest, self.root)

    def test_missing_threshold_specification_rejected(self):
        del self.manifest["scenarios"][0]["parameters"]
        with self.assertRaises(ValueError):
            verify_manifest(self.manifest, self.root)


if __name__ == "__main__":
    unittest.main()
