"""
config_loader.py
────────────────
Shared utility module imported by every pipeline script and app.py.
Reads config.json and resolves all paths relative to the repo root,
so the project works on any machine without editing any script.

Usage:
    from config_loader import config

    input_dir  = config.path("xml_input_dir")       # absolute resolved path
    model_name = config.get("extraction.model")     # "gemini-2.5-flash"
    batch_size = config.get("vector_db.batch_size") # 100
"""

import json
import os
from pathlib import Path


class Config:
    """Thin wrapper around config.json with path resolution."""

    def __init__(self):
        # Always resolve relative to the directory this file lives in (repo root)
        self._root = Path(__file__).parent.resolve()
        config_path = self._root / "config.json"

        if not config_path.exists():
            raise FileNotFoundError(
                f"config.json not found at {config_path}. "
                "Make sure you are running from the repo root."
            )

        with open(config_path, "r", encoding="utf-8") as f:
            self._data = json.load(f)

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, dotted_key: str, default=None):
        """
        Read any value from config using dot notation.
        Example: config.get("extraction.model") -> "gemini-2.5-flash"
        """
        keys = dotted_key.split(".")
        node = self._data
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    def path(self, key: str) -> str:
        """
        Read a path from config.paths and resolve it to an absolute path.
        Creates the directory if it doesn't exist yet.
        Example: config.path("xml_input_dir") -> "C:/Users/.../data/grobid_xml"
        """
        raw = self.get(f"paths.{key}")
        if raw is None:
            raise KeyError(
                f"Path key '{key}' not found in config.json under 'paths'. "
                f"Available keys: {list(self._data.get('paths', {}).keys())}"
            )
        resolved = (self._root / raw).resolve()

        # Auto-create directory (not for files — check if it looks like a file)
        if not Path(raw).suffix:
            resolved.mkdir(parents=True, exist_ok=True)

        return str(resolved)

    def root(self) -> str:
        """Return the absolute repo root path."""
        return str(self._root)


# ── Singleton ─────────────────────────────────────────────────────────────────
# Import this object directly in other scripts:
#   from config_loader import config
config = Config()


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    # Windows console fix — force UTF-8 output
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("[OK] Config loaded successfully!\n")
    print(f"  Repo root : {config.root()}")
    print(f"\n  Resolved paths:")
    for key in ["xml_input_dir", "processed_articles_dir", "extracted_data_dir",
                "cleaned_csv", "final_csv", "vector_database"]:
        print(f"    {key:30s} -> {config.path(key)}")
    print(f"\n  Extraction model  : {config.get('extraction.model')}")
    print(f"  Embedding model   : {config.get('vector_db.embedding_model')}")
    print(f"  Thickness cutoff  : {config.get('vector_db.thickness_outlier_threshold_nm')} nm")
