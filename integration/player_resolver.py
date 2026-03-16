"""Resolve API-Tennis display names to ATP numeric IDs.

Resolution chain (highest priority first):
  1. data/aliases.json — hand-maintained overrides
  2. Name dict built from raw ATP CSVs (winner_name/id, loser_name/id columns)
     Normalized: lowercase, strip accents, last_name + first_initial comparison
  3. Return None if unresolved (match will be skipped by the bot)
"""
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)


def _normalize(name: str) -> str:
    """Lowercase, strip accents, collapse whitespace."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", ascii_str.lower().strip())


def _last_first_initial(name: str) -> str:
    """Return 'lastname f' key for fuzzy matching."""
    parts = _normalize(name).split()
    if not parts:
        return ""
    last = parts[-1]
    first_initial = parts[0][0] if len(parts) > 1 else ""
    return f"{last} {first_initial}".strip()


class PlayerResolver:
    """Thread-safe (read-only after build) name → ATP ID resolver."""

    def __init__(self, aliases_path: str, atp_raw_dir: str) -> None:
        self._exact: dict[str, str] = {}       # normalized full name → id
        self._lfi: dict[str, str] = {}          # last+first_initial → id
        self._load_aliases(aliases_path)
        self._load_atp_csvs(atp_raw_dir)
        log.info("PlayerResolver: %d exact, %d lfi entries", len(self._exact), len(self._lfi))

    def _load_aliases(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            log.warning("Aliases file not found: %s", path)
            return
        with open(p) as f:
            raw = json.load(f)
        for name, pid in raw.items():
            if name.startswith("_"):
                continue
            norm = _normalize(name)
            self._exact[norm] = str(pid)
            lfi = _last_first_initial(name)
            if lfi:
                self._lfi[lfi] = str(pid)

    def _load_atp_csvs(self, atp_raw_dir: str) -> None:
        p = Path(atp_raw_dir)
        if not p.exists():
            log.warning("ATP raw dir not found: %s", atp_raw_dir)
            return
        csv_files = sorted(p.glob("atp_matches_*.csv"))
        if not csv_files:
            log.warning("No atp_matches_*.csv files found in %s", atp_raw_dir)
            return

        name_col_pairs = [("winner_name", "winner_id"), ("loser_name", "loser_id")]
        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path, usecols=["winner_name", "winner_id", "loser_name", "loser_id"], low_memory=False)
            except Exception as e:
                log.debug("Skipping %s: %s", csv_path.name, e)
                continue
            for name_col, id_col in name_col_pairs:
                for _, row in df[[name_col, id_col]].dropna().iterrows():
                    name = str(row[name_col])
                    pid = str(int(float(row[id_col])))
                    norm = _normalize(name)
                    if norm not in self._exact:
                        self._exact[norm] = pid
                    lfi = _last_first_initial(name)
                    if lfi and lfi not in self._lfi:
                        self._lfi[lfi] = pid

    def resolve(self, display_name: str) -> Optional[str]:
        """Return ATP ID string or None if unresolvable."""
        # 1. exact normalized match
        norm = _normalize(display_name)
        if norm in self._exact:
            return self._exact[norm]
        # 2. last name + first initial
        lfi = _last_first_initial(display_name)
        if lfi in self._lfi:
            return self._lfi[lfi]
        # 3. last name only (less reliable — only if unique)
        parts = norm.split()
        if parts:
            last = parts[-1]
            candidates = [v for k, v in self._lfi.items() if k.startswith(last + " ") or k == last]
            if len(candidates) == 1:
                return candidates[0]
        log.debug("PlayerResolver: unresolved '%s'", display_name)
        return None
