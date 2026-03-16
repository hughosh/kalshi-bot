"""Tests for player name resolution."""
import json
import os
import tempfile

import pytest

from integration.player_resolver import PlayerResolver, _last_first_initial, _normalize


class TestNormalize:
    def test_basic(self):
        assert _normalize("Carlos Alcaraz") == "carlos alcaraz"

    def test_accents(self):
        assert _normalize("Rafael Nadál") == "rafael nadal"

    def test_extra_whitespace(self):
        assert _normalize("  Novak   Djokovic  ") == "novak djokovic"


class TestLastFirstInitial:
    def test_two_names(self):
        assert _last_first_initial("Carlos Alcaraz") == "alcaraz c"

    def test_single_name(self):
        assert _last_first_initial("Federer") == "federer"

    def test_three_names(self):
        assert _last_first_initial("Juan Martin Del Potro") == "potro j"


class TestPlayerResolver:
    @pytest.fixture
    def resolver(self, tmp_path):
        # Create aliases
        aliases = {"Nole": "104925", "Rafa": "104745"}
        aliases_path = tmp_path / "aliases.json"
        with open(aliases_path, "w") as f:
            json.dump(aliases, f)

        # Create a minimal ATP CSV
        atp_dir = tmp_path / "atp"
        atp_dir.mkdir()
        csv_path = atp_dir / "atp_matches_2023.csv"
        csv_path.write_text(
            "winner_name,winner_id,loser_name,loser_id,other_col\n"
            "Carlos Alcaraz,100644,Novak Djokovic,104925,x\n"
            "Jannik Sinner,106421,Daniil Medvedev,106401,x\n"
        )
        return PlayerResolver(str(aliases_path), str(atp_dir))

    def test_alias_exact(self, resolver):
        assert resolver.resolve("Nole") == "104925"
        assert resolver.resolve("Rafa") == "104745"

    def test_atp_csv_exact(self, resolver):
        assert resolver.resolve("Carlos Alcaraz") == "100644"
        assert resolver.resolve("Novak Djokovic") == "104925"

    def test_case_insensitive(self, resolver):
        assert resolver.resolve("carlos alcaraz") == "100644"
        assert resolver.resolve("JANNIK SINNER") == "106421"

    def test_last_first_initial_fallback(self, resolver):
        # "C. Alcaraz" → lfi "alcaraz c" should match
        assert resolver.resolve("C. Alcaraz") is not None

    def test_unresolvable(self, resolver):
        assert resolver.resolve("Unknown Player") is None

    def test_missing_files(self, tmp_path):
        resolver = PlayerResolver(str(tmp_path / "nope.json"), str(tmp_path / "nope_dir"))
        assert resolver.resolve("Anyone") is None
