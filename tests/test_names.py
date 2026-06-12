import pytest

from wc2026.data.names import (
    NameResolver,
    UnresolvedTeamNameError,
    canonical_slug,
    default_resolver,
    load_canonical_ids,
)


class TestCanonicalSlug:
    @pytest.mark.parametrize(
        ("name", "slug"),
        [
            ("Côte d'Ivoire", "cote_divoire"),
            ("Bosnia and Herzegovina", "bosnia_and_herzegovina"),
            ("Curaçao", "curacao"),
            ("USA", "usa"),
            ("São Tomé and Príncipe", "sao_tome_and_principe"),
        ],
    )
    def test_normalization(self, name: str, slug: str) -> None:
        assert canonical_slug(name) == slug

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty slug"):
            canonical_slug("---")


class TestNameResolver:
    @pytest.fixture
    def resolver(self) -> NameResolver:
        return NameResolver(
            frozenset({"brazil", "united_states", "czech_republic"}),
            overrides={"odds_api": {"usa": "united_states"}},
        )

    def test_direct_hit(self, resolver: NameResolver) -> None:
        assert resolver.resolve("any_source", "Brazil") == "brazil"

    def test_override_applies_per_source(self, resolver: NameResolver) -> None:
        assert resolver.resolve("odds_api", "USA") == "united_states"
        with pytest.raises(UnresolvedTeamNameError):
            resolver.resolve("other_source", "USA")

    def test_unresolved_raises_loudly(self, resolver: NameResolver) -> None:
        with pytest.raises(UnresolvedTeamNameError, match=r"odds_api.*'Atlantis'"):
            resolver.resolve("odds_api", "Atlantis")

    def test_coverage_collects_instead_of_raising(self, resolver: NameResolver) -> None:
        report = resolver.coverage("odds_api", ["USA", "Brazil", "Atlantis", "Mordor"])
        assert report.mapped == {"USA": "united_states", "Brazil": "brazil"}
        assert report.unmapped == ("Atlantis", "Mordor")
        assert report.coverage == 0.5


class TestPackagedUniverse:
    def test_loads_and_contains_known_ids(self) -> None:
        ids = load_canonical_ids()
        assert {"brazil", "united_states", "ivory_coast", "czech_republic"} <= ids
        assert len(ids) > 300

    def test_default_resolver_handles_live_feed_spellings(self) -> None:
        resolver = default_resolver()
        # verified against the live feeds on 2026-06-12
        assert resolver.resolve("odds_api", "USA") == "united_states"
        assert resolver.resolve("football_data", "Cape Verde Islands") == "cape_verde"
        assert resolver.resolve("football_data", "Congo DR") == "dr_congo"
        assert resolver.resolve("football_data", "Czechia") == "czech_republic"
