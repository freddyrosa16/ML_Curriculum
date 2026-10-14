"""Tests for 07-data-and-deployment/03-apis.

Non-slow tests replace `requests.get` with a fake that serves canned JSON, so they
run offline and never hit rate limits. The slow tests call the real APIs.
"""
import time

import pytest

from helpers import load_task

requests = pytest.importorskip("requests")


class FakeResponse:
    def __init__(self, payload=None, status_code=200, headers=None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.ok = status_code < 400

    def json(self):
        return self._payload

    def raise_for_status(self):
        if not self.ok:
            raise requests.HTTPError(f"{self.status_code}")


def install_fake(monkeypatch, router):
    """Make requests.get(url, ...) return router(url, params)."""
    calls = []

    def fake_get(url, params=None, **kwargs):
        calls.append((url, params))
        return router(url, params or {})

    monkeypatch.setattr(requests, "get", fake_get)
    return calls


# ---------- 0. availableShips (SWAPI) ----------

SHIPS_PAGE2 = "https://swapi.dev/api/starships/?page=2"
SHIPS = {
    1: {"next": SHIPS_PAGE2, "results": [
        {"name": "CR90 corvette", "passengers": "600"},
        {"name": "Sentinel-class landing craft", "passengers": "75"},
        {"name": "Death Star", "passengers": "843,342"},
        {"name": "X-wing", "passengers": "0"},
        {"name": "TIE Advanced x1", "passengers": "n/a"},
    ]},
    2: {"next": None, "results": [
        {"name": "Millennium Falcon", "passengers": "6"},
        {"name": "Y-wing", "passengers": "0"},
        {"name": "Executor", "passengers": "38000"},
        {"name": "Slave 1", "passengers": "6"},
        {"name": "Imperial shuttle", "passengers": "20"},
        {"name": "EF76 Nebulon-B escort frigate", "passengers": "75"},
        {"name": "Calamari Cruiser", "passengers": "1200"},
        {"name": "A-wing", "passengers": "0"},
        {"name": "B-wing", "passengers": "0"},
        {"name": "Republic Cruiser", "passengers": "16"},
        {"name": "Naboo fighter", "passengers": "0"},
        {"name": "Droid control ship", "passengers": "unknown"},
        {"name": "Rebel transport", "passengers": "90"},
    ]},
}


def swapi_ships_router(url, params):
    if url == SHIPS_PAGE2 or params.get("page") in (2, "2"):
        return FakeResponse(SHIPS[2])
    if "starships" in url:
        return FakeResponse(SHIPS[1])
    return FakeResponse({"detail": "Not found"}, 404)


def test_available_ships_follows_pagination(monkeypatch):
    calls = install_fake(monkeypatch, swapi_ships_router)
    availableShips = load_task(__file__, "0-passengers.py", "availableShips")
    assert availableShips(4) == [
        "CR90 corvette", "Sentinel-class landing craft", "Death Star",
        "Millennium Falcon", "Executor", "Slave 1", "Imperial shuttle",
        "EF76 Nebulon-B escort frigate", "Calamari Cruiser", "Republic Cruiser",
        "Rebel transport"]
    assert len(calls) == 2


def test_available_ships_threshold_and_empty(monkeypatch):
    install_fake(monkeypatch, swapi_ships_router)
    availableShips = load_task(__file__, "0-passengers.py", "availableShips")
    assert availableShips(1000) == ["Death Star", "Executor", "Calamari Cruiser"]
    assert availableShips(10**7) == []


# ---------- 1. sentientPlanets (SWAPI) ----------

SPECIES_PAGE2 = "https://swapi.dev/api/species/?page=2"
P = "https://swapi.dev/api/planets/{}/"
PLANETS = {P.format(9): "Coruscant", P.format(14): "Kashyyyk",
           P.format(8): "Naboo", P.format(37): "Ord Mantell", P.format(2): "Alderaan"}
SPECIES = {
    1: {"next": SPECIES_PAGE2, "results": [
        {"name": "Human", "classification": "mammal", "designation": "sentient",
         "homeworld": P.format(9)},
        {"name": "Droid", "classification": "artificial", "designation": "sentient",
         "homeworld": None},
        {"name": "Wookie", "classification": "mammal", "designation": "sentient",
         "homeworld": P.format(14)},
        {"name": "Hutt", "classification": "gastropod", "designation": "reptilian",
         "homeworld": P.format(37)},
    ]},
    2: {"next": None, "results": [
        {"name": "Gungan", "classification": "amphibian", "designation": "sentient",
         "homeworld": P.format(8)},
        {"name": "Sentient goo", "classification": "Sentient blob", "designation": "unknown",
         "homeworld": P.format(2)},
        {"name": "Space slug", "classification": "reptile", "designation": "",
         "homeworld": P.format(37)},
    ]},
}


def swapi_species_router(url, params):
    if url in PLANETS:
        return FakeResponse({"name": PLANETS[url], "url": url})
    if url == SPECIES_PAGE2 or params.get("page") in (2, "2"):
        return FakeResponse(SPECIES[2])
    if "species" in url:
        return FakeResponse(SPECIES[1])
    return FakeResponse({"detail": "Not found"}, 404)


def test_sentient_planets(monkeypatch):
    install_fake(monkeypatch, swapi_species_router)
    sentientPlanets = load_task(__file__, "1-sentience.py", "sentientPlanets")
    assert sentientPlanets() == ["Coruscant", "Kashyyyk", "Naboo", "Alderaan"]


# ---------- 2. GitHub user location ----------

def test_user_location_found(monkeypatch):
    install_fake(monkeypatch, lambda url, p: FakeResponse(
        {"login": "octocat", "location": "San Francisco"}))
    get_user_location = load_task(__file__, "2-user_location.py", "get_user_location")
    assert get_user_location("https://api.github.com/users/octocat") == "San Francisco"


def test_user_location_missing(monkeypatch):
    install_fake(monkeypatch, lambda url, p: FakeResponse({"login": "x", "location": None}))
    get_user_location = load_task(__file__, "2-user_location.py", "get_user_location")
    assert get_user_location("https://api.github.com/users/x") == "No location found"


def test_user_not_found(monkeypatch):
    install_fake(monkeypatch, lambda url, p: FakeResponse({"message": "Not Found"}, 404))
    get_user_location = load_task(__file__, "2-user_location.py", "get_user_location")
    assert get_user_location("https://api.github.com/users/nobody-xyz") == "Not found"


def test_user_rate_limited(monkeypatch):
    reset = int(time.time()) + 5 * 60
    install_fake(monkeypatch, lambda url, p: FakeResponse(
        {"message": "API rate limit exceeded"}, 403, {"X-RateLimit-Reset": str(reset)}))
    get_user_location = load_task(__file__, "2-user_location.py", "get_user_location")
    assert get_user_location("https://api.github.com/users/octocat") in (
        "Reset in 4 min", "Reset in 5 min")


def test_user_location_script_not_run_on_import(monkeypatch):
    calls = install_fake(monkeypatch, lambda url, p: FakeResponse({"location": "x"}))
    load_task(__file__, "2-user_location.py")
    assert calls == []


# ---------- 3/4. Launch Library 2 ----------

def ll2_launch(name, net, rocket, pad, locality):
    return {"name": name, "net": net,
            "rocket": {"configuration": {"name": rocket, "full_name": rocket}},
            "pad": {"name": pad, "location": {"name": locality}}}


FIRST_PAGE = {"count": 3, "next": None, "previous": None, "results": [
    ll2_launch("Falcon 1 | DemoSat", "2007-03-21T01:10:00Z", "Falcon 1",
               "Omelek Island", "Kwajalein Atoll"),
    ll2_launch("Falcon 1 | FalconSAT-2", "2006-03-24T22:30:00Z", "Falcon 1",
               "Omelek Island", "Kwajalein Atoll"),
    ll2_launch("Falcon 1 | Trailblazer", "2008-08-03T03:34:00Z", "Falcon 1",
               "Omelek Island", "Kwajalein Atoll"),
]}


def test_first_launch(monkeypatch):
    calls = install_fake(monkeypatch, lambda url, p: FakeResponse(FIRST_PAGE))
    get_first_launch = load_task(__file__, "3-first_launch.py", "get_first_launch")
    assert calls == []   # nothing runs on import
    assert get_first_launch() == ("Falcon 1 | FalconSAT-2 (2006-03-24T22:30:00Z) "
                                  "Falcon 1 - Omelek Island (Kwajalein Atoll)")
    assert all("thespacedevs.com" in url for url, _ in calls)


LAUNCH_PAGE2 = "https://ll.thespacedevs.com/2.3.0/launches/?limit=100&lsp__name=SpaceX&offset=100"
LAUNCH_PAGES = {
    1: {"count": 9, "next": LAUNCH_PAGE2, "results": [
        ll2_launch("a", "2006", "Falcon 1", "p", "l"),
        ll2_launch("b", "2010", "Falcon 9", "p", "l"),
        ll2_launch("c", "2011", "Falcon 9", "p", "l"),
        ll2_launch("d", "2018", "Falcon Heavy", "p", "l"),
        ll2_launch("e", "2023", "Starship", "p", "l"),
    ]},
    2: {"count": 9, "next": None, "results": [
        ll2_launch("f", "2024", "Falcon 9", "p", "l"),
        ll2_launch("g", "2019", "Falcon Heavy", "p", "l"),
        ll2_launch("h", "2024", "Starship", "p", "l"),
        ll2_launch("i", "2008", "Falcon 1", "p", "l"),
    ]},
}


def ll2_pages_router(url, params):
    if url == LAUNCH_PAGE2:
        return FakeResponse(LAUNCH_PAGES[2])
    return FakeResponse(LAUNCH_PAGES[1])


def test_launches_per_rocket(monkeypatch):
    calls = install_fake(monkeypatch, ll2_pages_router)
    get_launches_per_rocket = load_task(__file__, "4-rocket_frequency.py",
                                        "get_launches_per_rocket")
    assert calls == []
    result = [tuple(r) for r in get_launches_per_rocket()]
    assert result == [("Falcon 9", 3), ("Falcon 1", 2), ("Falcon Heavy", 2), ("Starship", 2)]
    assert len(calls) == 2


# ---------- slow: real network ----------

@pytest.mark.slow
def test_live_swapi_ships():
    availableShips = load_task(__file__, "0-passengers.py", "availableShips")
    ships = availableShips(4)
    assert "CR90 corvette" in ships and "Death Star" in ships
    assert "X-wing" not in ships


@pytest.mark.slow
def test_live_swapi_planets():
    sentientPlanets = load_task(__file__, "1-sentience.py", "sentientPlanets")
    planets = sentientPlanets()
    assert "Coruscant" in planets and "Kashyyyk" in planets


@pytest.mark.slow
def test_live_github():
    get_user_location = load_task(__file__, "2-user_location.py", "get_user_location")
    out = get_user_location("https://api.github.com/users/octocat")
    assert out == "San Francisco" or out.startswith("Reset in")
    out = get_user_location("https://api.github.com/users/this-user-does-not-exist-9z8y7x")
    assert out == "Not found" or out.startswith("Reset in")


@pytest.mark.slow
def test_live_launch_library():
    get_first_launch = load_task(__file__, "3-first_launch.py", "get_first_launch")
    get_launches_per_rocket = load_task(__file__, "4-rocket_frequency.py",
                                        "get_launches_per_rocket")
    try:
        first = get_first_launch()
        counts = get_launches_per_rocket()
    except Exception as exc:  # the free tier allows ~15 requests per hour
        if "429" in str(exc):
            pytest.skip("Launch Library 2 rate limit hit, try again in an hour")
        raise
    assert "FalconSAT-2" in first and "Falcon 1" in first
    assert counts[0][0] == "Falcon 9"
