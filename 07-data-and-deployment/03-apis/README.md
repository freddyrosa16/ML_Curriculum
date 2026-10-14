# Data Collection: APIs

**Status:** Original, with one swap: the unofficial SpaceX API used by tasks 3-4 has been offline since 2025 (it answers with Cloudflare errors), so those tasks now use the free [Launch Library 2](https://thespacedevs.com/llapi) API, which has SpaceX's full launch history. SWAPI (Star Wars API) and the GitHub API are unchanged and were checked working in October 2026. Tasks 2-4 now `return` their result (the script part prints it) so they can be tested.

## Why this matters
A lot of the data you'll want for ML isn't in a file: it sits behind a web API. Knowing how to request it, page through results, follow links between resources, handle errors and rate limits, and turn JSON into Python objects is a core data-engineering skill, and the same skills apply to calling model APIs.

## Concepts to understand first
- **What is a REST API?** A web server that exposes data at URLs ("endpoints"). You send an HTTP request (usually `GET` to read) and get back a response with a status code and a body, usually JSON.
- **What are HTTP status codes?** Three-digit results: `200` OK, `404` not found, `403` forbidden (GitHub uses it when you've hit the rate limit), `429` too many requests, `5xx` server errors. Always check the code before trusting the body.
- **What is JSON?** A text format for nested data: objects (like Python dicts), arrays (lists), strings, numbers, `true/false`, `null`. `requests` can parse it for you into Python objects.
- **What is pagination?** APIs return big collections in pages. The response includes the results for this page plus a `next` URL (or `null` on the last page). You keep requesting `next` until there isn't one.
- **What are query parameters?** The `?key=value&key2=value2` part of a URL, used for filters, sorting and page size. `requests` can build that part from a dict for you.
- **What is rate limiting?** Free APIs cap how many requests you can make: unauthenticated GitHub allows 60 per hour, Launch Library 2 about 15 per hour. Headers like `X-RateLimit-Reset` (a Unix timestamp) tell you when the limit resets. Cache results while you develop.
- **What are linked resources?** Responses often contain URLs pointing to related objects (a species' `homeworld` is a planet URL). You fetch those URLs to get the details.

## Resources
- [Requests: quickstart](https://requests.readthedocs.io/en/latest/user/quickstart/)
- [MDN: An overview of HTTP](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/Overview) · [HTTP response status codes](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Status)
- [SWAPI documentation](https://swapi.dev/documentation)
- [GitHub REST API: users](https://docs.github.com/en/rest/users/users) · [rate limits](https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api)
- [Launch Library 2 API docs (browsable)](https://ll.thespacedevs.com/docs/)
- [Real Python: Python's Requests library](https://realpython.com/python-requests/)

## Rules
- `import requests` (plus standard-library modules like `sys`, `time`, `collections`).
- Use `requests.get` for requests (the tests swap it out for a fake that serves canned responses).
- Scripts must not do anything when imported: put the script part under `if __name__ == "__main__":`.
- Follow the `next` links for pagination rather than guessing page numbers.

## Setup
`uv pip install requests`. No API keys needed.

While developing tasks 3-4, use Launch Library 2's development server `https://lldev.thespacedevs.com/2.3.0` (same format, not rate limited, but its data is incomplete and stale), and only switch to the real `https://ll.thespacedevs.com/2.3.0` when your code works. That's what the `base_url` parameter is for.

## Tasks

### 0. Can I join?
**File:** `0-passengers.py` · **Signature:** `availableShips(passengerCount)`

Using SWAPI's starships endpoint `https://swapi.dev/api/starships/`, return the names of all ships that can carry at least `passengerCount` passengers, in the order the API lists them. Handle pagination. The `passengers` field is a string: it may contain thousands separators ("843,342") or not be a number at all ("n/a", "unknown"), which you should skip. Return an empty list if no ship qualifies.
```
availableShips(4)
->  ['CR90 corvette', 'Sentinel-class landing craft', 'Death Star', 'Millennium Falcon', 'Executor', 'Rebel transport', ...]
```

### 1. Where I am?
**File:** `1-sentience.py` · **Signature:** `sentientPlanets()`

Using SWAPI's species endpoint `https://swapi.dev/api/species/`, return the names of the home planets of all **sentient** species, in API order. A species counts as sentient if the word "sentient" appears (in any case) in its `classification` or its `designation`. The `homeworld` field is a planet URL you must request to get the name; skip species whose homeworld is `null`. Handle pagination.
```
sentientPlanets()
->  ['Coruscant', 'Kashyyyk', 'Rodia', 'Nal Hutta', 'unknown', ...]
    (one planet SWAPI itself names "unknown"; keep it, it's a real planet record)
```

### 2. Rate me if you can!
**File:** `2-user_location.py` · **Signature:** `get_user_location(api_url)`, plus a script part

Given a full GitHub API user URL, return:
- the user's `location`, or `No location found` if it's empty;
- `Not found` if the user doesn't exist (status 404);
- `Reset in X min` if you're rate limited (status 403), where X is the number of whole minutes from now until the time in the `X-RateLimit-Reset` header.

Run as a script, it takes the URL as its first command-line argument and prints the result.
```
python 2-user_location.py https://api.github.com/users/octocat        ->  San Francisco
python 2-user_location.py https://api.github.com/users/no-such-user-9z  ->  Not found
(after 60 requests in an hour)                                            ->  Reset in 37 min
```

### 3. First launch
**File:** `3-first_launch.py` · **Signature:** `get_first_launch(base_url="https://ll.thespacedevs.com/2.3.0")`, plus a script part

Using Launch Library 2's `/launches/` endpoint, find SpaceX's **earliest** launch and return one line in this format:
`<launch name> (<net>) <rocket name> - <pad name> (<pad location name>)`
- Filter to SpaceX with the query parameter `lsp__name=SpaceX`. Ask the API to sort by date (`ordering=net`) so the first page has the earliest launches, but still pick the launch with the smallest `net` yourself from the results you get (the first one in API order wins a tie).
- `net` ("no earlier than") is the launch time as an ISO 8601 string; use it as is.
- The rocket name is in `rocket` → `configuration` → `name`; the pad name in `pad` → `name`; its location in `pad` → `location` → `name`.

Run as a script, print the line.
```
get_first_launch()
->  'Falcon 1 | FalconSAT-2 (2006-03-24T22:30:00Z) Falcon 1 - Omelek Island (Ronald Reagan Ballistic Missile Defense Test Site, Kwajalein Atoll, Marshall Islands)'
```

### 4. How many by rocket?
**File:** `4-rocket_frequency.py` · **Signature:** `get_launches_per_rocket(base_url="https://ll.thespacedevs.com/2.3.0")`, plus a script part

Count all SpaceX launches (past and upcoming, filter `lsp__name=SpaceX`) per rocket name (`rocket` → `configuration` → `name`). Return a list of `(rocket name, count)` pairs sorted by count, highest first; ties go in alphabetical order. Follow the `next` links until you've seen every launch, and ask for 100 results per page (`limit=100`) so the whole history takes about 9 requests instead of 90 (remember the ~15/hour limit).

Run as a script, print one `<rocket name>: <count>` line per rocket.
```
python 4-rocket_frequency.py
->  Falcon 9: ...      (several hundred, by far the most)
    ...               (Starship, Falcon Heavy, Falcon 1)
```

## Check your work
From the `ml-curriculum` folder:
```bash
pytest 07-data-and-deployment/03-apis -m "not slow"   # offline, with fake responses
pytest 07-data-and-deployment/03-apis                 # also calls the real APIs (~12 requests to Launch Library 2)
```
