# Hints: Data Collection with APIs

Open one hint at a time. Try for a while before opening the next. Open each endpoint in your browser first: seeing the raw JSON makes every task easier.

## 0. Can I join?
<details><summary>Hint 1</summary>
Look at one page of the starships endpoint in the browser. The response is an object with a `results` list and a `next` URL. Your loop runs "while there is a URL to fetch".
</details>
<details><summary>Hint 2</summary>
Each iteration: request the current URL, parse the JSON, look at every ship in `results`, then replace the current URL with `next` (which becomes `None` on the last page, ending the loop).
</details>
<details><summary>Hint 3</summary>
To turn "843,342" into a number, remove the commas first. Strings have a method that tells you whether they consist only of digits; anything else ("n/a", "unknown") is skipped.
</details>

## 1. Where I am?
<details><summary>Hint 1</summary>
Same pagination loop as task 0, over species instead of starships.
</details>
<details><summary>Hint 2</summary>
Lowercase both fields before checking whether "sentient" is in them. The homeworld is a URL: request it and read the planet's `name` from that response.
</details>

## 2. Rate me if you can!
<details><summary>Hint 1</summary>
Branch on the response's status code before touching the body.
</details>
<details><summary>Hint 2</summary>
Response headers behave like a dictionary. The reset header is a Unix timestamp (seconds) as a string; the current Unix time comes from the `time` module. Whole minutes is integer division by 60.
</details>
<details><summary>Hint 3</summary>
The command-line arguments are in `sys.argv`; the first real argument is at index 1.
</details>

## 3. First launch
<details><summary>Hint 1</summary>
Try the URL in your browser with the dev server first, adding `?lsp__name=SpaceX&ordering=net&limit=5` to the launches endpoint. Find where each piece of the output line lives in the nested JSON.
</details>
<details><summary>Hint 2</summary>
`requests.get` accepts a `params` dict and builds the query string for you. Python's built-in `min` takes a `key` function, and on equal keys returns the first one it saw, which is exactly the tie rule.
</details>
<details><summary>Hint 3</summary>
ISO 8601 timestamps in the same format and time zone sort correctly as plain strings, so you can compare `net` values directly. Build the line with an f-string.
</details>

## 4. How many by rocket?
<details><summary>Hint 1</summary>
Pagination again, plus counting. The `collections` module has a dict subclass made for counting things.
</details>
<details><summary>Hint 2</summary>
The `next` URL already includes all the query parameters, so only pass your `params` dict on the first request.
</details>
<details><summary>Hint 3</summary>
For "count descending, then name ascending", sort with a key that returns a tuple: the negated count first, then the name.
</details>
