# Wikipedia Nexus — AI Navigation Engine

**Wikipedia Nexus** is a Python-based Wikipedia navigation experiment that uses semantic similarity and beam search to find a sequence of linked Wikipedia articles from a starting article to a destination article.

Instead of choosing links randomly, it uses a Sentence Transformers model to rank candidate article titles by how semantically similar they are to the destination. It then explores a limited number of promising routes through Wikipedia's article links.

> **Important:** Wikipedia Nexus is a heuristic route finder, not a guaranteed shortest-path solver. It may fail to find a route even when one exists, and a route it finds is not guaranteed to be the shortest possible route.

---

## Features

- **AI-assisted link ranking:** Uses `sentence-transformers/all-MiniLM-L6-v2` to compare article-title embeddings.
- **Beam search:** Keeps a limited number of promising routes instead of exploring every possible path.
- **Wikipedia API integration:** Resolves article titles and retrieves links through the MediaWiki API.
- **Redirect handling:** Resolves starting and destination titles, including Wikipedia redirects.
- **Link caching:** Reuses links already fetched during the current run.
- **Request handling:** Uses sequential requests, a minimum request interval, timeouts, and retries for temporary network or API errors.
- **Performance report:** Displays whether a route was found, elapsed time, click count, expanded links, the final route, and article URLs.

## How it works

1. **Choose the challenge.** Enter a starting Wikipedia article and a destination article.
2. **Resolve article titles.** Wikipedia's API checks the titles and follows redirects.
3. **Encode the destination.** The destination title is converted into a normalized semantic embedding.
4. **Retrieve outgoing links.** For each article being explored, the program requests its links to other main-namespace Wikipedia articles.
5. **Filter visited articles.** Articles already present in the current route are removed to avoid loops within that route.
6. **Rank candidate links.** The semantic engine encodes the remaining article titles and scores them against the destination embedding using cosine similarity.
7. **Keep promising routes.** Beam search retains only the highest-ranked routes for the next search depth.
8. **Check for the destination.** If the destination is directly linked from an article being explored, the program reports the route. Otherwise, it continues until it reaches the configured depth or runs out of candidate routes.

### Search strategy

The current default settings are:

| Setting | Default | Meaning |
|---|---:|---|
| `BEAM_WIDTH` | `5` | Number of candidate routes retained at each search depth. |
| `MAX_DEPTH` | `6` | Maximum number of search levels explored. |
| `TOP_CANDIDATES` | `25` | Maximum number of semantically ranked links retained for each explored article. |
| `MAX_LINK_BATCHES` | `1` | Maximum number of API link batches requested for each article. |

The semantic score is calculated from normalized embeddings. Because the embeddings are normalized, their dot product is equivalent to cosine similarity.

---

## Requirements

- Python
- An internet connection
- The Python packages listed below

### Python packages

- `requests`
- `numpy`
- `sentence-transformers`

Install them with:

```bash
pip install requests numpy sentence-transformers
```

`sentence-transformers` also installs its required machine-learning dependencies. The first run may take longer because the model `sentence-transformers/all-MiniLM-L6-v2` needs to be downloaded and loaded.

---

## Installation and setup

1. Save the Python code in a file, for example:

   ```text
   wikipedia_nexus.py
   ```

2. Open a terminal in the directory containing the file.

3. Install the dependencies:

   ```bash
   pip install requests numpy sentence-transformers
   ```

4. In `WikipediaClient`, replace the `YOUR_CONTACT` placeholder in the `User-Agent` value with a suitable contact email address or project website.

5. Run the program:

   ```bash
   python wikipedia_nexus.py
   ```

6. Enter the starting and destination article titles when prompted.

### Example

```text
Starting article: Potato
Destination article: Barack Obama
```

The program will then begin searching for a route through linked Wikipedia articles.

---

## Configuration

The main configuration values are defined near the beginning of the Python file:

| Variable | Default | Description |
|---|---|---|
| `API_URL` | `https://en.wikipedia.org/w/api.php` | Wikipedia's MediaWiki API endpoint. |
| `MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence Transformer model used for semantic ranking. |
| `BEAM_WIDTH` | `5` | Number of routes retained at each depth. |
| `MAX_DEPTH` | `6` | Maximum search depth. |
| `MAX_LINK_BATCHES` | `1` | Number of link API batches requested per article. |
| `TOP_CANDIDATES` | `25` | Number of top-scoring links retained per article. |
| `REQUEST_DELAY` | `0.5` | Defined in the configuration, but not currently used by the request logic. |

The `WikipediaClient` also sets `minimum_request_interval` to `1.0` second between requests. API requests use a 20-second timeout and can be retried up to three times for temporary failures.

Changing these values affects the trade-off between search coverage, runtime, and memory use. Increasing the beam width, depth, or candidate count can explore more possibilities, but may also increase the number of API calls and the amount of computation.

---

## Understanding the output

The performance report includes:

- **Success:** Whether the program found a route within its search limits.
- **Elapsed time:** Time spent in the navigation search.
- **Clicks:** Number of article-to-article transitions in the reported route.
- **Expanded links:** Number of unvisited outgoing links considered during the search.
- **Final route:** The ordered list of Wikipedia article titles, when successful.
- **Article URLs:** Wikipedia URLs for the articles in the final route.

A failed search means that this run did not find a route within the configured limits. It does **not** prove that no valid Wikipedia route exists.

---

## Current limitations

- **No shortest-route guarantee:** Beam search is heuristic and can discard a route that would eventually lead to a shorter solution.
- **Title-based semantic ranking:** The model compares article titles with the destination title. It does not compare full article text, page contents, or link context.
- **Limited candidate selection:** Only the top `TOP_CANDIDATES` links from each explored article are kept for route expansion.
- **Limited link retrieval:** `MAX_LINK_BATCHES` is currently `1`, so the program may not retrieve every outgoing link when Wikipedia returns continuation data.
- **Limited search depth and width:** The default depth and beam width restrict how much of Wikipedia's link graph can be explored.
- **Network dependency:** Article resolution and link retrieval require access to Wikipedia's API.
- **No persistent cache:** The link cache exists only while the program is running.

These limitations are important when interpreting the result: a fast route is not necessarily the shortest route, and failure to find a route is not proof that no route exists.

---

## Project structure

The current implementation is a single Python script organized into these main sections:

| Component | Responsibility |
|---|---|
| Configuration | API endpoint, model name, and search parameters. |
| Initialization | Loads the Sentence Transformer model. |
| `WikipediaClient` | Communicates with Wikipedia, resolves titles, retrieves links, caches results, and builds article URLs. |
| `SemanticEngine` | Encodes the destination and ranks candidate article titles. |
| `WikipediaNavigator` | Performs the beam search and constructs routes. |
| `display_report()` | Prints the search results and performance information. |
| Main program | Creates the components, accepts user input, and starts the search. |

---

## Technologies used

- **Python** — application logic
- **Wikipedia MediaWiki API** — article title resolution and link retrieval
- **Sentence Transformers** — semantic embeddings
- **all-MiniLM-L6-v2** — sentence embedding model
- **NumPy** — vector operations and similarity ranking
- **Requests** — HTTP communication with Wikipedia

---

## Disclaimer

This is an educational and experimental project. It uses Wikipedia's publicly accessible API to retrieve article information. Please use the service responsibly and follow Wikipedia's API usage guidance. This project is not affiliated with or endorsed by the Wikimedia Foundation.
