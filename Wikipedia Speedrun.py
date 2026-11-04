import time
import requests
import numpy as np
from urllib.parse import quote
from sentence_transformers import SentenceTransformer
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

# ============================================================
# 1. CONFIGURATION
# ============================================================

API_URL = "https://en.wikipedia.org/w/api.php"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BEAM_WIDTH = 5
MAX_DEPTH = 6
# Maximum number of API link batches per article.
MAX_LINK_BATCHES = 1
# Number of most promising links retained per article.
TOP_CANDIDATES = 25
REQUEST_DELAY = 0.5

# ============================================================
# 2. INITIALIZATION
# ============================================================
print("\n" + "=" * 60)
print("             WIKIPEDIA NEXUS")
print("          AI NAVIGATION ENGINE")
print("=" * 60)
print("\nLoading semantic intelligence engine...")
model = SentenceTransformer(MODEL_NAME)
print("AI model successfully loaded!")

# ============================================================
# 3. WIKIPEDIA API CLIENT
# ============================================================
class WikipediaClient:
    def __init__(self):
        self.session = requests.Session()
        # Replace the contact information with your own
        # project website or a suitable contact email.
        self.session.headers.update({
            "User-Agent": (
                "WikipediaNexus/1.1 "
                "(educational project; contact: YOUR_CONTACT)"
            )
        })
        self.link_cache = {}
        # Keep requests sequential and spaced apart.
        self.minimum_request_interval = 1.0
        self.last_request_time = 0.0

    def request(self, params):
        params = params.copy()
        params["format"] = "json"
        params["maxlag"] = 5
        max_attempts = 3
        for attempt in range(max_attempts):
            # Enforce a minimum interval between requests.
            elapsed = (
                time.monotonic() - self.last_request_time
            )
            if elapsed < self.minimum_request_interval:
                time.sleep(
                    self.minimum_request_interval - elapsed
                )

            try:
                response = self.session.get(
                    API_URL,
                    params=params,
                    timeout=20
                )
                self.last_request_time = time.monotonic()
                # Handle temporary rate limits and server overload.
                if response.status_code in (429, 503):
                    retry_after = response.headers.get(
                        "Retry-After"
                    )
                    if retry_after:
                        try:
                            wait_time = float(retry_after)
                        except ValueError:
                            try:
                                retry_date = (
                                    parsedate_to_datetime(retry_after)
                                )
                                wait_time = max(
                                    0,
                                    (
                                        retry_date
                                        - datetime.now(timezone.utc)
                                    ).total_seconds()
                                )
                            except (ValueError, TypeError):

                                wait_time = 5 * (2 ** attempt)
                    else:
                        # Exponential backoff.
                        wait_time = 5 * (2 ** attempt)
                    if attempt == max_attempts - 1:
                        response.raise_for_status()
                    print(
                        f"\n[WIKIPEDIA API] "
                        f"HTTP {response.status_code}."
                    )
                    print(
                        f"Waiting {wait_time:.1f} seconds "
                        f"before retrying..."
                    )
                    response.close()
                    time.sleep(wait_time)
                    continue
                response.raise_for_status()
                data = response.json()
                if "error" in data:
                    raise RuntimeError(
                        f"Wikipedia API error: {data['error']}"
                    )
                return data
            except requests.exceptions.RequestException:
                if attempt == max_attempts - 1:
                    raise
                wait_time = 5 * (2 ** attempt)
                print(
                    f"\nNetwork error. "
                    f"Retrying in {wait_time} seconds..."
                )
                time.sleep(wait_time)
        raise RuntimeError(
            "Wikipedia request failed after all retries."
        )

    def resolve_title(self, title):
        data = self.request({
            "action": "query",
            "titles": title,
            "redirects": 1
        })
        pages = data["query"]["pages"]
        page = next(iter(pages.values()))
        if "missing" in page:
            raise ValueError(
                f"Wikipedia article not found: {title}"
            )
        return page["title"]

    def get_url(self, title):
        encoded_title = quote(
            title.replace(" ", "_"),
            safe="()_,-"
        )
        return (
            "https://en.wikipedia.org/wiki/"
            + encoded_title
        )

    def get_links(self, title):
        if title in self.link_cache:
            print(f"   Using cached links: {title}")
            return self.link_cache[title]
        print(f"   Fetching links: {title}")
        all_links = []
        continuation = None
        for batch in range(MAX_LINK_BATCHES):
            params = {
                "action": "query",
                "prop": "links",
                "titles": title,
                "plnamespace": 0,
                "pllimit": "max"
            }
            if continuation:
                params.update(continuation)
            data = self.request(params)
            pages = data.get(
                "query", {}
            ).get("pages", {})
            for page in pages.values():
                page_links = page.get("links", [])
                for link in page_links:
                    all_links.append(link["title"])
            print(
                f"   Batch {batch + 1}: "
                f"{len(page_links)} links"
            )
            continuation = data.get("continue")
            if not continuation:
                break
        all_links = list(dict.fromkeys(all_links))
        self.link_cache[title] = all_links
        print(
            f"   Total unique links: {len(all_links)}"
        )
        return all_links


# ============================================================
# 4. SEMANTIC INTELLIGENCE ENGINE
# ============================================================

class SemanticEngine:
    def __init__(self, embedding_model):
        self.model = embedding_model

    def encode_target(self, target):
        return self.model.encode(
            target,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

    def rank_candidates(
        self,
        candidates,
        target_embedding
    ):
        if not candidates:
            return []
        embeddings = self.model.encode(
            candidates,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=64,
            show_progress_bar=False
        )
        # Since embeddings are normalized,
        # dot product gives cosine similarity.
        scores = embeddings @ target_embedding
        ranked_indices = np.argsort(scores)[::-1]
        ranked = []
        for index in ranked_indices[:TOP_CANDIDATES]:
            ranked.append((
                candidates[index],
                float(scores[index])
            ))
        return ranked

# ============================================================
# 5. BEAM SEARCH NAVIGATION ENGINE
# ============================================================

class WikipediaNavigator:
    def __init__(self, client, intelligence):
        self.client = client
        self.intelligence = intelligence

    def search(self, start, target):
        start = self.client.resolve_title(start)
        target = self.client.resolve_title(target)
        print("\n" + "=" * 60)
        print("                 SPEEDRUN STARTED")
        print("=" * 60)
        print(f"\nSTART: {start}")
        print(f"TARGET: {target}")
        start_time = time.perf_counter()
        target_embedding = (
            self.intelligence.encode_target(target)
        )
        if start.casefold() == target.casefold():
            return {
                "success": True,
                "path": [start],
                "steps": 0,
                "elapsed": time.perf_counter() - start_time
            }
        # Each beam contains a complete possible route.
        beam = [[start]]
        # Avoid revisiting the same article at an
        # equal or greater depth.
        best_depth = {start: 0}
        total_expansions = 0
        for depth in range(MAX_DEPTH):
            print(f"\n--- SEARCH DEPTH {depth + 1} ---")
            next_paths = {}
            for path in beam:
                current = path[-1]
                print(f"\nExploring: {current}")
                links = self.client.get_links(current)
                # Remove articles already in this route.
                unvisited = [
                    link for link in links
                    if link.casefold() not in {
                        page.casefold() for page in path
                    }
                ]
                if not unvisited:
                    continue
                # Check whether the target is directly linked.
                target_link = next(
                    (
                        link for link in unvisited
                        if link.casefold() == target.casefold()
                    ),
                    None
                )
                if target_link:
                    final_path = path + [target_link]
                    elapsed = (
                        time.perf_counter() - start_time
                    )
                    print("\n" + "=" * 60)
                    print("              TARGET REACHED!")
                    print("=" * 60)
                    for number, page in enumerate(
                        final_path, 1
                    ):
                        print(f"{number}. {page}")
                    print(
                        f"\nTotal clicks: {len(final_path) - 1}"
                    )
                    print(f"Time: {elapsed:.2f} seconds")
                    return {
                        "success": True,
                        "path": final_path,
                        "steps": len(final_path) - 1,
                        "elapsed": elapsed,
                        "expansions": total_expansions
                    }
                # Rank candidate links by semantic similarity.
                ranked = self.intelligence.rank_candidates(
                    unvisited,
                    target_embedding
                )
                total_expansions += len(unvisited)
                for title, score in ranked:
                    new_depth = depth + 1
                    previous_depth = best_depth.get(
                        title, float("inf")
                    )
                    if new_depth > previous_depth:
                        continue
                    best_depth[title] = new_depth
                    new_path = path + [title]
                    # Retain the strongest route to each page.
                    if (
                        title not in next_paths
                        or score > next_paths[title][0]
                    ):
                        next_paths[title] = (
                            score,
                            new_path
                        )
            if not next_paths:
                print("\nNo promising routes remain.")
                break
            # Select the strongest candidate routes.
            sorted_paths = sorted(
                next_paths.values(),
                key=lambda item: item[0],
                reverse=True
            )
            beam = [
                path for score, path
                in sorted_paths[:BEAM_WIDTH]
            ]
            print("\nCurrent best routes:")
            for index, path in enumerate(beam, 1):
                print(
                    f"{index}. {' -> '.join(path)}"
                )
        elapsed = time.perf_counter() - start_time
        print("\n" + "=" * 60)
        print("             SEARCH TERMINATED")
        print("=" * 60)
        print("No route found within the search limits.")
        return {
            "success": False,
            "path": None,
            "steps": 0,
            "elapsed": elapsed,
            "expansions": total_expansions
        }

# ============================================================
# 6. PERFORMANCE REPORT
# ============================================================

def display_report(result):
    print("\n" + "=" * 60)
    print("               PERFORMANCE REPORT")
    print("=" * 60)
    print(
        f"Success: {'YES' if result['success'] else 'NO'}"
    )
    print(f"Elapsed time: {result['elapsed']:.2f}s")
    print(f"Clicks: {result['steps']}")
    print(
        f"Expanded links: {result.get('expansions', 0)}"
    )
    if result["success"]:
        print("\nFINAL ROUTE:")
        for index, page in enumerate(
            result["path"], 1
        ):
            print(
                f"{index}. {page}"
            )
            if index < len(result["path"]):
                print(
                    "   |"
                )
                print(
                    "   v"
                )
        print("\nARTICLE URLS:")
        for page in result["path"]:
            print(
                WikipediaClient().get_url(page)
            )

# ============================================================
# 7. MAIN PROGRAM
# ============================================================
if __name__ == "__main__":
    wikipedia = WikipediaClient()
    intelligence = SemanticEngine(model)
    navigator = WikipediaNavigator(
        wikipedia,
        intelligence
    )
    print("\nEnter your speedrun challenge.")
    start_article = input(
        "\nStarting article: "
    ).strip()
    target_article = input(
        "Destination article: "
    ).strip()
    try:
        result = navigator.search(
            start_article,
            target_article
        )
        display_report(result)
    except (
        requests.exceptions.RequestException,
        ValueError,
        RuntimeError
    ) as error:
        print(f"\nERROR: {error}")