import json
import os
import time
from pathlib import Path

import httpx

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "catalog.json"
Card = dict  # {"id", "name", "category", "tags"}


class LocalCatalog:
    """Static JSON catalog (data/catalog.json)."""

    def __init__(self, path: str | Path = DEFAULT_PATH):
        self.name = f"local:{Path(path).name}"
        self._cards = json.loads(Path(path).read_text())

    async def load(self) -> list[Card]:
        return self._cards


class ReloadlyCatalog:
    """Reloadly Gift Cards API (sandbox by default). Normalizes products to the Card shape
    and caches the catalog in-process."""

    AUTH_URL = "https://auth.reloadly.com/oauth/token"

    def __init__(self, client_id, client_secret, sandbox=True, country="US", ttl_seconds=3600, http=None):
        self.client_id, self.client_secret, self.country, self.ttl = client_id, client_secret, country, ttl_seconds
        self.base = "https://giftcards-sandbox.reloadly.com" if sandbox else "https://giftcards.reloadly.com"
        self.name = f"reloadly:{'sandbox' if sandbox else 'live'}:{country}"
        self.http = http or httpx.AsyncClient(timeout=30)
        self._cards: list[Card] | None = None
        self._loaded_at = 0.0

    async def _token(self) -> str:
        res = await self.http.post(
            self.AUTH_URL,
            json={"client_id": self.client_id, "client_secret": self.client_secret,
                  "grant_type": "client_credentials", "audience": self.base},
        )
        res.raise_for_status()
        return res.json()["access_token"]

    async def load(self) -> list[Card]:
        if self._cards is not None and time.monotonic() - self._loaded_at < self.ttl:
            return self._cards
        headers = {"Authorization": f"Bearer {await self._token()}",
                   "Accept": "application/com.reloadly.giftcards-v1+json"}
        products, page = [], 1
        while True:
            res = await self.http.get(f"{self.base}/countries/{self.country}/products", headers=headers,
                                      params={"size": 200, "page": page})
            res.raise_for_status()
            data = res.json()
            products += data if isinstance(data, list) else data.get("content", [])
            if isinstance(data, list) or page >= data.get("totalPages", 1):
                break
            page += 1
        self._cards, self._loaded_at = normalize_reloadly(products), time.monotonic()
        return self._cards


def normalize_reloadly(products: list[dict]) -> list[Card]:
    """One card per brand; Reloadly lacks tags, so product names are used for matching."""
    cards: dict[str, Card] = {}
    for p in products:
        brand = (p.get("brand") or {}).get("brandName") or p.get("productName")
        if not brand:
            continue
        category = (p.get("category") or {}).get("name") or "general"
        product_name = (p.get("productName") or "").lower()
        cards.setdefault(brand.lower(), {
            "id": str(p["productId"]),
            "name": brand,
            "category": category.lower(),
            "tags": [product_name] if product_name and product_name != brand.lower() else [],
        })
    return list(cards.values())


def catalog_from_env():
    if os.getenv("RELOADLY_CLIENT_ID") and os.getenv("RELOADLY_CLIENT_SECRET"):
        return ReloadlyCatalog(
            os.environ["RELOADLY_CLIENT_ID"], os.environ["RELOADLY_CLIENT_SECRET"],
            sandbox=os.getenv("RELOADLY_SANDBOX", "true").lower() != "false",
            country=os.getenv("RELOADLY_COUNTRY", "US"),
        )
    return LocalCatalog(os.getenv("CATALOG_PATH", str(DEFAULT_PATH)))
