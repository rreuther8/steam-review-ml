"""Minimal IGDB v4 HTTP client (Twitch OAuth)."""

from __future__ import annotations

import time
from typing import Any

import requests


class IGDBClient:
    TOKEN_URL = "https://id.twitch.tv/oauth2/token"
    API_BASE = "https://api.igdb.com/v4"

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        *,
        request_min_interval_s: float = 0.26,
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.request_min_interval_s = request_min_interval_s
        self._access_token: str | None = None
        self._token_expires_at = 0.0
        self._last_request_at = 0.0

    def _refresh_token(self) -> None:
        resp = requests.post(
            self.TOKEN_URL,
            params={
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "grant_type": "client_credentials",
            },
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        self._access_token = str(payload["access_token"])
        expires_in = int(payload.get("expires_in", 3600))
        self._token_expires_at = time.time() + max(60, expires_in - 120)

    def _ensure_token(self) -> str:
        if self._access_token is None or time.time() >= self._token_expires_at:
            self._refresh_token()
        assert self._access_token is not None
        return self._access_token

    def post(self, endpoint: str, body: str) -> list[dict[str, Any]]:
        elapsed = time.time() - self._last_request_at
        if elapsed < self.request_min_interval_s:
            time.sleep(self.request_min_interval_s - elapsed)

        token = self._ensure_token()
        resp = requests.post(
            f"{self.API_BASE}/{endpoint.lstrip('/')}",
            headers={
                "Client-ID": self.client_id,
                "Authorization": f"Bearer {token}",
            },
            data=body,
            timeout=60,
        )
        self._last_request_at = time.time()
        if resp.status_code == 401:
            self._refresh_token()
            return self.post(endpoint, body)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            raise ValueError(f"Unexpected IGDB response type for {endpoint}: {type(data)}")
        return data
