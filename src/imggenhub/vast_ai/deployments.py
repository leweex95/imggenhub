"""High-level helpers for Vast.ai deployment management using documented endpoints."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import requests


class VastApiError(RuntimeError):
    """Raised when a Vast.ai API call fails."""


class VastDeploymentClient:
    """Client for managing Vast.ai deployments via the public REST and serverless APIs."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://console.vast.ai/api/v0/",
        route_url: str = "https://run.vast.ai/route/",
        timeout: int = 30,
    ) -> None:
        if not api_key or not isinstance(api_key, str):
            raise ValueError("api_key must be a non-empty string")
        if timeout <= 0:
            raise ValueError("timeout must be a positive integer")

        self.api_key = api_key.strip()
        self.base_url = self._ensure_trailing_slash(base_url)
        self.route_url = self._ensure_trailing_slash(route_url)
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    @staticmethod
    def _ensure_trailing_slash(url: str) -> str:
        cleaned = url.strip()
        if not cleaned:
            raise ValueError("URL cannot be empty")
        return cleaned if cleaned.endswith("/") else f"{cleaned}/"

    def list_deployments(self) -> List[Dict[str, Any]]:
        """Return every active deployment for the authenticated account."""
        url = urljoin(self.base_url, "deployments/")
        data = self._request("GET", url)
        deployments = data.get("deployments") if isinstance(data, dict) else None
        if deployments is None:
            if isinstance(data, list):
                return data
            return []
        return deployments

    def create_deployment(self, deployment_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Create a deployment using the exact spec expected by Vast.ai."""
        if not isinstance(deployment_spec, dict) or not deployment_spec:
            raise ValueError("deployment_spec must be a non-empty dictionary")

        url = urljoin(self.base_url, "deployments/")
        return self._request("POST", url, json=deployment_spec)

    def get_deployment_status(self, deployment_id: str | int) -> Dict[str, Any]:
        """Fetch the latest status for a deployment."""
        if isinstance(deployment_id, str) and not deployment_id.strip():
            raise ValueError("deployment_id cannot be an empty string")
        if isinstance(deployment_id, int) and deployment_id <= 0:
            raise ValueError("deployment_id must be positive")

        deployment_id_str = str(deployment_id).strip()
        url = urljoin(self.base_url, f"deployments/{deployment_id_str}/")
        return self._request("GET", url)

    def invoke_route(
        self,
        route: str,
        payload: Optional[Dict[str, Any]] = None,
        method: str = "POST",
    ) -> Dict[str, Any]:
        """Call a serverless route exposed under run.vast.ai/route/."""
        if not route or not isinstance(route, str):
            raise ValueError("route must be a non-empty string")
        verb = method.upper().strip()
        if verb not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
            raise ValueError("Unsupported HTTP method for invoke_route")

        url = urljoin(self.route_url, route.lstrip("/"))
        kwargs: Dict[str, Any] = {}
        if payload is not None:
            if not isinstance(payload, dict):
                raise ValueError("payload must be a dictionary")
            kwargs["json"] = payload

        return self._request(verb, url, include_auth=True, **kwargs)

    def _request(
        self,
        method: str,
        url: str,
        *,
        include_auth: bool = True,
        **kwargs: Any,
    ) -> Any:
        try:
            if include_auth:
                response = self.session.request(method, url, timeout=self.timeout, **kwargs)
            else:
                response = requests.request(method, url, timeout=self.timeout, **kwargs)
            response.raise_for_status()
        except requests.HTTPError as exc:
            message = self._format_http_error(exc.response)
            raise VastApiError(message) from exc
        except requests.RequestException as exc:
            raise VastApiError(f"Request to {url} failed: {exc}") from exc

        if response.content:
            try:
                return response.json()
            except ValueError as exc:
                raise VastApiError("Received malformed JSON from Vast.ai") from exc
        return {}

    @staticmethod
    def _format_http_error(response: Optional[requests.Response]) -> str:
        if response is None:
            return "Vast.ai request failed and no response object was returned"

        try:
            payload = response.json()
            detail = payload.get("error") or payload.get("detail") or payload
        except ValueError:
            detail = response.text or "Unknown error"
        return (
            "Vast.ai request failed with status "
            f"{response.status_code}: {detail}"
        )


API_KEY = "<YOUR_KEY>"


def _example_usage() -> None:
    """Demonstrates listing, creating, and checking deployments."""
    client = VastDeploymentClient(api_key=API_KEY)

    # a) List current deployments
    try:
        deployments = client.list_deployments()
        if not deployments:
            print("No deployments found for this account.")
        else:
            for deployment in deployments:
                name = deployment.get("name", deployment.get("id", "unknown"))
                status = deployment.get("status")
                print(f"- {name}: {status}")
    except VastApiError as error:
        print(f"Failed to list deployments: {error}")
        return

    # b) Create a new deployment
    deployment_spec = {
        "name": "imggenhub-demo",
        "image": "nvidia/cuda:12.2.0-base-ubuntu22.04",
        "command": ["python", "-c", "print('hello from Vast.ai deployment')"],
        "resources": {
            "gpu_count": 1,
            "gpu_type": "A100",
            "cpu_count": 4,
            "memory_gb": 16,
        },
        "env": {
            "EXAMPLE_FLAG": "1",
        },
    }

    try:
        created = client.create_deployment(deployment_spec)
        deployment_id = created.get("id") or created.get("deployment_id")
        print(f"Created deployment: {deployment_id}")
    except VastApiError as error:
        print(f"Failed to create deployment: {error}")
        return

    # c) Query the newly created deployment
    if deployment_id is None:
        print("Server did not return a deployment identifier; cannot query status.")
        return

    try:
        status = client.get_deployment_status(deployment_id)
        print(f"Deployment status: {status.get('status', 'unknown')}")
    except VastApiError as error:
        print(f"Failed to query deployment status: {error}")


if __name__ == "__main__":
    _example_usage()
