from __future__ import annotations

from refiner.platform.client.api import MacrodataClient, sanitize_terminal_text


def build_job_tracking_url(
    *, client: MacrodataClient, job_id: str, workspace_slug: str | None = None
) -> str:
    safe_base_url = sanitize_terminal_text(client.base_url).strip().rstrip("/")
    safe_job_id = sanitize_terminal_text(job_id).strip() or job_id
    safe_workspace_slug = (
        sanitize_terminal_text(workspace_slug).strip() if workspace_slug else None
    )
    if safe_workspace_slug:
        return f"{safe_base_url}/jobs/{safe_workspace_slug}/{safe_job_id}"
    return f"{safe_base_url}/jobs/{safe_job_id}"
