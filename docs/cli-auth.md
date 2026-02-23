---
title: "CLI Auth"
description: "Log in to Macrodata from the CLI using an API key"
---

Use the `macrodata` CLI to store and validate a Macrodata API key for local tooling and future launcher integrations.

## Commands

### `macrodata login`

Prompts for an `ing_...` API key, validates it via `GET /api/me`, and stores it locally.

```bash
macrodata login
```

Non-interactive options:

```bash
macrodata login --token ing_xxx
printf '%s' 'ing_xxx' | macrodata login --token-stdin
```

The CLI prints your authenticated identity (`name` / `username` / `email`) after successful validation.

### `macrodata whoami`

Verifies the stored key and shows the current authenticated identity:

```bash
macrodata whoami
```

### `macrodata logout`

Removes the locally stored API key:

```bash
macrodata logout
```

## Configuration

- Base URL defaults to the Macrodata platform control plane.
- Set `MACRODATA_BASE_URL` to override it temporarily (for dev/staging).
- `macrodata whoami` uses `MACRODATA_API_KEY` if set; otherwise it reads the local key file.
- Credentials are stored in XDG config (`~/.config/macrodata/api_key` on Linux if `XDG_CONFIG_HOME` is unset).

## Internal Notes

- `macrodata login` is API-key-only in v1 (no email/password flow in the CLI).
- Validation uses the platform `GET /api/me` endpoint and expects API key metadata plus nested user identity fields.
