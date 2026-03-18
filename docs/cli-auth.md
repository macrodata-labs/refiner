---
title: "CLI Auth"
description: "Authenticate Refiner against Macrodata with an API key"
---

Use the `macrodata` CLI to store and validate a Macrodata API key.

Default API key page:

- https://macrodata.co/settings/api-keys

If you set `MACRODATA_BASE_URL` to point at another deployment, the CLI uses that base URL instead.

## Login

Interactive login:

```bash
macrodata login
```

Non-interactive login:

```bash
macrodata login --token md_xxx
printf '%s' 'md_xxx' | macrodata login --token-stdin
```

The CLI validates the key via `GET /api/me`, stores it locally, and prints:

- authenticated user identity
- API key name
- workspace name / slug when present

## Check Current Auth

```bash
macrodata whoami
```

`whoami` verifies the current key and shows the same identity summary without changing local state.

## Logout

```bash
macrodata logout
```

This removes the locally stored API key.

## Where Credentials Come From

Lookup order:

1. `MACRODATA_API_KEY`
2. local credential file created by `macrodata login`

Stored credentials live in XDG config:

- Linux default: `~/.config/macrodata/api_key`

## Notes

- Refiner expects `md_...` API keys, not a separate username/password flow.
- `launch_local(...)` can use the stored key for platform lifecycle reporting.
- `launch_cloud(...)` requires Macrodata auth and uses the same key lookup path.
