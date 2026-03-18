---
title: "CLI"
description: "Macrodata CLI commands used with Refiner"
---

Refiner installs the `macrodata` CLI.

## Auth

Create a Macrodata API key:

- https://macrodata.co/settings/api-keys

### Login

```bash
macrodata login
```

Non-interactive options:

```bash
macrodata login --token md_xxx
printf '%s' 'md_xxx' | macrodata login --token-stdin
```

### Check Current Auth

```bash
macrodata whoami
```

### Logout

```bash
macrodata logout
```

## Credential Lookup

Lookup order:

1. `MACRODATA_API_KEY`
2. local credential file created by `macrodata login`

Stored credentials live in XDG config:

- Linux default: `~/.config/macrodata/api_key`

## Notes

- `launch_local(...)` can use the stored key for platform lifecycle reporting
- `launch_cloud(...)` requires Macrodata auth
