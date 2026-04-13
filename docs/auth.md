---
title: "Auth"
description: "Macrodata authentication used by Refiner and the CLI"
---

Refiner uses the same Macrodata credential lookup order as the `macrodata` CLI.

## Credential Lookup

Lookup order:

1. `MACRODATA_API_KEY`
2. local credential file created by `macrodata login`

Stored credentials live in XDG config:

- Linux default: `~/.config/macrodata/api_key`

## Behavior

- `launch_local(...)` can run without Macrodata auth
- `launch_cloud(...)` requires Macrodata auth
- `macrodata whoami` uses the same lookup order

## Related Pages

- [CLI](cli.md)
- [Launchers](launchers.md)
