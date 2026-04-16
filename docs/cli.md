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

## Run A Script

Use `macrodata run` to run a Macrodata Refiner pipeline script.

```bash
macrodata run examples/local_log_stream.py
```

Optional flags:

```bash
macrodata run --logs one examples/local_log_stream.py -- --workers 4 --rows 20
```

`macrodata run` currently supports:

- `--logs all|none|one|errors`
- `Ctrl+C` exits with code `130`
- local launcher resume/failure messages are printed cleanly, while ordinary script exceptions still surface normally
- the script directory is added to `sys.path`, so sibling imports work the same way they do with `python script.py`

## Credential Lookup

See [Auth](auth.md) for the shared credential lookup order and credential file location.

## Notes

- `launch_local(...)` does not require Macrodata auth
- `launch_cloud(...)` requires Macrodata auth
