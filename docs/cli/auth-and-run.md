---
title: "Auth and Run"
description: "Authenticate and run Refiner scripts from the CLI"
---

# Auth And Run

## Login

```bash
macrodata login
```

Interactive login prompts for a Macrodata API key and validates it with the
platform. It stores the key locally so later CLI commands can talk to the
current workspace without passing a token every time.

Typical interactive output:

```text
Logged in as Ada Lovelace <ada@example.com>
API key name: laptop
Workspace: Acme Robotics (acme-robotics)
Credentials saved to /home/ada/.config/macrodata/credentials
```

Non-interactive environments can pass the key directly or through stdin:

```bash
macrodata login --token md_...
printf '%s' "$MACRODATA_API_KEY" | macrodata login --token-stdin --quiet
```

| Option | Use |
| --- | --- |
| `--token <key>` | Store a key passed as an argument |
| `--token-stdin` | Read the key from stdin |
| `--quiet` | Suppress banner and key-creation prompt |

## Check Identity

```bash
macrodata whoami
```

`whoami` verifies the locally stored API key against the platform and prints
the user, API key name, and active workspace. Use it before submitting jobs from
a new shell, CI runner, or shared machine.

Example output:

```text
Logged in as Ada Lovelace <ada@example.com>
API key name: laptop
Workspace: Acme Robotics (acme-robotics)
```

If the key is missing or expired, the command exits non-zero and tells you to
run `macrodata login`.

## Logout

```bash
macrodata logout
```

`logout` deletes the locally stored API key. It does not revoke the key in the
platform; delete API keys from workspace settings when you want to invalidate
them.

Example output:

```text
Logged out. Local credentials removed.
```

If no credentials exist, it prints:

```text
No local credentials found.
```

## Run A Script

```bash
macrodata run train_data_pipeline.py
```

`run` executes a Python script with CLI-managed Refiner behavior. The script
should build and launch a pipeline, usually by calling `launch_local()` or
`launch_cloud()`.

For cloud launches, the CLI controls whether the terminal attaches to the
submitted job or returns after submission. In detached mode the launcher prints
follow-up commands:

```text
Cloud job submitted.
Job ID: job_123
URL: http://localhost:3000/app/acme-robotics/jobs/job_123
Attach: macrodata jobs attach job_123
Summary: macrodata jobs get job_123
Logs: macrodata jobs logs job_123 --stage 0
Workers: macrodata jobs workers job_123
Cancel: macrodata jobs cancel job_123
```

Pass script arguments after the script path:

```bash
macrodata run train_data_pipeline.py --dataset aloha --limit 10
```

Cloud launches can run attached or detached:

```bash
macrodata run --attach train_data_pipeline.py
macrodata run --detach train_data_pipeline.py
```

Attached mode can tune log display:

```bash
macrodata run --logs all train_data_pipeline.py
macrodata run --logs one train_data_pipeline.py
macrodata run --logs errors train_data_pipeline.py
macrodata run --logs none train_data_pipeline.py
```

| Option | Use |
| --- | --- |
| `--attach` | Force attached mode for cloud launches |
| `--detach` | Force detached mode for cloud launches |
| `--logs all` | Show live logs from capped workers |
| `--logs one` | Follow one worker |
| `--logs errors` | Show error lines only |
| `--logs none` | Hide log lines and update only the header |

## Related Pages

- [Workspaces and API Keys](../platform/workspaces-and-api-keys.md)
- [Cloud Launcher](../running-pipelines/cloud-launcher.md)
