---
title: "Workspaces and API keys"
description: "Organize platform resources and authenticate Refiner jobs"
---

# Workspaces and API keys

A workspace owns the platform resources created by Refiner: jobs, logs, metrics,
manifests, files, secrets, API keys, services, viewer access, billing, and
members.

Everyone starts with a personal workspace. Create shared workspaces for team
datasets, production workflows, lab projects, or anything where other users
need access to the same jobs and secrets.

## Manage workspaces

Open [Settings > Workspaces](/settings/workspaces). The page shows your
workspaces, the active workspace, pending invitations, and actions for shared
workspaces.

Use **Create Workspace** to add a workspace. The dialog asks for:

| Field | Meaning |
| --- | --- |
| Name | Human-readable workspace name. |
| Slug | URL identifier used in workspace-scoped routes. |

Use **Switch** to make a workspace active. Workspace-specific entry points
redirect to the active workspace:

| Entry point | Opens |
| --- | --- |
| [Jobs](/jobs) | Active workspace jobs. |
| [Settings > General](/settings/general) | Active workspace details. |
| [Settings > Billing](/settings/billing) | Active workspace billing. |
| [Settings > API Keys](/settings/api-keys) | Active workspace API keys. |
| [Settings > Secrets](/settings/secrets) | Active workspace secrets. |
| [Settings > Members](/settings/members) | Active workspace members. |
| [Services](/services) | Active workspace runtime services. |
| [Viewer](/viewer) | Active workspace data viewer. |

You can create up to 3 workspaces by default. Contact the team if you need more.

## Personal and shared workspaces

Personal workspaces are private to your user. They are the default place for
first runs and individual experiments.

Shared workspaces support members and invitations. Use them when jobs, secrets,
API keys, billing, services, or viewer access should belong to a team instead
of one person.

To invite users:

1. Open [Settings > Workspaces](/settings/workspaces).
2. Click **Manage Members** for the shared workspace, or open
   [Settings > Members](/settings/members).
3. Click **Invite**.
4. Enter the user's email address.
5. Click **Send Invite**.

Pending invitations appear on both the workspace members page and the invited
user's workspaces page. Invitations can be accepted, rejected, or revoked.

## Roles

Workspace roles control who can administer the workspace.

| Role | Can do |
| --- | --- |
| owner | Manage workspace settings, billing, members, roles, API keys, and secrets. |
| admin | Manage most workspace resources and invite or remove members, except owner-only changes. |
| member | Use workspace jobs, services, secrets, API keys, and viewer access allowed to members. |

Role changes to elevated permissions require confirmation in the UI. Member
management is not available for personal workspaces; create a shared workspace
first.

## Workspace settings

Open [Settings](/settings) to see account and workspace controls. The sidebar
has a workspace selector and these workspace tabs:

| Tab | Direct URL | Use it for |
| --- | --- | --- |
| General | [Settings > General](/settings/general) | Rename the workspace and view its slug. |
| Billing | [Settings > Billing](/settings/billing) | Credits, payment method, auto-recharge, and usage breakdown. |
| API Keys | [Settings > API Keys](/settings/api-keys) | Create and delete API keys for CLI and cloud submission. |
| Secrets | [Settings > Secrets](/settings/secrets) | Store encrypted environment secrets for jobs and the viewer. |
| Members | [Settings > Members](/settings/members) | Invite users, change roles, revoke invitations, and remove members. |

## Create an API key

API keys authenticate the CLI and Python cloud launcher. They are scoped to the
workspace where they are created.

1. Open [Settings > API Keys](/settings/api-keys).
2. Click **Create API Key**.
3. Enter a name between 1 and 64 characters, such as `laptop` or `ci-prod`.
4. Click **Create API Key**.
5. Copy the key immediately. The full key is shown only once.

The API key table shows name, masked value, last-used time, created time, and a
delete action. Workspace owners and admins can also see member API keys.

## Log in locally

Use the copied key with the CLI:

```bash
macrodata login --token md_...
macrodata whoami
```

For CI:

```bash
export MACRODATA_API_KEY="md_..."
macrodata whoami
```

Environment credentials take precedence over locally saved credentials.

## Delete or rotate keys

Delete old keys from [Settings > API Keys](/settings/api-keys). Deleting a key
immediately prevents future CLI and cloud-launch requests with that key.

Rotate keys by creating a new key, updating local machines or CI secrets, then
deleting the old key after the new one has been used successfully.

## Related pages

- [Submitting to the Platform](submitting-to-the-platform.md)
- [CLI Auth and Run](../cli/auth-and-run.md)
- [Cloud Launcher](../running-pipelines/cloud-launcher.md)
- [Secrets and environment](secrets-and-environment.md)
