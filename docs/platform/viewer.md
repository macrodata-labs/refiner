---
title: "Viewer"
description: "Inspect Parquet, JSON, and CSV files from workspace-accessible storage"
---

# Viewer

The viewer is a browser data inspector for Parquet, JSON, and CSV files. It can
open public files or private files resolved with workspace secrets.

Open [Viewer](/viewer), which redirects to the active workspace.

## What It Can Open

The viewer accepts:

| Path type | Example |
| --- | --- |
| S3 | `s3://bucket/path/file.parquet` |
| Google Cloud Storage | `gs://bucket/path/file.jsonl.gz` |
| Hugging Face | `hf://datasets/org/repo/path/file.csv` |
| HTTP/HTTPS | `https://example.com/data/file.parquet` |

Supported file types are **Auto**, **Parquet**, **JSON**, and **CSV**. Auto
uses the file extension. Choose the type explicitly when the extension is
ambiguous.

## Load A File

1. Open [Viewer](/viewer).
2. Pick a secrets environment from the **Environment** selector.
3. Paste a file URL or storage path.
4. Choose **Auto**, **Parquet**, **JSON**, or **CSV**.
5. Click **Load**.

For public files, choose **Environment: None**. For private buckets or private
Hugging Face repositories, choose the workspace secret environment that contains
the credentials.

## Private Storage

Secrets are managed in [Settings > Secrets](/settings/secrets).

Use these secret names:

| Storage | Secret names |
| --- | --- |
| S3-compatible storage | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, optional `AWS_SESSION_TOKEN`, `AWS_REGION` or `AWS_DEFAULT_REGION`, optional `AWS_ENDPOINT_URL` |
| Google Cloud Storage | `GOOGLE_APPLICATION_CREDENTIALS_JSON` |
| Hugging Face | `HF_TOKEN` |

The viewer uses the selected workspace secret environment to resolve storage
paths into browser-fetchable URLs. Secret values are not exposed in the UI.

## How It Works

For `http://` and `https://` paths, the viewer uses the URL directly.

For `s3://`, `gs://`, and `hf://` paths, the viewer calls the platform API for
the current workspace. The API checks workspace access, loads the selected
secret environment, verifies the source, and returns a temporary
browser-fetchable URL plus the canonical source path.

The browser then queries the file with DuckDB-Wasm. The data file is fetched by
the browser for preview, search, sort, and pagination.

## Table Controls

After loading a file, the viewer shows:

| Control | Use |
| --- | --- |
| Active URL/path | Confirms what source is loaded. |
| Row count | Number of rows when count succeeds. |
| Page controls | Move through 100-row pages. |
| Search | Search all text columns. |
| Column headers | Click to sort by a column. |
| Cells | Open large values, vectors, images, videos, audio, and file references. |

String cells that look like media or file URLs are rendered as links or previews.
S3, GCS, and Hugging Face cell values can also resolve through the same
workspace secret environment.

## CORS And Browser Fetching

The viewer fetches data client-side. For private files, the platform may return
a signed URL, but the browser still needs permission to read it.

If you see a CORS error, configure the bucket to allow the app origin. The
viewer error message includes a minimal CORS example for S3/R2 or GCS.

For S3-style buckets, allow:

```json
[
  {
    "AllowedOrigins": ["https://macrodata.co"],
    "AllowedMethods": ["GET"],
    "AllowedHeaders": ["Range", "*"],
    "ExposeHeaders": ["Accept-Ranges", "Content-Length", "Content-Range", "ETag"],
    "MaxAgeSeconds": 3600
  }
]
```

For GCS buckets, allow:

```json
[
  {
    "origin": ["https://macrodata.co"],
    "method": ["GET"],
    "responseHeader": ["Accept-Ranges", "Content-Length", "Content-Range", "ETag"],
    "maxAgeSeconds": 3600
  }
]
```

Use the actual app origin shown in your browser if you are on a different
deployment or local preview.

## Troubleshooting

| Error | Fix |
| --- | --- |
| Choose a file type | Auto could not infer the type. Select Parquet, JSON, or CSV. |
| Access denied | Check the selected secret environment and storage permissions. |
| File not found | Check the bucket/repo/path. |
| Missing credentials | Add the required secrets in [Settings > Secrets](/settings/secrets). |
| CORS or browser access error | Configure bucket CORS for the app origin. |
| Invalid file type | The selected file type does not match the file contents. |

## Related Pages

- [Secrets and Environment](secrets-and-environment.md)
- [Path Formats](../reference/path-formats.md)
- [Writing Data](../writing-data/index.md)
