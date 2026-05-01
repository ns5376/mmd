---
title: Manuscript Pipeline
emoji: "📜"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# Manuscript Pipeline

This repository is configured to run as a Hugging Face Docker Space.

## What it does

- Upload a manuscript catalog PDF
- Convert pages into preview images
- Select a page range
- Run the extraction pipeline
- Download CSV results

## Important limitation on free hosting

This Space stores runtime files in temporary storage. Uploaded PDFs, generated page images, logs, and CSV outputs can be lost whenever the Space restarts or sleeps.

## Runtime notes

- The app asks each user for their own Anthropic API key at run time.
- No server-side Anthropic API key is required just to launch the app.
