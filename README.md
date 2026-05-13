---
title: MMD - Moroccan Manuscript Database Pipeline
emoji: 📜
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# MMD — Moroccan Manuscript Database Pipeline

A web-based pipeline for extracting structured metadata from Arabic manuscript catalog PDFs using Claude AI (Anthropic). Designed for digitization and cataloging of historical manuscripts.

## Live App

**[https://mmd12-mmd.hf.space](https://mmd12-mmd.hf.space)**

---

## What It Does

1. **Upload** a manuscript catalog PDF
2. **Preview** pages as images and select a page range to process
3. **Run** the AI extraction pipeline — Claude reads each page and extracts structured fields
4. **Match authors** against the WAAMD (World Authority file of Arabic-script Manuscript Data) database
5. **Download** results as a CSV file

---

## Extracted Fields

For each manuscript entry, the pipeline extracts:

| Field | Description |
|-------|-------------|
| `collection_no` | Collection / shelf number |
| `title` | Title of the manuscript |
| `author` | Author name (Arabic) |
| `author_romanized` | Author name (romanized) |
| `author_nisba` | Author's nisba (epithet) |
| `author_death` | Author's death date |
| `copyist` | Name of the copyist |
| `date_copied` | Date the manuscript was copied |
| `first_line` | Opening line of the text |
| `last_line` | Closing line of the text |
| `subject` | Subject / topic |
| `subject_romanized` | Subject romanized |
| `form` | Literary form |
| `form_romanized` | Literary form romanized |
| `dimensions_condition` | Physical dimensions and condition notes |
| `waamd_author_id` | Matched WAAMD author ID |

---

## How to Use

### Step 1 — Get an Anthropic API Key
The app requires your own [Anthropic API key](https://console.anthropic.com/). No key is stored server-side; you enter it in the app at runtime.

### Step 2 — Upload a PDF
Upload your manuscript catalog PDF via the web interface.

### Step 3 — Select Page Range
Preview the pages and choose which range to process (e.g. pages 10–50).

### Step 4 — Run Pipeline
Click **Run** and monitor progress. The pipeline uses:
- `claude-sonnet-4` for vision steps (image + text)
- `claude-haiku-4` for text-only steps (~20x cheaper)

### Step 5 — Download CSV
Once complete, download the structured CSV output. Author IDs are automatically matched against the WAAMD database.

---

## Models Used

| Model | Used For |
|-------|----------|
| `claude-sonnet-4-20250514` | Vision steps — reading manuscript page images |
| `claude-haiku-4-5-20251001` | Text-only steps — romanization, subject normalization, etc. |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Web framework | Flask 3.x |
| Server | Gunicorn |
| Image processing | Pillow |
| AI | Anthropic Claude API (via HTTP, no SDK) |
| Containerization | Docker |
| Hosting | Hugging Face Spaces (Docker SDK) |

---

## Hugging Face Space Configuration

This app runs as a **Docker Space** on Hugging Face. Key settings in the README frontmatter:

```yaml
sdk: docker        # Uses the Dockerfile at repo root
app_port: 7860     # Port exposed by gunicorn inside the container
```

The `Dockerfile` installs system dependencies (including `poppler-utils` for PDF-to-image conversion), installs Python packages from `requirements.txt`, and starts the Flask app via gunicorn with a 300-second timeout (to handle large PDFs).

---

## Important Notes

- **Temporary storage:** Uploaded PDFs, page images, and CSV outputs are stored in ephemeral container storage. Files will be lost when the Space restarts or sleeps.
- **API key:** Each user provides their own Anthropic API key at runtime. It is never stored.
- **Rate limiting:** The pipeline adds a 4-second delay between API calls to avoid hitting rate limits.
- **PDF size:** Very large PDFs may take several minutes to process depending on page count.

---

## Repository Structure

```
├── Dockerfile                  # Container definition
├── pipeline.py                 # Core extraction pipeline
├── add_waamd_author_ids.py     # WAAMD author ID matching
├── waamd.csv                   # WAAMD reference database
├── requirements.txt            # Python dependencies
├── prompts/                    # Per-field extraction prompts
│   ├── collectionno.txt
│   ├── title.txt
│   ├── author.txt
│   ├── author_nisba_prompt.txt
│   ├── author_romanization_prompt.txt
│   ├── authordeath.txt
│   ├── copyist.txt
│   ├── datecopied.txt
│   ├── firstline.txt
│   ├── lastline.txt
│   ├── subject.txt
│   ├── form.txt
│   ├── form_romanization_prompt.txt
│   ├── dimensions_condition_prompt.txt
│   └── catalog_header_page_prompt.txt
└── webapp/
    ├── app.py                  # Flask web application
    ├── static/styles.css       # Styling
    └── templates/
        ├── index.html          # Upload & page selection UI
        └── job.html            # Job progress & results UI
```

---

## Local Development

```bash
git clone https://huggingface.co/spaces/mmd12/MMD
cd MMD
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
cd webapp && python app.py
```

App will be available at `http://localhost:7860`.

---

## Links

- **Live Space:** [https://huggingface.co/spaces/mmd12/MMD](https://huggingface.co/spaces/mmd12/MMD)
- **Anthropic Console:** [https://console.anthropic.com](https://console.anthropic.com)
- **WAAMD Database:** [https://www.waamd.org](https://www.waamd.org)
