# Book-to-Text-Adventure Engine

A pipeline that converts any novel (EPUB) into a playable, LLM-driven text adventure game — with optional AI-generated scene images.

---

## How it works

```
EPUB
  └─► epub_summarizer.py   →  book_summary/ch_*.md  (structured chapter summaries)
                                        │
                          ┌─────────────┴──────────────┐
                          ▼                            ▼
              image_pregen.py                     serve.py + game.html
              (pre-generates scene              (local web server +
               images via ComfyUI)              LLM game master via Ollama)
```

1. **epub_summarizer.py** — reads the EPUB, chunks it, and asks Ollama to extract structured summaries (events, characters, locations, creatures, items, lore, tone) for each chapter. Produces one markdown file per chapter plus an all-in-one summary file.

2. **image_pregen.py** *(optional)* — reads the chapter summaries, asks Ollama to identify every distinct location and scene, enhances each description into an optimised Flux prompt, then generates images via ComfyUI. Produces `images/manifest.json` that the game uses.

3. **serve.py + game.html** — a local web server that proxies Ollama calls and serves the game. The game reads each chapter's markdown, builds a system prompt from it, and uses Ollama as a real-time game master. The player navigates the story in their browser.

---

## Requirements

### Python packages
```bash
pip install ebooklib beautifulsoup4 requests
```

### External services
| Service | Purpose | Default URL |
|---|---|---|
| [Ollama](https://ollama.com) | LLM for summarisation, image enhancement, and game master | `http://localhost:11434` |
| [ComfyUI](https://github.com/comfyanonymous/ComfyUI) | Image generation via Flux | `http://127.0.0.1:8188` *(optional)* |

### Ollama model
```bash
ollama pull gemma3:27b
```
Any instruction-following model works. Larger = better narrative quality.

### ComfyUI models *(only needed for image generation)*
Place in the appropriate `models/` subdirectories inside your ComfyUI installation:
- `models/unet/flux1-schnell.safetensors` or `flux1-dev.safetensors`
- `models/vae/ae.safetensors`
- `models/clip/clip_l.safetensors`
- `models/clip/t5xxl_fp16.safetensors`

---

## Step-by-step setup

### 1 — Summarise your book

```bash
python epub_summarizer.py my_book.epub
```

Output goes to `book_summary/` by default. This is the slowest step — a 400-page novel takes 30–90 minutes depending on your model. Progress is saved automatically; you can interrupt and resume.

```
Options:
  --model       Ollama model name           (default: gemma3:27b)
  --output      Output directory            (default: book_summary/)
  --chunk-size  Words per chunk             (default: 1800)
  --overlap     Overlap words between chunks (default: 150)
```

### 2 — Configure the game

Edit `config.json` for your book. At minimum, update the `book` section and the `comfyui.style_suffix` to match your book's setting. See the full [config.json reference](#configjson-reference) below.

### 3 — Pre-generate images *(optional)*

```bash
python image_pregen.py --summaries book_summary/all_chapter_summaries.md --output images/
```

This step calls Ollama to extract a scene list, enhances each prompt for Flux, then generates images via ComfyUI. Use `--list-only` first to review and edit the scene list before committing to generation.

```
Options:
  --summaries   Path to the all_chapter_summaries.md file  (default: all_chapter_summaries.md)
  --output      Directory for images and manifest           (default: images/)
  --model       flux-schnell (fast) or flux-dev (quality)  (default: flux-dev)
  --list-only   Extract and print scene list only, no generation
```

Image generation is resumable — already-generated slugs are skipped. Edit `images/image_list.json` between runs to adjust prompts.

### 4 — Start the game

```bash
python serve.py
```

Opens `http://localhost:8080/game.html` automatically. Pass a port number as an argument to use a different port: `python serve.py 9000`.

---

## File layout

```
project/
├── config.json                  # All settings — edit this for your book
├── game.html                    # Browser game client (no editing needed)
├── serve.py                     # Local web server + Ollama proxy
├── epub_summarizer.py           # Step 1: EPUB → chapter summaries
├── image_pregen.py              # Step 2: summaries → pre-generated images
├── comfy_generator.py           # Standalone ComfyUI image tool (optional)
│
├── book_summary/                # Created by epub_summarizer.py
│   ├── ch_000_chapter_title.md
│   ├── ch_001_chapter_title.md
│   ├── ...
│   ├── all_chapter_summaries.md
│   ├── world_document.md
│   └── progress.json            # Resumable progress tracker
│
└── images/                      # Created by image_pregen.py
    ├── some_location.png
    ├── some_scene.png
    ├── ...
    ├── image_list.json          # Editable scene list (slugs + prompts)
    ├── manifest.json            # Used by game.html to find images
    └── progress.json            # Resumable progress tracker
```

---

## config.json reference

### `book`

| Key | Description |
|---|---|
| `title` | Displayed in the browser tab, game header, and title screen |
| `subtitle` | Displayed on the title screen below the title |
| `chapters_dir` | Path to the directory containing `ch_*.md` files (relative to project root) |
| `images_dir` | Path to the directory containing generated images |
| `manifest` | Path to `manifest.json` produced by `image_pregen.py` |

### `ollama`

| Key | Description |
|---|---|
| `url` | Full URL of the Ollama chat endpoint |
| `model` | Model to use as game master (e.g. `gemma3:27b`, `llama3:8b`) |
| `temperature` | Creativity of responses. `0.5`–`0.9` is a good range. Higher = more unpredictable |
| `max_tokens` | Maximum tokens per game master response (`num_predict` in Ollama terms) |

### `comfyui`

| Key | Description |
|---|---|
| `url` | Base URL of the ComfyUI server |
| `enabled` | `true` / `false` — whether dynamic image generation is attempted for slugs not in the manifest |
| `model` | `"flux-schnell"` (4 steps, fast) or `"flux-dev"` (20 steps, higher quality) |
| `steps` | Diffusion steps. `4` for schnell, `20` for dev. |
| `width` | Output image width in pixels |
| `height` | Output image height in pixels |
| `style_suffix` | Appended to every dynamic image prompt. Tune this to your book's visual identity. |

### `theme`

All colours are CSS hex values or `rgba(...)` strings. They are injected as CSS variables at runtime.

| Key | CSS variable | Purpose |
|---|---|---|
| `background` | `--bg` | Page background |
| `surface` | `--surface` | Panel backgrounds (header, input bar) |
| `surface_raised` | `--surface-raised` | Elevated elements (input field) |
| `border` | `--border` | Dividers and box borders |
| `text` | `--text` | Body text |
| `text_dim` | `--text-dim` | Hints, labels, status text |
| `text_bright` | `--text-bright` | Headings, player input echo |
| `accent` | `--accent` | Primary accent (borders, inventory tags, send button) |
| `accent_bright` | `--accent-bright` | Highlighted accent (chapter label, links) |
| `accent_glow` | `--accent-glow` | Glow / hover fill — should be `rgba` with low alpha |
| `danger` | `--danger` | Error backgrounds |
| `danger_bright` | `--danger-bright` | Error text |
| `font_display` | `--font-display` | Title and chapter heading font stack |
| `font_body` | `--font-body` | Narrative text font stack |
| `font_ui` | `--font-ui` | UI labels, input, monospace elements |
| `image_height` | `--image-height` | Fixed height of the scene image panel (e.g. `420px`) |
| `narrative_height` | `--narrative-height` | Reserved for future use |

### `game`

| Key | Description |
|---|---|
| `show_hints` | `true` / `false` — show the subtle next-action hint below each narrative response |
| `show_inventory` | `true` / `false` — show the inventory bar in the status strip |
| `show_visited_locations` | `true` / `false` — reserved for future visited-location display |
| `typewriter_speed` | Reserved — typewriter effect speed in ms per character |
| `chapter_intro_delay` | Milliseconds to wait after displaying the chapter header before the opening narration begins |
| `title_screen` | Set to `false` to skip the title screen and go straight into the game |

---

## How the game master works

Each chapter markdown file is parsed to extract: events, characters, locations, creatures, lore, and tone. This structured data forms the system prompt sent to Ollama on every player turn. The LLM is instructed to respond in a strict JSON envelope:

```json
{
  "narrative": "...",
  "image_slug": "slug_or_null",
  "new_location": "location name or null",
  "inventory_add": [],
  "inventory_remove": [],
  "chapter_complete": false,
  "hint": "..."
}
```

The game applies the state changes (location, inventory), looks up or generates the image, and appends the narrative. Conversation history is maintained within each chapter; it resets when a new chapter loads.

---

## Image strategy

The game uses a two-tier image system:

1. **Pre-generated** (preferred): `image_pregen.py` generates images in advance and records them in `manifest.json`. When the game master returns a slug, the game looks it up in the manifest and loads the local file instantly.

2. **Dynamic** (fallback): If `comfyui.enabled` is `true` and a slug has no manifest entry, the game submits a prompt to ComfyUI in real time and polls for the result. This adds 30–120 seconds of wait time per image.

If neither applies, the image panel shows a placeholder.

---

## Chapter progress

The game saves which chapters have been completed to `localStorage`, keyed by book title (e.g. `game_progress_the_skinner`). On the title screen, a **Continue** button appears if saved progress exists. **New Game** clears progress and restarts from chapter 1. Each book gets its own independent progress slot.

---

## Adapting to a new book

1. Run `epub_summarizer.py` on your new EPUB.
2. Copy `config.json`, update `book.title`, `book.subtitle`, `comfyui.style_suffix`, and the `theme` colours to match your book's visual identity.
3. Optionally run `image_pregen.py` with the new summaries.
4. Run `serve.py` — everything else is driven by the chapter markdown files.

No changes to `game.html` or `serve.py` are needed between books.

---

## Troubleshooting

**"No chapter files found"** — Check that `book.chapters_dir` in `config.json` points to your `book_summary/` folder and that it contains files matching `ch_*.md`.

**"Ollama unreachable"** — Make sure Ollama is running (`ollama serve`) and the `ollama.url` in `config.json` is correct.

**Images not showing** — If using pre-generated images, check that `images/manifest.json` exists and `book.manifest` points to it. If using dynamic generation, check that ComfyUI is running and the Flux model files are in place.

**epub_summarizer.py crashed mid-run** — Just re-run the same command. Progress is saved after every chunk and chapter.

**image_pregen.py skipping everything** — Check `images/progress.json`. Delete an entry (or the whole file) to force regeneration.
