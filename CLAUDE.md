# CLAUDE.md — Codebase context for AI assistants

This file gives you the context needed to work on this project effectively. Read it before making any changes.

---

## What this project is

A pipeline that converts a novel (EPUB) into a browser-based, LLM-driven text adventure game with optional AI image generation. It is intentionally generic — no book-specific logic lives in any of the Python or HTML files. All book-specific content comes from `config.json` and the chapter markdown files in `book_summary/`.

---

## File responsibilities

| File | Role |
|---|---|
| `epub_summarizer.py` | Reads EPUB, chunks text, calls Ollama to extract structured chapter summaries, writes `book_summary/ch_*.md` |
| `image_pregen.py` | Reads chapter summaries, uses Ollama to extract scene list and enhance prompts for Flux, generates images via ComfyUI, writes `images/manifest.json` |
| `serve.py` | Minimal Python HTTP server: serves static files, exposes `/api/chapters` (sorted list of chapter files), proxies `/ollama/*` → Ollama to avoid browser CORS issues |
| `game.html` | Self-contained browser game client. Reads `config.json` at boot, loads chapter markdown files on demand, calls Ollama via the serve.py proxy, manages all game state in a single `G` object |
| `comfy_generator.py` | Standalone PyQt6 GUI tool for manual ComfyUI image generation with Ollama prompt enhancement. Independent of the game pipeline. |
| `config.json` | Single source of truth for all book-specific settings: titles, paths, Ollama model, ComfyUI settings, UI theme, game behaviour flags |

---

## Architecture principles

**Data-driven, not code-driven.** When adapting for a new book, only `config.json` and the chapter markdown files change. `game.html`, `serve.py`, `epub_summarizer.py`, and `image_pregen.py` should remain unchanged between books.

**No build step.** `game.html` is a single self-contained file (HTML + CSS + JS inline). Do not split it into separate files or introduce a bundler.

**No external JS dependencies.** `game.html` uses no npm packages, no CDN links, no frameworks. Vanilla JS only.

**Progressive fallback for images.** Pre-generated manifest → dynamic ComfyUI → placeholder. Never block the game on image availability.

**Resumable long operations.** Both `epub_summarizer.py` and `image_pregen.py` write progress JSON after every unit of work and skip already-completed items on re-run. Preserve this pattern in any additions.

---

## Key data flows

### Boot sequence (game.html)
```
boot()
  └─ fetch /config.json
  └─ set PROGRESS_KEY from book title        ← scoped per book, uses localStorage
  └─ fetch images/manifest.json              ← optional, {} if missing
  └─ fetch /api/chapters                     ← served by serve.py
  └─ showTitleScreen() or startGame()
```

### Per-turn game loop (game.html)
```
handlePlayerInput()
  └─ callGameMaster(text)
       └─ buildSystemPrompt()                ← assembles chapter data + game state
       └─ ollamaChat()                       ← POST /ollama/api/chat (proxied)
       └─ parseGMResponse()                  ← JSON with fallback extraction
       └─ update location / inventory / image
       └─ appendNarrative()
       └─ if chapter_complete → saveProgress() + appendChapterComplete()
```

### Image resolution (game.html → showImage)
```
showImage(slug)
  ├─ slug in manifest?  →  load local file path
  └─ comfyui.enabled?   →  generateImageDynamic(slug)
                              └─ buildFluxWorkflow()
                              └─ POST comfyui.url/prompt
                              └─ poll /history/{id}
                              └─ cache result in G.manifest
```

### Chapter markdown format (produced by epub_summarizer.py)
```markdown
# Chapter: Title Here

## Events
- bullet list of key events

## Characters Present
- **Name:** description

## Locations
- **Name:** description

## Creatures & Organisms
- **Name:** description

## Items & Technology
- ...

## Lore & World Details
- ...

## Tone & Atmosphere
One or two sentences.
```

The game's `parseChapter()` function uses regex section extraction. The `## Characters Present` and `## Locations` sections are parsed into structured arrays; others are kept as raw strings. The first character listed becomes the player character; the first location becomes the starting location.

---

## The GM JSON envelope

Ollama is prompted to return this exact JSON structure on every turn. Do not change field names without updating both `buildSystemPrompt()` and `parseGMResponse()` / all call sites:

```json
{
  "narrative": "2nd-person prose, \\n\\n for paragraph breaks",
  "image_slug": "manifest_slug_or_null",
  "new_location": "location name or null",
  "inventory_add": ["item1"],
  "inventory_remove": ["item2"],
  "chapter_complete": false,
  "hint": "subtle hint for player"
}
```

`parseGMResponse()` strips markdown fences, attempts `JSON.parse`, then falls back to regex extraction of the first `{...}` block, then falls back to treating the entire response as raw narrative. Don't break this fallback chain.

---

## Progress persistence (localStorage)

The key is `game_progress_<book_title_lowercased_underscored>`, set in `boot()` after config loads. This scoping ensures different books don't share progress slots. Structure stored:

```json
{ "lastCompleted": 2, "savedAt": "2025-..." }
```

`lastCompleted` is the 0-based chapter index. On resume, `startGame(lastCompleted + 1)` is called.

---

## Image prompt pipeline (image_pregen.py)

```
chapter summaries
  └─ Ollama: EXTRACTION_PROMPT  →  list of {type, slug, prompt} items
  └─ saved to images/image_list.json  ← edit this between runs to adjust
  └─ for each item:
       └─ Ollama: ENHANCE_PROMPT(prompt + STYLE_SUFFIX)  →  enhanced Flux prompt
       └─ ComfyUI: build_flux_workflow(enhanced_prompt)
       └─ poll ComfyUI until done
       └─ download and save PNG
       └─ write to progress.json + manifest.json
```

The `STYLE_SUFFIX` is prepended to the raw prompt *before* enhancement, so Ollama sees the full stylistic intent and can weave it into natural language. The manifest stores both `prompt` (original) and `prompt_enhanced` (what was actually sent to Flux).

---

## ComfyUI workflow notes

Both `image_pregen.py` and `game.html` build the same native Flux workflow graph using these node IDs:

| Node | Class | Role |
|---|---|---|
| 1 | UNETLoader | Loads Flux UNET weights |
| 2 | VAELoader | Loads VAE (`ae.safetensors`) |
| 3 | DualCLIPLoader | Loads CLIP-L + T5-XXL |
| 4 | CLIPTextEncode | Encodes the positive prompt |
| 5 | EmptyLatentImage | Sets resolution |
| 6 | FluxGuidance | Sets guidance scale (1.0 schnell / 3.5 dev) |
| 7 | KSampler | Runs diffusion |
| 8 | VAEDecode | Decodes latent to pixels |
| 9 | SaveImage | Saves to ComfyUI output dir |

Known quirk: node 7 wires `negative: ["4", 0]`, reusing the positive CLIP output as the negative. This is harmless for Flux Schnell (negatives ignored) but would need a separate empty CLIPTextEncode node if proper negative prompts are ever needed for Flux Dev.

---

## serve.py proxy

The Ollama base URL is derived by splitting on `/api/`:
```python
ollama_base = config["ollama"]["url"].rsplit("/api/", 1)[0]
```
So `http://localhost:11434/api/chat` → `http://localhost:11434`, and the path `/ollama/api/chat` → `/api/chat` is appended. If the Ollama URL format changes, update this logic.

Config is re-read from disk on every request (not cached at startup), so editing `config.json` while the server is running takes effect immediately.

---

## What to preserve when making changes

- **Generic-ness**: no book titles, character names, or world-specific strings in `game.html`, `serve.py`, `epub_summarizer.py`, or `image_pregen.py`. Those belong in `config.json` and the chapter markdown files only.
- **Single-file HTML**: keep `game.html` as one file.
- **Progress resumability**: any new long-running operation should checkpoint after each unit of work.
- **The GM JSON envelope**: field names are shared between `buildSystemPrompt()` and several handler functions. Change them together or not at all.
- **PROGRESS_KEY scoping**: must remain derived from `config.book.title` so books are isolated in localStorage.
- **Image fallback chain**: manifest → dynamic → placeholder. New image sources slot in at the top, not as replacements.
