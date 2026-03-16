#!/usr/bin/env python3
"""
EPUB Summarizer for Game World Generation
Converts an ebook into a structured game design bible via Ollama.

Usage:
    python epub_summarizer.py <path_to_epub> [--model gemma3:27b] [--output output_dir]

Requires:
    pip install ebooklib beautifulsoup4 requests
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub


# ── Configuration ─────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "gemma3:27b"

CHUNK_SIZE_WORDS = 1800      # words per chunk (leaves room for prompt + output)
CHUNK_OVERLAP_WORDS = 150    # overlap between chunks to avoid boundary loss
REQUEST_TIMEOUT = 300        # seconds — big model, give it time


# ── Prompts ────────────────────────────────────────────────────────────────────

CHUNK_PROMPT = """You are analyzing a chunk of a science fiction novel for the purpose of building a text adventure game.
Extract ONLY what is explicitly present in this chunk. Do not invent or infer.

Return a structured summary with these sections (omit any section if nothing relevant appears):

## Events
- Key things that happen, in order

## Characters Present
- Name, brief description, role, notable traits or dialogue

## Locations
- Name, physical description, atmosphere, notable features

## Creatures & Organisms
- Name, appearance, behavior, danger level

## Items & Technology
- Notable objects, weapons, tools, alien tech

## Lore & World Details
- Facts about the world, history, factions, rules (e.g. the virus, Hooper physiology)

## Tone & Atmosphere
- One or two sentences on the mood of this section

Chunk text:
{chunk_text}
"""

CHAPTER_SYNTHESIS_PROMPT = """You are synthesizing multiple chunk summaries from a single chapter of a science fiction novel.
Combine them into one coherent chapter summary, eliminating repetition but keeping all unique details.
Maintain the same structured format:

## Events
## Characters Present
## Locations
## Creatures & Organisms
## Items & Technology
## Lore & World Details
## Tone & Atmosphere

Chunk summaries to synthesize:
{chunk_summaries}
"""

WORLD_DOCUMENT_PROMPT = """You are a game designer creating a structured world bible for a text adventure game based on a novel.
Using the chapter summaries provided, produce a comprehensive reference document.

Output the following sections in full detail:

# WORLD OVERVIEW
Brief description of the world, its dangers, and overall tone.

# LOCATIONS
For each distinct location: name, full description, atmosphere, points of interest, dangers, connected locations.

# CHARACTERS
For each significant character: full name, physical description, personality, background, role in story, relationships, special abilities or traits.

# CREATURES & ORGANISMS
For each creature: name, appearance, behavior, habitat, danger level, any special properties.

# FACTIONS & GROUPS
Name, goals, members, relationship to player/protagonist.

# MECHANICS & RULES
Special world rules: the virus, Hooper physiology, AI drones, anything that would affect gameplay.

# KEY ITEMS & TECHNOLOGY
Significant objects, weapons, ships, tech — with descriptions.

# PLOT ARC
Major story beats in order, suitable for structuring a game narrative.

# ATMOSPHERE & TONE GUIDE
How the world should feel. Vocabulary, mood, what makes this world unique. Reference specific details.

Chapter summaries:
{chapter_summaries}
"""


# ── Ollama call ────────────────────────────────────────────────────────────────

def call_ollama(prompt: str, model: str) -> str:
    """Send a prompt to Ollama and return the response text."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": 0.3,   # low temp for factual extraction
            "num_predict": 2048,
        }
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        print("\n[ERROR] Cannot connect to Ollama. Is it running? (ollama serve)")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("\n[ERROR] Ollama request timed out. Try reducing CHUNK_SIZE_WORDS.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Ollama call failed: {e}")
        sys.exit(1)


# ── EPUB parsing ───────────────────────────────────────────────────────────────

def extract_chapters_from_epub(epub_path: str) -> list[dict]:
    """
    Parse EPUB and return list of chapters.
    Each chapter: {"title": str, "text": str}
    """
    print(f"[*] Loading EPUB: {epub_path}")
    book = epub.read_epub(epub_path)

    chapters = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), "html.parser")

            # Try to find a chapter title
            title_tag = soup.find(["h1", "h2", "h3", "title"])
            title = title_tag.get_text(strip=True) if title_tag else f"Section {len(chapters)+1}"

            # Get clean text
            text = soup.get_text(separator=" ", strip=True)
            text = re.sub(r'\s+', ' ', text).strip()

            # Skip very short sections (TOC, copyright pages, etc.)
            if len(text.split()) < 100:
                continue

            chapters.append({"title": title, "text": text})

    print(f"[*] Found {len(chapters)} content sections.")
    return chapters


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


# ── Progress management ────────────────────────────────────────────────────────

def load_progress(progress_file: Path) -> dict:
    if progress_file.exists():
        with open(progress_file) as f:
            data = json.load(f)
        print(f"[*] Resuming from saved progress ({len(data.get('chapter_summaries', {}))} chapters done).")
        return data
    return {"chunk_summaries": {}, "chapter_summaries": {}}


def save_progress(progress_file: Path, progress: dict):
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def process_book(epub_path: str, model: str, output_dir: Path,
                 chunk_size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS):
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_file = output_dir / "progress.json"
    progress = load_progress(progress_file)

    chapters = extract_chapters_from_epub(epub_path)
    total_words = sum(len(c["text"].split()) for c in chapters)
    print(f"[*] Total words: {total_words:,}")
    print(f"[*] Model: {model}")
    print(f"[*] Chunk size: {chunk_size} words\n")

    # ── Pass 1 & 2: Per-chapter processing ────────────────────────────────────
    for ch_idx, chapter in enumerate(chapters):
        ch_key = f"ch_{ch_idx:03d}"
        ch_title = chapter["title"]

        if ch_key in progress["chapter_summaries"]:
            print(f"[✓] Chapter {ch_idx+1}/{len(chapters)}: '{ch_title}' (cached)")
            continue

        print(f"\n[→] Chapter {ch_idx+1}/{len(chapters)}: '{ch_title}'")
        words = chapter["text"].split()
        print(f"    {len(words):,} words", end="")

        chunks = chunk_text(chapter["text"], chunk_size, overlap)
        print(f" → {len(chunks)} chunk(s)")

        chunk_results = []
        for c_idx, chunk in enumerate(chunks):
            chunk_key = f"{ch_key}_chunk_{c_idx:03d}"

            if chunk_key in progress["chunk_summaries"]:
                print(f"    [✓] Chunk {c_idx+1}/{len(chunks)} (cached)")
                chunk_results.append(progress["chunk_summaries"][chunk_key])
                continue

            print(f"    [→] Chunk {c_idx+1}/{len(chunks)}...", end=" ", flush=True)
            t0 = time.time()
            prompt = CHUNK_PROMPT.format(chunk_text=chunk)
            result = call_ollama(prompt, model)
            elapsed = time.time() - t0
            print(f"done ({elapsed:.0f}s)")

            progress["chunk_summaries"][chunk_key] = result
            chunk_results.append(result)
            save_progress(progress_file, progress)

        # Synthesize chunks into chapter summary
        if len(chunk_results) == 1:
            chapter_summary = chunk_results[0]
        else:
            print(f"    [→] Synthesizing {len(chunk_results)} chunks into chapter summary...", end=" ", flush=True)
            t0 = time.time()
            combined = "\n\n---\n\n".join(
                f"[Chunk {i+1}]\n{s}" for i, s in enumerate(chunk_results)
            )
            prompt = CHAPTER_SYNTHESIS_PROMPT.format(chunk_summaries=combined)
            chapter_summary = call_ollama(prompt, model)
            elapsed = time.time() - t0
            print(f"done ({elapsed:.0f}s)")

        progress["chapter_summaries"][ch_key] = {
            "title": ch_title,
            "summary": chapter_summary
        }
        save_progress(progress_file, progress)

        # Save individual chapter summary
        ch_file = output_dir / f"{ch_key}_{slugify(ch_title)}.md"
        with open(ch_file, "w") as f:
            f.write(f"# Chapter: {ch_title}\n\n{chapter_summary}\n")

    # ── Pass 3: World document ─────────────────────────────────────────────────
    world_doc_file = output_dir / "world_document.md"
    if world_doc_file.exists():
        print(f"\n[✓] World document already exists: {world_doc_file}")
    else:
        print(f"\n[→] Building world document from {len(progress['chapter_summaries'])} chapter summaries...")

        all_summaries = []
        for ch_key in sorted(progress["chapter_summaries"].keys()):
            entry = progress["chapter_summaries"][ch_key]
            all_summaries.append(f"## {entry['title']}\n\n{entry['summary']}")

        combined = "\n\n---\n\n".join(all_summaries)
        prompt = WORLD_DOCUMENT_PROMPT.format(chapter_summaries=combined)

        t0 = time.time()
        world_doc = call_ollama(prompt, model)
        elapsed = time.time() - t0
        print(f"[✓] World document generated ({elapsed:.0f}s)")

        with open(world_doc_file, "w") as f:
            f.write(f"# World Document\n*Generated from: {Path(epub_path).name}*\n\n")
            f.write(world_doc)

        print(f"[✓] Saved: {world_doc_file}")

    # Also save all chapter summaries in one file for convenience
    all_chapters_file = output_dir / "all_chapter_summaries.md"
    with open(all_chapters_file, "w") as f:
        f.write(f"# Chapter Summaries\n*Generated from: {Path(epub_path).name}*\n\n")
        for ch_key in sorted(progress["chapter_summaries"].keys()):
            entry = progress["chapter_summaries"][ch_key]
            f.write(f"---\n\n## {entry['title']}\n\n{entry['summary']}\n\n")

    print(f"\n[✓] All done!")
    print(f"    Chapter summaries : {all_chapters_file}")
    print(f"    World document    : {world_doc_file}")
    print(f"    Progress file     : {progress_file}")


# ── Utilities ──────────────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    text = re.sub(r'[^\w\s-]', '', text.lower())
    return re.sub(r'[\s_-]+', '_', text).strip('_')[:50]


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Summarize an EPUB for game world generation.")
    parser.add_argument("epub", help="Path to the EPUB file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--output", default="book_summary", help="Output directory (default: book_summary/)")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE_WORDS,
                        help=f"Words per chunk (default: {CHUNK_SIZE_WORDS})")
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP_WORDS,
                        help=f"Overlap words between chunks (default: {CHUNK_OVERLAP_WORDS})")
    args = parser.parse_args()

    if not os.path.exists(args.epub):
        print(f"[ERROR] File not found: {args.epub}")
        sys.exit(1)

    process_book(args.epub, args.model, Path(args.output),
                 chunk_size=args.chunk_size, overlap=args.overlap)


if __name__ == "__main__":
    main()
