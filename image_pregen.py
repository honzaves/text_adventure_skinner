#!/usr/bin/env python3
"""
Spatterjay Image Pre-Generator
Extracts locations/scenes from chapter summaries, then generates images via ComfyUI.

Usage:
    python image_pregen.py --summaries all_chapter_summaries.md --output images/
    python image_pregen.py --summaries all_chapter_summaries.md --output images/ --model flux-dev
    python image_pregen.py --list-only   # just extract & show the image list, no generation

Requires:
    pip install requests
ComfyUI running at http://127.0.0.1:8188/
Ollama running at http://localhost:11434/
"""

import argparse
import base64
import json
import os
import random
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

import requests


# ── Config ─────────────────────────────────────────────────────────────────────

COMFYUI_URL  = "http://127.0.0.1:8188"
OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma3:27b"

# Image settings
IMAGE_WIDTH  = 1344
IMAGE_HEIGHT = 768  # widescreen, good for adventure game panels

# Flux schnell: fast, 4 steps. Flux dev: quality, 20 steps.
FLUX_SCHNELL_STEPS = 4
FLUX_DEV_STEPS     = 20
FLUX_CFG           = 1.0   # Flux ignores CFG mostly, keep at 1

# Spatterjay style suffix appended to every prompt
STYLE_SUFFIX = (
    "science fiction, biopunk ocean world, dark and oppressive atmosphere, "
    "alien sea creatures, grimy frontier settlement, dramatic lighting, "
    "detailed digital painting, cinematic composition, Neal Asher aesthetic"
)

# Negative prompt (used with dev; schnell ignores it)
NEGATIVE_PROMPT = (
    "cartoon, anime, bright colors, cheerful, clean, sterile, "
    "low quality, blurry, text, watermark"
)


# ── Ollama ─────────────────────────────────────────────────────────────────────

def ollama_call(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.4, "num_predict": 4096}
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=300)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        print("[ERROR] Cannot connect to Ollama. Is it running?")
        sys.exit(1)


EXTRACTION_PROMPT = """You are a game designer reading chapter summaries of Neal Asher's science fiction novel "The Skinner" (set on the ocean world Spatterjay).

Your task: extract a definitive list of images to pre-generate for a text adventure game.

You MUST output ONLY lines in these exact formats, nothing else:
LOCATION | slug_name | detailed image generation prompt
SCENE | slug_name | detailed image generation prompt

Example output (follow this format exactly):
LOCATION | olians_tavern | dimly lit alien tavern interior, rough wooden tables, flickering bioluminescent lights, weathered fishermen drinking, nautical equipment hanging on walls, dark ocean visible through grimy windows, oppressive humid atmosphere
LOCATION | open_ocean | vast dark alien ocean, massive bioluminescent waves, strange creatures breaking the surface, two moons in overcast sky, threatening and oppressive, deep purple and black tones
SCENE | leech_attack | enormous alien leech surging from dark water, glistening segmented body, rows of teeth, spray of brine, rocky shoreline at night, horror atmosphere

Rules:
- Every single line must start with either LOCATION or SCENE followed by a pipe character |
- slug_name: lowercase letters and underscores only, no spaces, no special characters
- Image prompts: purely visual description of what you SEE, no character names, no plot
- Be specific about colors, lighting, architecture, creatures, weather, time of day
- Output 15-25 LOCATION lines and 10-15 SCENE lines
- Do not write any other text, headers, explanations, or commentary — ONLY the pipe-delimited lines

Chapter summaries:
{summaries}
"""


ENHANCE_PROMPT = """You are an expert at writing image generation prompts for Flux, a state-of-the-art text-to-image model.

Flux works best with:
- Natural language sentences, not keyword lists
- Specific lighting descriptions (e.g. "lit by a single lantern casting warm orange light")
- Cinematographic language (e.g. "wide establishing shot", "close-up", "low angle")
- Concrete visual details rather than abstract quality boosters
- Style anchors like "in the style of a 1970s sci-fi paperback cover" or "photorealistic, 35mm film"
- Avoiding: "masterpiece", "highly detailed", "8k", "best quality" — Flux ignores these

Rewrite the following image description as an optimised Flux prompt.
Return ONLY the improved prompt text — no explanation, no preamble, no quotes.

Description to improve:
{description}
"""


def enhance_prompt(description: str) -> str:
    """Use Ollama to rewrite a raw description as an optimised Flux prompt."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user",
                      "content": ENHANCE_PROMPT.format(description=description)}],
        "stream": False,
        "options": {"temperature": 0.5, "num_predict": 400},
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        r.raise_for_status()
        result = r.json()["message"]["content"].strip()
        # Strip any accidental quotes or markdown fences
        result = result.strip('"\'`')
        return result
    except requests.exceptions.ConnectionError:
        print("[ERROR] Cannot connect to Ollama for prompt enhancement.")
        sys.exit(1)
    except Exception as ex:
        print(f"[!] Enhance failed ({ex}) — using original prompt")
        return description


def extract_image_list(summaries_text: str) -> list[dict]:
    """Use Ollama to extract structured list of locations and scenes."""
    print("[→] Asking Ollama to extract image list from chapter summaries...")
    raw = ollama_call(EXTRACTION_PROMPT.format(summaries=summaries_text))

    items = []
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("LOCATION |") or line.startswith("SCENE |"):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                items.append({
                    "type":   parts[0].lower(),
                    "slug":   parts[1],
                    "prompt": parts[2],
                })

    if len(items) == 0:
        print("[!] Extraction returned 0 items. Raw Ollama response (first 2000 chars):")
        print("-" * 60)
        print(raw[:2000])
        print("-" * 60)
        print("[!] Delete images/image_list.json if it exists and try again.")
        sys.exit(1)

    print(f"[✓] Extracted {len(items)} images ({sum(1 for i in items if i['type']=='location')} locations, "
          f"{sum(1 for i in items if i['type']=='scene')} scenes)")
    return items


# ── ComfyUI ────────────────────────────────────────────────────────────────────

def get_flux_model_name(prefer_dev: bool = False) -> str:
    """Query ComfyUI UNETLoader to find the Flux model filename."""
    try:
        r = requests.get(f"{COMFYUI_URL}/object_info/UNETLoader", timeout=10)
        r.raise_for_status()
        models = r.json()["UNETLoader"]["input"]["required"]["unet_name"][0]
        flux_models = [m for m in models if "flux" in m.lower()]
        if not flux_models:
            print(f"[ERROR] No Flux models found in models/unet/. Available: {models}")
            print("[ERROR] Did you symlink flux1-dev.safetensors into models/unet/?")
            sys.exit(1)

        if prefer_dev:
            dev = [m for m in flux_models if "dev" in m.lower()]
            if dev:
                return dev[0]
        schnell = [m for m in flux_models if "schnell" in m.lower()]
        if schnell:
            return schnell[0]
        return flux_models[0]
    except Exception as e:
        print(f"[ERROR] Cannot connect to ComfyUI at {COMFYUI_URL}: {e}")
        sys.exit(1)


def build_flux_workflow(prompt: str, model_name: str,
                        width: int, height: int,
                        steps: int, seed: int) -> dict:
    """
    Build a ComfyUI workflow for Flux using native Flux nodes.
    Requires files in:
      models/unet/flux1-dev.safetensors (or flux1-schnell.safetensors)
      models/vae/ae.safetensors
      models/clip/clip_l.safetensors
      models/clip/t5xxl_fp16.safetensors
    """
    # Schnell uses euler+simple, dev uses euler+simple too but more steps
    # Guidance: schnell=1.0, dev=3.5
    guidance = 1.0 if "schnell" in model_name else 3.5
    clip_type = "flux"

    return {
        # Load UNET (diffusion model only)
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": model_name,
                "weight_dtype": "default"
            }
        },
        # Load VAE
        "2": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "ae.safetensors"
            }
        },
        # Load dual CLIP (clip_l + t5xxl)
        "3": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": "clip_l.safetensors",
                "clip_name2": "t5xxl_fp16.safetensors",
                "type": clip_type
            }
        },
        # Encode positive prompt
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["3", 0]
            }
        },
        # Empty latent
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1
            }
        },
        # FluxGuidance — sets guidance scale
        "6": {
            "class_type": "FluxGuidance",
            "inputs": {
                "guidance": guidance,
                "conditioning": ["4", 0]
            }
        },
        # KSampler with Flux-appropriate settings
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model":        ["1", 0],
                "positive":     ["6", 0],
                "negative":     ["4", 0],  # Flux ignores negative, reuse positive
                "latent_image": ["5", 0],
                "seed":         seed,
                "steps":        steps,
                "cfg":          1.0,
                "sampler_name": "euler",
                "scheduler":    "simple",
                "denoise":      1.0
            }
        },
        # Decode
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["7", 0],
                "vae":     ["2", 0]
            }
        },
        # Save
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "spatterjay",
                "images": ["8", 0]
            }
        }
    }


def submit_workflow(workflow: dict) -> str:
    """Submit workflow to ComfyUI, return prompt_id."""
    payload = {"prompt": workflow}
    r = requests.post(f"{COMFYUI_URL}/prompt", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["prompt_id"]


def wait_for_image(prompt_id: str, timeout: int = 600) -> str | None:
    """Poll ComfyUI history until image is ready, return filename."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10)
            r.raise_for_status()
            history = r.json()
            if prompt_id in history:
                outputs = history[prompt_id].get("outputs", {})
                for node_id, node_output in outputs.items():
                    if "images" in node_output:
                        img = node_output["images"][0]
                        return img["filename"], img.get("subfolder", ""), img.get("type", "output")
        except Exception:
            pass
        time.sleep(2)
    return None, None, None


def download_image(filename: str, subfolder: str, img_type: str, dest_path: Path):
    """Download generated image from ComfyUI and save locally."""
    params = f"filename={filename}&subfolder={subfolder}&type={img_type}"
    url = f"{COMFYUI_URL}/view?{params}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(r.content)


def generate_image(prompt: str, slug: str, output_dir: Path,
                   model_name: str, steps: int) -> Path | None:
    """Full pipeline: build workflow → submit → wait → download.
    Expects an already-enhanced prompt."""
    dest = output_dir / f"{slug}.png"
    seed = random.randint(0, 2**32 - 1)

    workflow = build_flux_workflow(
        prompt=prompt,
        model_name=model_name,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        steps=steps,
        seed=seed
    )

    try:
        prompt_id = submit_workflow(workflow)
    except Exception as e:
        print(f"    [ERROR] Submit failed: {e}")
        return None

    filename, subfolder, img_type = wait_for_image(prompt_id)
    if not filename:
        print(f"    [ERROR] Timed out waiting for image")
        return None

    try:
        download_image(filename, subfolder, img_type, dest)
        return dest
    except Exception as e:
        print(f"    [ERROR] Download failed: {e}")
        return None


# ── Main pipeline ──────────────────────────────────────────────────────────────

def load_progress(progress_file: Path) -> dict:
    if progress_file.exists():
        with open(progress_file) as f:
            return json.load(f)
    return {"generated": {}, "manifest": {}}


def save_progress(progress_file: Path, progress: dict):
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)


def run(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    progress_file = output_dir / "progress.json"
    manifest_file = output_dir / "manifest.json"
    image_list_file = output_dir / "image_list.json"

    # ── Load summaries ─────────────────────────────────────────────────────────
    summaries_path = Path(args.summaries)
    if not summaries_path.exists():
        print(f"[ERROR] Summaries file not found: {summaries_path}")
        sys.exit(1)

    with open(summaries_path) as f:
        summaries_text = f.read()
    print(f"[*] Loaded summaries: {len(summaries_text):,} chars")

    # ── Extract or load image list ─────────────────────────────────────────────
    if image_list_file.exists():
        with open(image_list_file) as f:
            items = json.load(f)
        print(f"[*] Loaded existing image list: {len(items)} items")
    else:
        items = extract_image_list(summaries_text)
        with open(image_list_file, "w") as f:
            json.dump(items, f, indent=2)
        print(f"[✓] Saved image list: {image_list_file}")

    if args.list_only:
        print("\n── Image List ──────────────────────────────────────────")
        for item in items:
            print(f"  [{item['type'].upper():8s}] {item['slug']}")
            print(f"             {item['prompt'][:80]}...")
        print(f"\nTotal: {len(items)} images")
        print(f"Edit {image_list_file} to adjust prompts before generating.")
        return

    # ── Check ComfyUI ─────────────────────────────────────────────────────────
    prefer_dev = (args.model == "flux-dev")
    model_name = get_flux_model_name(prefer_dev=prefer_dev)
    steps = FLUX_DEV_STEPS if "dev" in model_name.lower() else FLUX_SCHNELL_STEPS
    print(f"[*] Using model: {model_name} ({steps} steps)")

    # ── Generate images ────────────────────────────────────────────────────────
    progress = load_progress(progress_file)

    for idx, item in enumerate(items):
        slug   = item["slug"]
        prompt = item["prompt"]
        itype  = item["type"]
        dest   = output_dir / f"{slug}.png"

        if slug in progress["generated"]:
            print(f"[✓] {idx+1}/{len(items)}: {slug} (cached)")
            progress["manifest"][slug] = {
                "type": itype,
                "path": str(dest),
                "prompt": prompt
            }
            continue

        print(f"\n[→] {idx+1}/{len(items)}: {slug}")
        print(f"    {prompt[:90]}...")

        print(f"    [✦] Enhancing prompt via Ollama...")
        enhanced = enhance_prompt(f"{prompt}, {STYLE_SUFFIX}")
        print(f"    [✦] Enhanced: {enhanced[:90]}...")

        t0 = time.time()
        result = generate_image(enhanced, slug, output_dir, model_name, steps)
        elapsed = time.time() - t0

        if result:
            print(f"    [✓] Saved: {result} ({elapsed:.0f}s)")
            progress["generated"][slug] = str(result)
            progress["manifest"][slug] = {
                "type":             itype,
                "path":             str(result),
                "prompt":           prompt,
                "prompt_enhanced":  enhanced,
            }
        else:
            print(f"    [✗] Failed after {elapsed:.0f}s — skipping")

        save_progress(progress_file, progress)

    # ── Write final manifest ───────────────────────────────────────────────────
    with open(manifest_file, "w") as f:
        json.dump(progress["manifest"], f, indent=2)

    total = len(progress["generated"])
    print(f"\n[✓] Done! {total}/{len(items)} images generated")
    print(f"    Images:   {output_dir}/")
    print(f"    Manifest: {manifest_file}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pre-generate images for Spatterjay text adventure.")
    parser.add_argument("--summaries", default="all_chapter_summaries.md",
                        help="Path to chapter summaries markdown file")
    parser.add_argument("--output", default="images",
                        help="Output directory for images and manifest")
    parser.add_argument("--model", choices=["flux-schnell", "flux-dev"], default="flux-dev",
                        help="Which Flux model to use (default: flux-dev for pre-gen quality)")
    parser.add_argument("--list-only", action="store_true",
                        help="Only extract and display image list, don't generate")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
