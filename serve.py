#!/usr/bin/env python3
"""
Local game server.
Serves game files, lists chapter files, and proxies Ollama API calls.

Usage:
    python serve.py          # runs on port 8080
    python serve.py 9000     # runs on custom port
"""

import http.server
import json
import os
import sys
import urllib.request
import urllib.error
import webbrowser
from pathlib import Path
from threading import Timer


PORT = int(sys.argv[1]) if len(sys.argv) > 1 else None   # resolved after config load


def default_port(config: dict) -> int:
    """Extract port from config.server.url, fallback to 8080."""
    try:
        url = config["server"]["url"]          # e.g. "http://localhost:8080"
        return int(url.rstrip("/").rsplit(":", 1)[-1])
    except Exception:
        return 8080
GAME_DIR = Path(__file__).parent.resolve()


def load_config() -> dict:
    config_path = GAME_DIR / "config.json"
    if not config_path.exists():
        print("[ERROR] config.json not found")
        sys.exit(1)
    with open(config_path) as f:
        return json.load(f)


class GameHandler(http.server.SimpleHTTPRequestHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(GAME_DIR), **kwargs)

    # ── GET ────────────────────────────────────────────────────────────────────

    def do_GET(self):
        if self.path == "/api/chapters":
            self.handle_chapters()
        else:
            super().do_GET()

    def handle_chapters(self):
        """Return sorted list of chapter markdown files."""
        config = load_config()
        chapters_dir = (GAME_DIR / config["book"]["chapters_dir"]).resolve()

        if not chapters_dir.exists():
            self.json_error(404, f"chapters_dir not found: {chapters_dir}")
            return

        files = sorted(
            str(f.relative_to(GAME_DIR))
            for f in chapters_dir.glob("ch_*.md")
        )

        self.json_response({"chapters": files, "count": len(files)})

    # ── POST ───────────────────────────────────────────────────────────────────

    def do_POST(self):
        if self.path.startswith("/ollama/"):
            self.handle_ollama_proxy()
        else:
            self.send_response(404)
            self.end_headers()

    def handle_ollama_proxy(self):
        """Proxy POST requests to Ollama, stripping the /ollama prefix."""
        config = load_config()
        ollama_base = config["ollama"]["url"].rsplit("/api/", 1)[0]  # http://localhost:11434
        ollama_path = self.path[len("/ollama"):]                     # /api/chat

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            req = urllib.request.Request(
                f"{ollama_base}{ollama_path}",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                response_body = resp.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(response_body)

        except urllib.error.URLError as e:
            self.json_error(502, f"Ollama unreachable: {e.reason}")
        except Exception as e:
            self.json_error(500, str(e))

    # ── OPTIONS (CORS preflight) ───────────────────────────────────────────────

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    # ── Helpers ────────────────────────────────────────────────────────────────

    def json_response(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def json_error(self, status: int, message: str):
        self.json_response({"error": message}, status)

    def log_message(self, fmt, *args):
        # Only log errors, not every 200/304
        if len(args) >= 2 and args[1] not in ("200", "304"):
            super().log_message(fmt, *args)


def main():
    os.chdir(GAME_DIR)
    config = load_config()
    title  = config["book"]["title"]

    port = PORT if PORT is not None else default_port(config)

    print(f"[*] {title} — Game Server")
    print(f"[*] Serving from: {GAME_DIR}")
    print(f"[*] http://localhost:{port}/game.html")
    print(f"[*] Press Ctrl+C to stop\n")

    Timer(1.2, lambda: webbrowser.open(f"http://localhost:{port}/game.html")).start()

    with http.server.HTTPServer(("", port), GameHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[*] Server stopped")


if __name__ == "__main__":
    main()
