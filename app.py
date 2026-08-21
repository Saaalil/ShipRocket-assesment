from __future__ import annotations

import os
import sys
from pathlib import Path

# Hugging Face Spaces force Gradio SSR by default (Node proxy on :7861).
# That experimental path leaves asyncio event loops that die with
# "Invalid file descriptor: -1" during cleanup. Disable SSR before Gradio loads.
os.environ["GRADIO_SSR_MODE"] = "0"
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

_ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(_ROOT), str(_ROOT / "src")]

from demo.gradio_app import build_demo

demo = build_demo()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", os.environ.get("PORT", 7860))),
        share=False,
        ssr_mode=False,
    )
