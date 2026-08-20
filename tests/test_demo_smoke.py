from __future__ import annotations

import pytest


def test_gradio_app_builds() -> None:
    pytest.importorskip("gradio")
    from demo.gradio_app import build_demo

    demo = build_demo()
    assert demo is not None
