from __future__ import annotations

import pytest

gr = pytest.importorskip("gradio")

from demo.gradio_app import build_demo


def test_gradio_app_builds() -> None:
    demo = build_demo()
    assert demo is not None
