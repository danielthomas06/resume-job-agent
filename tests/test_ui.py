from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parents[1] / "src/jobagent/ui/app.py")


def test_ui_renders_without_resume():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value.startswith("🧭")
    assert at.button[0].disabled  # Find jobs needs an upload
