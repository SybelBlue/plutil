from __future__ import annotations

import pytest

from plutil.magic.decorator import clip_plmagic_tracebacks


@pytest.fixture(autouse=True, scope="session")
def show_plmagic_internals_during_tests():
    """Keep complete tracebacks while testing Plmagic itself."""
    token = clip_plmagic_tracebacks.set(False)
    yield
    clip_plmagic_tracebacks.reset(token)


# server_files_course = Path(__file__).resolve().parents[2]
# if str(server_files_course) not in sys.path:
#     sys.path.insert(0, str(server_files_course))
