from __future__ import annotations

import sys
from pathlib import Path


server_files_course = Path(__file__).resolve().parents[2]
if str(server_files_course) not in sys.path:
    sys.path.insert(0, str(server_files_course))
