"""Resource paths for the A.L.G app (repo or PyInstaller-frozen).

ALG freezes with only its own data (no OpenCV/reportlab - D1):
fonts live inside the vendored rfalib package (bundled automatically);
the app icon is the user-provided blue ribbon.
"""
from __future__ import annotations

import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent  # .../ALG


def icon_file() -> Path:
    """The user-provided blue ribbon app icon (packaging/icons/ALG.ico)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable)
    return APP_ROOT / "packaging" / "icons" / "ALG.ico"
