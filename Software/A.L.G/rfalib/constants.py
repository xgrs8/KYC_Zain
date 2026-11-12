"""Constants for A.L.G and its supporting library.

Keep this module dependency-free (stdlib only).
"""
from __future__ import annotations

import pathlib
import sys

# ---------------------------------------------------------------- versions
RFALIB_VERSION = "0.1.0"
APP_VERSION = "0.1.0"

# ---------------------------------------------------------------- programs
APP_NAME_ALG = "A.L.G - Automatic Label Generator"
APP_TITLE_ALG = "Automatic Label Generator"

# ---------------------------------------------------------------- doc types
DOC_TYPES = ("ID", "TAX", "LIC")
DOC_TYPE_LABELS = {
    "ID": "National ID card",
    "TAX": "Tax card",
    "LIC": "Business license",
}
DOC_TYPE_ARABIC = {
    "ID": "بطاقة الهوية الوطنية",
    "TAX": "البطاقة الضريبية",
    "LIC": "رخصة العمل / شهادة التأسيس",
}

# ---------------------------------------------------------------- tokens
UNREADABLE = "[UNREADABLE]"
FIELD_STATUS = ("typed", "accepted-truth", "edited", "unreadable")
QUEUE_STATES = ("Waiting", "In progress", "Done", "Skipped")
CONDITION_TAGS = ("glare", "blur", "low-res", "other")

# ---------------------------------------------------------------- naming
DOC_PREFIX = {"ID": "ID", "TAX": "TAX", "LIC": "LIC"}
PAD_WIDTH = 2  # ID_01 .. ID_99, unpadded from 100 (ID_100)

# ---------------------------------------------------------------- dates
DATE_FORMAT = "%Y/%m/%d"  # Western digits, YYYY/MM/DD
DATE_MIN = {"ID": "2016-01-01", "TAX": "2015-01-01", "LIC": "2015-01-01"}
VALIDITY_YEARS = {"ID": 10, "TAX": 5, "LIC": 5}  # expiry = issue + Ny - 1d
AGE_MIN, AGE_MAX = 18, 85  # plausible age at issue

# ---------------------------------------------------------------- merge (D11)
MERGE_DEFAULT_WIDTH = 1600  # px; = the project dataset and the GATE Block (main/GATE Block/config/id_merge.yaml); configurable 1000-1600
MERGE_MIN_WIDTH = 1000
MERGE_MAX_WIDTH = 1600
MERGE_DEFAULT_GAP = 24  # px white gap between front and back

# ---------------------------------------------------------------- person / face gate (D6)
FACE_SCALE_FACTOR = 1.08
FACE_MIN_NEIGHBORS = 5
FACE_MIN_SIZE = 24

# ---------------------------------------------------------------- synthetic marking (D7)
SYN_PREFIX = "SYN-"
SYN_DIGITS = 6

# ---------------------------------------------------------------- split buckets (D5)
SPLIT_RATIOS = (80, 10, 10)  # train / val / test
SPLITS = ("train", "val", "test")

# ---------------------------------------------------------------- workspace
def default_workspace() -> pathlib.Path:
    """%USERPROFILE%\\Documents\\RFALG on Windows; ~/Documents/RFALG elsewhere."""
    if sys.platform == "win32":
        base = pathlib.Path.home() / "Documents"
    else:
        base = pathlib.Path.home() / "Documents"
    return base / "RFALG"


WORKSPACE_SUBDIRS = (
    "templates",   # the user's three template source images
    "generated",   # generated images
    "labeled",     # A.L.G outputs
    "print",       # print/<batch>/... PDFs
    "quarantine",  # samples that failed the face gate
)
WORKSPACE_FILES = ("state.json", "settings.json", "alg_queue.json")

# ---------------------------------------------------------------- settings defaults
DEFAULT_SETTINGS = {
    "workspace": None,             # str or None -> default_workspace()
    "merge_width": MERGE_DEFAULT_WIDTH,
    "merge_gap": MERGE_DEFAULT_GAP,
    "jpeg_option": False,          # also write .jpg q95 next to the .png
    "synthetic_banner": False,     # diagonal SPECIMEN watermark instead of SYN code
    "case_mode": False,            # D4: default OFF
    "defects": False,              # D4: default OFF
}

# ---------------------------------------------------------------- MRZ (TD1)
MRZ_LINE_LEN = 30
MRZ_DOC_CODE = "ID"
MRZ_STATE = "IRQ"  # Iraq
MRZ_NATIONALITY = "IRQ"

# Arabic -> Latin transliteration (ASSUMED; align with ICAO 9303-3 - plan D3/M4)
TRANSLIT_TABLE = {
    "ا": "A", "ب": "B", "ت": "T", "ث": "TH", "ج": "J", "ح": "H", "خ": "KH",
    "د": "D", "ذ": "DH", "ر": "R", "ز": "Z", "س": "S", "ش": "SH", "ص": "S",
    "ض": "D", "ط": "T", "ظ": "Z", "ع": "A", "غ": "GH", "ف": "F", "ق": "Q",
    "ك": "K", "ل": "L", "م": "M", "ن": "N", "ه": "H", "و": "W", "ي": "Y",
    "ى": "Y", "ة": "H", "ء": "", "ؤ": "W", "إ": "A", "أ": "A", "آ": "A",
    "ئ": "Y", "لا": "LA",
}

# ---------------------------------------------------------------- pools
POOL_FILES = (
    "given_m.txt", "given_f.txt", "surnames.txt", "companies.txt",
    "activities.txt", "purposes.txt", "cities.txt", "areas.txt",
    "streets.txt", "authorities.txt", "company_types.txt",
)
