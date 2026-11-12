"""GATE Block: Block 1 of the KYC document pipeline.

Checks that an upload is one of the supported documents, merges the two sides of the national ID, measures
the photo's quality, lets Laya choose between accept / enhance / reject, validates that choice against the
measurements, and routes the photo to TeleOCR, to DocRes, back to the user, or to a person.

    from gate_block import Gate
    gate = Gate()                                   # templates/ and config/ next to the package
    result = gate.process_id("front.jpg", "back.jpg", workdir="out")
    print(result.route, result.status_code, result.message_ar)
"""
from .config import ConfigError, GateConfig
from .imageio import ImageLoadError
from .laya_client import QUESTIONS, BackendError, LayaBackend, StaticBackend, onehot
from .pipeline import Gate, GateResult
from .templates import TemplateError

__version__ = "1.0.0"
__all__ = ["Gate", "GateResult", "GateConfig", "ConfigError", "TemplateError", "ImageLoadError", "LayaBackend",
           "StaticBackend", "BackendError", "QUESTIONS", "onehot", "__version__"]
