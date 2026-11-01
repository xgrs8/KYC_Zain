"""Preload the Qt DLLs from PySide6's bundle before importing QtCore."""

import ctypes
import os
import pathlib
import sys


if sys.platform == "win32":
    _root = pathlib.Path(sys._MEIPASS)
    _shiboken = _root / "shiboken6"
    _pyside = _root / "PySide6"
    _system = pathlib.Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32"

    _dll_dirs = (_root, _shiboken, _pyside)
    _dll_dir_handles = [os.add_dll_directory(str(path)) for path in _dll_dirs if path.is_dir()]
    os.environ["PATH"] = os.pathsep.join(
        [str(_pyside), str(_shiboken), str(_root), str(_system), os.environ.get("PATH", "")]
    )

    _dlls = (
        _shiboken / "shiboken6.abi3.dll",
        _shiboken / "MSVCP140.dll",
        _shiboken / "VCRUNTIME140_1.dll",
        _pyside / "MSVCP140_1.dll",
        _pyside / "MSVCP140_2.dll",
        _system / "icuuc.dll",
        _pyside / "Qt6Core.dll",
        _pyside / "pyside6.abi3.dll",
    )
    for _dll in _dlls:
        if _dll.is_file():
            ctypes.WinDLL(str(_dll))
