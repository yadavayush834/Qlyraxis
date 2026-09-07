# PyInstaller build definition for the Qlyraxis desktop and CLI application.

from pathlib import Path
import sys


project_root = Path(SPECPATH).parent
python_library_root = Path(sys.base_prefix) / "lib"

a = Analysis(
    [str(project_root / "src" / "qlyraxis" / "__main__.py")],
    pathex=[str(project_root / "src")],
    binaries=[
        (str(python_library_root / "libtcl9.0.so"), "."),
        (str(python_library_root / "libtcl9tk9.0.so"), "."),
    ],
    datas=[
        (str(project_root / "configs"), "configs"),
        (str(project_root / "models"), "models"),
        (str(project_root / "docs"), "docs"),
    ],
    hiddenimports=["tkinter", "tkinter.filedialog", "tkinter.messagebox"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["onnx", "onnxruntime"],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Qlyraxis",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Qlyraxis",
)
