# PyInstaller spec for agent.exe -- build with:  pyinstaller agent.spec
# Build from a venv that has only requirements.txt (+ pyinstaller) installed, so
# torch and the training stack are not pulled into the exe.
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

datas = [
    ("agent/models/food_classifier.onnx", "agent/models"),
    ("agent/data/nutrition_facts.json", "agent/data"),
    (".env.example", "."),
]
datas += collect_data_files("certifi")
datas += collect_data_files("langchain_core")

hiddenimports = (
    collect_submodules("langgraph")
    + collect_submodules("langchain_openai")
    + collect_submodules("onnxruntime")
    + ["tiktoken_ext.openai_public", "tiktoken_ext"]
)
binaries = collect_dynamic_libs("onnxruntime")

a = Analysis(
    ["agent_main.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["torch", "torchvision", "matplotlib", "seaborn", "pandas", "sklearn", "cv2", "IPython", "jupyter", "notebook", "tkinter", "pytest"],
    noarchive=False,
)
# onnxruntime's native module needs a recent MSVC runtime (14.3x+). PyInstaller
# can pick up an old msvcp140.dll / vcruntime140.dll (14.16, shipped by some
# wheel), and onnxruntime then crashes with an access violation on import.
# Bundle the build machine's current System32 copies instead.
import os

_RUNTIME = {"msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "vcruntime140.dll", "vcruntime140_1.dll", "concrt140.dll"}
_system32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
a.binaries = [b for b in a.binaries if os.path.basename(b[0]).lower() not in _RUNTIME]
for _dll in sorted(_RUNTIME):
    _src = os.path.join(_system32, _dll)
    if os.path.exists(_src):
        a.binaries.append((_dll, _src, "BINARY"))

pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="agent",
    console=True,
    upx=False,
    icon=None,
)
