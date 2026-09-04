"""PyInstaller runtime hook: point Playwright at the bundled Chromium.

In one-file mode PyInstaller unpacks bundled data to sys._MEIPASS at runtime.
The build bundles the ms-playwright browser folder there; this hook makes
Playwright look inside it instead of the user's %LOCALAPPDATA%.
"""
import os
import sys

_meipass = getattr(sys, "_MEIPASS", None)
if _meipass:
    bundled = os.path.join(_meipass, "ms-playwright")
    if os.path.isdir(bundled):
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = bundled
