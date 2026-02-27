@echo off
REM ───────────────────────────────────────────
REM  WebScraper Pro — PyInstaller build script
REM  Produces a single-file Windows EXE
REM ───────────────────────────────────────────

echo [1/3] Installing PyInstaller …
pip install pyinstaller

echo [2/3] Building EXE …
pyinstaller --onefile ^
            --windowed ^
            --name "WebScraperPro" ^
            --add-data "config/config.yaml;config" ^
            gui/app.py

echo [3/3] Done!
echo.
echo   EXE created in:  dist\WebScraperPro.exe
echo.
pause
