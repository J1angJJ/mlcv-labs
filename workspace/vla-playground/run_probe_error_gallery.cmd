@echo off
chcp 65001 > nul
setlocal

set "PROJECT_ROOT=R:\mlcv-labs\workspace\vla-playground"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

cd /d "%PROJECT_ROOT%"
"C:\Users\JJ406\.conda\envs\cv-train\python.exe" experiments\build_probe_error_gallery.py ^
  --probe-dir outputs\clip_labeled_300_probe ^
  --output-html outputs\clip_labeled_300_probe\error_gallery.html ^
  --method linear

endlocal
