@echo off
setlocal
if "%~1"=="" (
  echo Usage: copy_runtime.cmd ^<OutDir^>
  exit /b 1
)
set "DEST=%~1"
set "ROOT=%~dp0.."
if not exist "%DEST%" mkdir "%DEST%"

copy /Y "%ROOT%\onnxruntime\lib\onnxruntime.dll" "%DEST%" >nul || exit /b 1
copy /Y "%ROOT%\model\tiny_cnn_fp32.onnx" "%DEST%" >nul || exit /b 1
copy /Y "%ROOT%\data\input.bin" "%DEST%" >nul || exit /b 1
copy /Y "%ROOT%\data\golden.bin" "%DEST%" >nul || exit /b 1
echo Experiment 08 runtime files copied to %DEST%
