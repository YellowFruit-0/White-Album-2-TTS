set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%"
set "PATH=%SCRIPT_DIR%\runtime;%PATH%"
echo "White Album 2 TTS Project by 黄水果天下第一"
echo "关于server.py的详细配置参数见下，可按照需要修改 start_server.bat 的内容进行配置。"
runtime\python.exe server.py -h
runtime\python.exe server.py --base_path %~dp0 --character "kaz" --no_translate --no_recognition
pause