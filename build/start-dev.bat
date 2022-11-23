@echo off
REM echo %~dp0 --Directory of the script
cd %~dp0
cd ..

docker run -it --rm^
    --workdir /src^
    -p 5000:5000^
    -v %CD%/src:/app^
    -v%CD%:/src^
    -v%CD%/.vscode/extensions:/root/.vscode-server/extensions^
    glados-tts:devel /bin/bash