@echo off

@REM CD'ing to root

cd %~dp0
cd ..

docker build -t glados-tts . -f .\build\Dockerfile
@REM docker build -t glados-tts:devel . -f .\build\Dockerfile-dev
