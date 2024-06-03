@ECHO OFF

PUSHD %~dp0

REM ModulationPy
git clone https://github.com/kirlf/ModulationPy

REM DUIDD
git clone https://github.com/IIP-Group/DUIDD

POPD

ECHO Done!
ECHO.

PAUSE
