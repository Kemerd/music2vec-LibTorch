@echo off
echo Running Music2Vec model conversion script...
echo.

:: First, try to install the requirements
echo Installing required packages with compatible versions...
pip install -r requirements.txt || (
    echo Failed to install with pip, trying with pip3...
    pip3 install -r requirements.txt
)

if %ERRORLEVEL% neq 0 (
    echo.
    echo Failed to install dependencies. Please check your Python installation.
    pause
    exit /b 1
)

:: Try to use Python or Python3 command
echo Running conversion script...
python convert.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo Conversion failed! There was an error during the conversion process.
    echo.
    echo If you're still having dependency issues, try creating a new conda environment:
    echo conda create -n mixmate python=3.10
    echo conda activate mixmate
    echo pip install -r requirements.txt
    echo python convert.py
    pause
    exit /b 1
)

echo.
echo Conversion completed! Press any key to exit...
pause 