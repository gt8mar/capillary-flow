@echo off
REM Analyze All Frogs
REM This script runs the frog color analysis tool on all JPG and CR2 images

setlocal enabledelayedexpansion

REM Get the hostname
for /f "tokens=*" %%a in ('hostname') do set HOSTNAME=%%a

REM Set paths based on hostname
if "%HOSTNAME%"=="LAPTOP-I5KTBOR3" (
    set CAPFLOW_PATH=C:\Users\gt8ma\capillary-flow
    set DOWNLOAD_PATH=C:\Users\gt8ma\Downloads\whole-frog
) else if "%HOSTNAME%"=="Quake-Blood" (
    set CAPFLOW_PATH=C:\Users\gt8mar\capillary-flow
    set DOWNLOAD_PATH=C:\Users\gt8mar\Downloads\whole-frog
) else (
    set CAPFLOW_PATH=\hpc\projects\capillary-flow
    set DOWNLOAD_PATH=\home\downloads\whole-frog
)

REM Create log directory if it doesn't exist
if not exist "%CAPFLOW_PATH%\logs" mkdir "%CAPFLOW_PATH%\logs"

REM Create timestamp for log file
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set datetime=%%a
set LOG_DATE=%datetime:~0,8%_%datetime:~8,6%
set LOG_FILE=%CAPFLOW_PATH%\logs\frog_analysis_%LOG_DATE%.log

echo Starting frog analysis process at %date% %time% > %LOG_FILE%
echo Looking for image files in %DOWNLOAD_PATH% >> %LOG_FILE%

REM Count the number of image files (both JPG and CR2)
set NUM_FILES=0
for %%f in ("%DOWNLOAD_PATH%\*.JPG" "%DOWNLOAD_PATH%\*.jpg" "%DOWNLOAD_PATH%\*.CR2" "%DOWNLOAD_PATH%\*.cr2") do set /a NUM_FILES+=1

echo Found %NUM_FILES% image files to process >> %LOG_FILE%
echo Found %NUM_FILES% image files to process

REM Check if rawpy is installed
python -c "import rawpy" 2>nul
if errorlevel 1 (
    echo ERROR: rawpy module is not installed. Please install it using: >> %LOG_FILE%
    echo ERROR: rawpy module is not installed. Please install it using:
    echo pip install rawpy >> %LOG_FILE%
    echo pip install rawpy
    exit /b 1
)

REM Process each image file
set COUNTER=0
for %%f in ("%DOWNLOAD_PATH%\*.JPG" "%DOWNLOAD_PATH%\*.jpg" "%DOWNLOAD_PATH%\*.CR2" "%DOWNLOAD_PATH%\*.cr2") do (
    set /a COUNTER+=1
    set FILENAME=%%~nxf
    
    echo [!COUNTER!/%NUM_FILES%] Processing !FILENAME!... >> %LOG_FILE%
    echo [!COUNTER!/%NUM_FILES%] Processing !FILENAME!...
    
    REM Run the analysis script
    python -m src.tools.frog_total_color "!FILENAME!" >> %LOG_FILE% 2>&1
    
    echo Completed !FILENAME! >> %LOG_FILE%
    echo ---------------------------------------- >> %LOG_FILE%
)

echo Analysis process completed at %date% %time% >> %LOG_FILE%
echo Processed %COUNTER% files >> %LOG_FILE%
echo Log file saved to %LOG_FILE%

endlocal 