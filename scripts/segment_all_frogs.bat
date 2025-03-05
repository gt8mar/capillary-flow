@echo off
REM Segment All Frogs
REM This script runs the frog segmentation tool on all JPG images in the Downloads/whole-frog folder
REM 
REM Usage:
REM   scripts\segment_all_frogs.bat

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
set LOG_FILE=%CAPFLOW_PATH%\logs\frog_segmentation_%LOG_DATE%.log

echo Starting frog segmentation process at %date% %time% > %LOG_FILE%
echo Looking for JPG files in %DOWNLOAD_PATH% >> %LOG_FILE%

REM Count the number of JPG files
set NUM_FILES=0
for %%f in ("%DOWNLOAD_PATH%\*.JPG") do set /a NUM_FILES+=1

echo Found %NUM_FILES% JPG files to process >> %LOG_FILE%
echo Found %NUM_FILES% JPG files to process

REM Process each JPG file
set COUNTER=0
for %%f in ("%DOWNLOAD_PATH%\*.JPG") do (
    set /a COUNTER+=1
    set FILENAME=%%~nxf
    
    echo [!COUNTER!/%NUM_FILES%] Processing !FILENAME!... >> %LOG_FILE%
    echo [!COUNTER!/%NUM_FILES%] Processing !FILENAME!...
    
    REM Run the segmentation script
    python scripts/frog_segmentation.py "%%f"
    
    echo Completed !FILENAME! >> %LOG_FILE%
    echo ---------------------------------------- >> %LOG_FILE%
)

echo Segmentation process completed at %date% %time% >> %LOG_FILE%
echo Processed %COUNTER% files >> %LOG_FILE%
echo Log file saved to %LOG_FILE%

endlocal 