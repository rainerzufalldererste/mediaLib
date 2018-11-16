git submodule foreach --recursive "git clean -ffdx"

git merge origin/master
IF %ERRORLEVEL% GEQ 1 (
  echo "merged with master failed"
  git merge --abort
  exit 1;
)
echo "merged successfully with master"

setlocal
call :GetUnixTime UNIXTIME

git submodule sync --recursive
git submodule update --init
"premake/premake5" vs2015 --buildtype=GIT_BUILD
"C:\Program Files (x86)\MSBuild\14.0\Bin\amd64\MSBuild.exe" mediaLib.sln /p:Configuration="%~1" /p:Platform="x64" /p:BuildTag="GIT_BRANCH=%CI_BUILD_REF_NAME%;GIT_BUILD=%CI_PIPELINE_ID%;GIT_REF=%CI_BUILD_REF%;BUILD_TIME=%UNIXTIME%" /m:4 /v:m

IF %ERRORLEVEL% GEQ 1 (exit 1)

"buildscripts/opencppcoverage/OpenCppCoverage.exe" --quiet --sources "mediaLib" --sources "mediaLibTest\src" --sources "mediaLibTest\include" --modules "mediaLib\lib\mediaLibD.lib" --modules "mediaLibTest\bin\mediaLibTest.exe" --working_dir "mediaLibTest\bin" --export_type=html:coverage -- "mediaLibTest\bin\mediaLibTest.exe"
IF %ERRORLEVEL% GEQ 1 (exit 1)

type "coverage\index.html"
IF %ERRORLEVEL% GEQ 1 (exit 1)

:skip_copy
goto :EOF

:GetUnixTime
setlocal enableextensions
for /f %%x in ('wmic path win32_utctime get /format:list ^| findstr "="') do (set %%x)
set /a z=(14-100%Month%%%100)/12, y=10000%Year%%%10000-z
set /a ut=y*365+y/4-y/100+y/400+(153*(100%Month%%%100+12*z-3)+2)/5+Day-719469
set /a ut=ut*86400+100%Hour%%%100*3600+100%Minute%%%100*60+100%Second%%%100
endlocal & set "%1=%ut%" & goto :EOF




