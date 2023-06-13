git clean -ffdx
git submodule foreach --recursive "git clean -ffdx"
git fetch origin

git merge origin/master
IF %ERRORLEVEL% GEQ 1 (
  echo "merged with master failed"
  git merge --abort
  exit 1;
)
echo "merged successfully with master"

git submodule sync --recursive
git submodule update --init
"premake/premake5" vs2019 --buildtype=GIT_BUILD
"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\MSBuild\Current\Bin\MSBuild.exe" mediaLib.sln /p:Configuration="%~1" /p:Platform="x64" /p:BuildTag="GIT_BRANCH=%CI_BUILD_REF_NAME%;GIT_BUILD=%CI_PIPELINE_ID%;GIT_REF=%CI_BUILD_REF%;BUILD_TIME=%UNIXTIME%" /m:4 /v:m
IF %ERRORLEVEL% GEQ 1 (exit 1)
