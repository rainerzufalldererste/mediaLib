git clean -ffxd
git submodule foreach --recursive "git clean -ffdx"

git merge origin/master
if [ $? -ne 0 ]; then
  echo "merged with master failed"
  git merge --abort
  exit 1
fi

echo "merged successfully with master"

git submodule sync --recursive
git submodule update --init
if [ $? -ne 0 ]; then exit 1; fi

"premake/premake5.exe" vs2015 --buildtype=GIT_BUILD
if [ $? -ne 0 ]; then exit 1; fi

"C:/Program Files (x86)/MSBuild/14.0/Bin/amd64/MSBuild.exe" mediaLib.sln //p:Configuration=Debug //p:Platform="x64" //m:4 //v:m
if [ $? -ne 0 ]; then exit 1; fi

"./buildscripts/opencppcoverage/OpenCppCoverage.exe" --quiet --sources "mediaLib\src" --sources "mediaLib\include" --sources "mediaLibTest\src" --modules "mediaLib\lib\mediaLibD.lib" --modules "mediaLibTest\bin\mediaLibTest.exe" --working_dir "mediaLibTest\bin" --export_type=html:coverage -- "mediaLibTest\bin\mediaLibTest.exe"
if [ $? -ne 0 ]; then exit 1; fi

grep -m 1 ">Cover " "./coverage/index.html"
if [ $? -ne 0 ]; then exit 1; fi
