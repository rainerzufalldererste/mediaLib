if [ "$1" == "Debug" ]; then
  "./buildscripts/opencppcoverage/OpenCppCoverage.exe" --quiet --sources "mediaLib\src" --sources "mediaLib\include" --sources "mediaLibTest\src" --modules "mediaLib\lib\mediaLibD.lib" --modules  "mediaLibTest\bin\mediaLibTest.exe" --working_dir "mediaLibTest\bin" --export_type=html:coverage -- "mediaLibTest\bin\mediaLibTest.exe"
  if [ $? -ne 0 ]; then exit 1; fi

  grep -m 1 ">Cover " "./coverage/index.html"
  if [ $? -ne 0 ]; then exit 1; fi

  rm -rf coverage
else
  cd mediaLibTest\\bin

  ./mediaLibTest.exe
  if [ $? -ne 0 ]; then exit 1; fi
fi
