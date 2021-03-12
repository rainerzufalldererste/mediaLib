ProjectName = "mediaLibTest"
project(ProjectName)

  --Settings
  kind "ConsoleApp"
  language "C++"
  flags { "FatalWarnings" }
  dependson { "mediaLib" }

  staticruntime "On"

  buildoptions { '/Gm-' }
  buildoptions { '/MP' }
  ignoredefaultlibraries { "msvcrt" }
  disablewarnings  { '4127' } -- ignore conditional expression is constant

  filter {}
  defines { "_CRT_SECURE_NO_WARNINGS", "SSE2" }

  objdir "intermediate/obj"

  files { "src/**.cpp", "src/**.h", "src/**.inl" }
  files { "project.lua" }

  includedirs { "src**" }
  includedirs { "../mediaLib/include/**" }
  includedirs { "../mediaLib/include" }
  includedirs { "3rdParty/**" }

  filter { "configurations:Release" }
    links { "../mediaLib/lib/mediaLib.lib" }
    links { "3rdParty/gtest/lib/gtest.lib" }
  filter { "configurations:Debug" }
    links { "../mediaLib/lib/mediaLibD.lib" }
    links { "3rdParty/gtest/lib/gtestd.lib" }
  
  filter { }
  
  filter { "configurations:Debug", "system:Windows" }
    ignoredefaultlibraries { "libcmt" }
  filter { }
  
  configuration { }
  
  targetname(ProjectName)
  targetdir "bin"
  debugdir "bin"
  
filter {}
configuration {}

-- Strings
if os.getenv("CI_BUILD_REF_NAME") then
  defines { "GIT_BRANCH=\"" .. os.getenv("CI_BUILD_REF_NAME") .. "\"" }
end
if os.getenv("CI_COMMIT_SHA") then
  defines { "GIT_REF=\"" .. os.getenv("CI_COMMIT_SHA") .. "\"" }
end

-- Numbers
if os.getenv("CI_PIPELINE_ID") then
  defines { "GIT_BUILD=" .. os.getenv("CI_PIPELINE_ID") }
else
  defines { "DEV_BUILD" }
end
if os.getenv("UNIXTIME") then
  defines { "BUILD_TIME=" .. os.getenv("UNIXTIME") }
end

warnings "Extra"

targetname "%{prj.name}"

flags { "NoMinimalRebuild", "NoPCH" }
exceptionhandling "Off"
rtti "Off"
floatingpoint "Fast"

filter { "configurations:Debug*" }
  defines { "_DEBUG" }
  optimize "Off"
  symbols "On"

filter { "configurations:Release" }
  defines { "NDEBUG" }
  optimize "Full"
  flags { "NoBufferSecurityCheck" }
  omitframepointer "On"
  symbols "On"
  editandcontinue "Off"

filter { "system:windows" }
	defines { "WIN32", "_WINDOWS" }
	links { "kernel32.lib", "user32.lib", "gdi32.lib", "winspool.lib", "comdlg32.lib", "advapi32.lib", "shell32.lib", "ole32.lib", "oleaut32.lib", "uuid.lib", "odbc32.lib", "odbccp32.lib", "winmm.lib", "setupapi.lib", "version.lib", "Imm32.lib" }

filter { "system:windows", "configurations:Release" }
	flags { "NoIncrementalLink" }
