ProjectName = "mediaLibTest"
project(ProjectName)

  --Settings
  kind "ConsoleApp"
  language "C++"
  flags { "StaticRuntime", "FatalWarnings" }
  dependson { "mediaLib" }

  buildoptions { '/Gm-' }
  buildoptions { '/MP' }
  ignoredefaultlibraries { "msvcrt" }
  disablewarnings  { '4127' } -- ignore conditional expression is constant

  filter { "configurations:Release" }
    flags { "LinkTimeOptimization" }

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

if _OPTIONS["buildtype"] == "GIT_BUILD" then
  defines { "GIT_BUILD" }
else
  defines { "DEV_BUILD" }
end

configuration {}

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
  flags { "NoFramePointer", "NoBufferSecurityCheck" }
  symbols "On"

filter { "system:windows" }
	defines { "WIN32", "_WINDOWS" }
	links { "kernel32.lib", "user32.lib", "gdi32.lib", "winspool.lib", "comdlg32.lib", "advapi32.lib", "shell32.lib", "ole32.lib", "oleaut32.lib", "uuid.lib", "odbc32.lib", "odbccp32.lib" }

filter { "system:windows", "configurations:Release", "action:vs2012" }
	buildoptions { "/d2Zi+" }

filter { "system:windows", "configurations:Release", "action:vs2013" }
	buildoptions { "/Zo" }

filter { "system:windows", "configurations:Release" }
	flags { "NoIncrementalLink" }

filter {}
  flags { "NoFramePointer", "NoBufferSecurityCheck" }
