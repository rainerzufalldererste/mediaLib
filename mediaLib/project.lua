ProjectName = "mediaLib"
project(ProjectName)

  --Settings
  kind "StaticLib"
  language "C++"
  flags { "StaticRuntime", "FatalWarnings" }

  buildoptions { '/Gm-' }
  buildoptions { '/MP' }
  buildoptions { '/MT' }
  flags { "LinkTimeOptimization" }
  
  defines { "_CRT_SECURE_NO_WARNINGS" }
  
  objdir "intermediate/obj"

  files { "src/**.cpp", "src/**.h", "src/**.inl", "src/**rc" }
  files { "include/**.cpp", "include/**.h", "include/**.inl", "src/**rc" }
  files { "project.lua" }
  
  includedirs { "src" }
  includedirs { "include" }
  
  filter { "configurations:Debug", "system:Windows" }
    ignoredefaultlibraries { "libcmt" }
  filter { }
  
  targetname(ProjectName)
  targetdir "lib"
  debugdir "lib"
  
filter {}
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

filter { "configurations:DebugOpt*" }
	defines { "_DEBUG" }
	optimize "Debug"
	symbols "On"

filter { "configurations:Release*" }
	defines { "NDEBUG" }
	optimize "Full"
	flags { "NoFramePointer", "NoBufferSecurityCheck" }

filter { "configurations:Release*" }
	symbols "On"

filter { "system:windows", "platforms:x86" }
	vectorextensions "SSE2"

filter { "system:windows", "configurations:Release", "action:vs2012" }
	buildoptions { "/d2Zi+" }

filter { "system:windows", "configurations:Release", "action:vs2013" }
	buildoptions { "/Zo" }

filter { "system:windows", "configurations:Release" }
	flags { "NoIncrementalLink" }

filter { "system:windows", "platforms:x64", "configurations:Debug" }
	flags { "NoIncrementalLink" }

filter {}
