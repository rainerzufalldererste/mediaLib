ProjectName = "TestApp"
project(ProjectName)

  --Settings
  kind "ConsoleApp"
  language "C++"
  flags { "StaticRuntime", "FatalWarnings" }
  dependson { "mediaLib" }

  buildoptions { '/Gm-' }
  buildoptions { '/MP' }

  filter { "configurations:Release" }
    flags { "LinkTimeOptimization" }

  filter {}
  defines { "_CRT_SECURE_NO_WARNINGS" }

  objdir "intermediate/obj"

  files { "src/**.cpp", "src/**.h", "src/**.inl" }
  files { "include/**.cpp", "include/**.h", "include/**.inl" }
  files { "project.lua" }

  includedirs { "../mediaLib/include" }
  includedirs { "3rdParty/SDL2/include" }

  filter { "configurations:Release" }
    links { "../mediaLib/lib/mediaLib.lib" }
  filter { "configurations:Debug" }
    links { "../mediaLib/lib/mediaLibD.lib" }
  
  filter { }

  links { "3rdParty/SDL2/lib/sdl2.lib" }
  links { "3rdParty/SDL2/lib/sdl2main.lib" }
  
  filter { "configurations:Debug", "system:Windows" }
    ignoredefaultlibraries { "libcmt" }
  filter { }
  
  configuration { }

  postbuildcommands { "{COPY} 3rdParty/SDL2/bin/SDL2.dll bin" } 
  
  targetname(ProjectName)
  targetdir "bin"
  debugdir "bin"
  
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
