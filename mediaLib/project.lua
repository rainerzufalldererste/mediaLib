ProjectName = "mediaLib"
project(ProjectName)

  --Settings
  kind "StaticLib"
  language "C++"
  flags { "StaticRuntime", "FatalWarnings" }

  filter { "system:windows" }
    buildoptions { '/Gm-' }
    buildoptions { '/MP' }
    disablewarnings  { '4127' } -- ignore conditional expression is constant

    ignoredefaultlibraries { "msvcrt" }
  
    defines { "_CRT_SECURE_NO_WARNINGS", "SSE2", "GLEW_STATIC" }
  
  filter { }
  
  objdir "intermediate/obj"

  files { "src/**.cpp", "src/**.h", "src/**.inl", "src/**rc" }
  files { "include/**.cpp", "include/**.h", "include/**.inl", "src/**rc" }
  files { "project.lua" }
  
  includedirs { "include" }
  includedirs { "include/**" }
  includedirs { "3rdParty" }
  includedirs { "3rdParty/DirectXMath/Inc" }
  includedirs { "3rdParty/DirectXMath/Extensions" }
  includedirs { "3rdParty/stb" }
  includedirs { "3rdParty/SDL2/include" }
  includedirs { "3rdParty/glew/include" }
  includedirs { "3rdParty/utf8proc/include" }
  includedirs { "3rdParty/imgui" }

  filter { "configurations:Debug", "system:Windows" }
    ignoredefaultlibraries { "libcmt" }
  filter { }
  
  targetname(ProjectName)
  targetdir "lib"
  debugdir "lib"
  
filter {}
configuration {}

linkoptions { "3rdParty/utf8proc/lib/utf8proc_static.lib" }
defines { "UTF8PROC_STATIC" }

linkoptions { "3rdParty/freetype-gl/lib/freetype-gl.lib" }
linkoptions { "3rdParty/freetype/lib/freetype.lib" }
linkoptions { "Shlwapi.lib" }

warnings "Extra"

filter {"configurations:Release"}
  targetname "%{prj.name}"
filter {"configurations:Debug"}
  targetname "%{prj.name}D"

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
	optimize "Speed"
	flags { "NoFramePointer", "NoBufferSecurityCheck" }
	symbols "On"
  editandcontinue "Off"

filter { "system:windows", "configurations:Release", "action:vs2012" }
	buildoptions { "/d2Zi+" }

filter { "system:windows", "configurations:Release", "action:vs2013" }
	buildoptions { "/Zo" }

filter { "system:windows", "configurations:Release" }
	flags { "NoIncrementalLink" }
