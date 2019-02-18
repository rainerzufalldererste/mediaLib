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

  configuration "avx2"
    vectorextensions "AVX2"
    defines { "AVX2", "AVX", "SSE42", "SSE41", "SSSE3", "SSE3" }
  configuration {}
  
  targetname(ProjectName)
  targetdir "lib"
  debugdir "lib"
  
filter {}
configuration {}

linkoptions { "3rdParty/utf8proc/lib/utf8proc_static.lib" }
defines { "UTF8PROC_STATIC" }

linkoptions { "3rdParty/SDL2/lib/SDL2.lib" }
linkoptions { "3rdParty/SDL2/lib/SDL2main.lib" }
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
