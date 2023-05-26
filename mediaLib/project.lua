ProjectName = "mediaLib"
project(ProjectName)

  --Settings
  kind "StaticLib"
  language "C++"
  flags { "FatalWarnings" }
  
  staticruntime "On"

  filter { "system:windows" }
    buildoptions { '/Gm-' }
    buildoptions { '/MP' }
    disablewarnings { '4127' } -- ignore conditional expression is constant

    ignoredefaultlibraries { "msvcrt" }
  
    defines { "_CRT_SECURE_NO_WARNINGS", "SSE2", "GLEW_STATIC" }
  filter { }

  filter { "system:windows", "action:vs2019 or vs202*" }
    cppdialect "C++17"
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
  includedirs { "3rdParty/imgui/include" }
  includedirs { "3rdParty/ogg/include" }
  includedirs { "3rdParty/opus/include" }
  includedirs { "3rdParty/libsamplerate/include" }
  includedirs { "3rdParty/StackWalker/include" }
  includedirs { "3rdParty/expat/include" }
  includedirs { "3rdParty/turbojpeg/include" }
  includedirs { "3rdParty/curl/include" }
  includedirs { "3rdParty/fpng/include" }
  includedirs { "3rdParty/freetype/include" }
  includedirs { "3rdParty/dragonbox/include" }

  if os.target() == "windows" then
    if os.getenv("CUDA_PATH") then -- for nvidia
      print("Initializing Project with Nvidia CUDA SDK OpenCL Headers.");
      includedirs { "$(CUDA_PATH)/include" }
    elseif os.getenv("AMDAPPSDKROOT") then -- for amd/radeon
      print("Initializing Project with AMD APP SDK OpenCL Headers.");
      includedirs { "$(AMDAPPSDKROOT)/include" }
    else
      print "Neither 'CUDA_PATH', nor 'AMDAPPSDKROOT' environment variable is defined. Using generic OpenCL headers from 3rdParty/OpenCL."
      includedirs { "3rdParty/OpenCL/include" } -- copied from `AMDAPPSDKROOT`
    end
  else
    includedirs { "3rdParty/OpenCL/include" } -- copied from `AMDAPPSDKROOT` on windows.
  end

  filter { "configurations:Debug", "system:Windows" }
    ignoredefaultlibraries { "libcmt" }
  filter { }

  configuration "avx2"
    vectorextensions "AVX2"
    defines { "AVX2", "AVX", "SSE42", "SSE41", "SSSE3", "SSE3" }
  configuration {}

  if _OPTIONS['asan'] then
    buildoptions { "/fsanitize=address" }
    flags { "NoIncrementalLink" }
    defines { "ASAN_ENABLED" }
    editandcontinue "Off"
    debugenvs { "PATH=$(VC_ExecutablePath_x64);%PATH%" }
    debugenvs { "ASAN_SYMBOLIZER_PATH=$(VC_ExecutablePath_x64)" }
    debugenvs { "ASAN_OPTIONS=verbosity=1:windows_hook_legacy_allocators=true" }
  end
  
  targetname(ProjectName)
  targetdir "lib"
  debugdir "lib"
  
filter {}
configuration {}

linkoptions { "3rdParty/utf8proc/lib/utf8proc_static.lib" }
defines { "UTF8PROC_STATIC" }

linkoptions { "3rdParty/SDL2/lib/SDL2.lib" }
linkoptions { "3rdParty/SDL2/lib/SDL2main.lib" }
linkoptions { "3rdParty/freetype/lib/freetype.lib" }
linkoptions { "3rdParty/ogg/lib/libogg_static.lib" }
linkoptions { "3rdParty/opus/lib/opus.lib" }
linkoptions { "3rdParty/opus/lib/opusfile.lib" }
linkoptions { "3rdParty/opus/lib/opusenc.lib" }
linkoptions { "3rdParty/libsamplerate/lib/samplerate.lib" }
linkoptions { "3rdParty/expat/lib/expat.lib" }
linkoptions { "3rdParty/turbojpeg/lib/turbojpeg-static.lib" }
linkoptions { "3rdParty/curl/lib/libcurl.lib" }
linkoptions { "Shlwapi.lib" }

filter {"configurations:Release"}
  linkoptions { "3rdParty/fpng/lib/fpng.lib" }
filter {"configurations:Debug"}
  linkoptions { "3rdParty/fpng/lib/fpngD.lib" }
filter { }

warnings "Extra"

filter {"configurations:Release"}
  targetname "%{prj.name}"
filter {"configurations:Debug"}
  targetname "%{prj.name}D"
filter { }

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
  flags { "NoBufferSecurityCheck" }
  omitframepointer "On"
  symbols "On"
  editandcontinue "Off"

filter { "system:windows", "configurations:Release" }
  flags { "NoIncrementalLink" }
