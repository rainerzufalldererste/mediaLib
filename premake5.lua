
newoption {
  trigger     = "buildtype",
  description = "Enables build specific Macros for automated builds.",
  allowed     = {{ "GIT_BUILD", "defines GIT_BUILD" }, {"DEV_BUILD", "defines DEV_BUILD"}}
}

newoption {
  trigger     = "avx2",
  description = "Build with AVX2 code generation"
}

solution "mediaLib"

if not _OPTIONS["buildtype"] then
  _OPTIONS["buildtype"] = "DEV_BUILD"
end

  editorintegration "On"
  configurations { "Debug", "Release" }
  platforms { "x64" }

  dofile "mediaLib/project.lua"
    location("mediaLib")

  dofile "mediaLibTest/project.lua"
    location("mediaLibTest")
