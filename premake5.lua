solution "mediaLib"
  
  editorintegration "On"
  configurations { "Debug", "Release" }
  platforms { "x64" }

  dofile "mediaLib/project.lua"
    location("mediaLib")

  dofile "mediaLibTest/project.lua"
    location("mediaLibTest")

  dofile "TestApp/project.lua"
    location("TestApp")
