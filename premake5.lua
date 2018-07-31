solution "mediaLib"
  
  editorintegration "On"
  configurations { "Debug", "Release" }
  platforms { "x64" }

  dofile "mediaLib/project.lua"
    location("mediaLib")

    dofile "TestApp/project.lua"
      location("TestApp")