// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include "mUI.h"

#include "imgui.h"

#if defined(mRENDERER_OPENGL)
#define IMGUI_IMPL_OPENGL_LOADER_GLEW
#include "imgui/examples/imgui_impl_sdl.h"
//#include "imgui/examples/imgui_impl_opengl3.h"

#include "imgui/examples/imgui_impl_sdl.cpp"
#include "imgui/examples/imgui_impl_opengl3.cpp"
#include "imgui/imgui.cpp"
#include "imgui/imgui_demo.cpp"
#include "imgui/imgui_draw.cpp"
#include "imgui/imgui_widgets.cpp"
#endif

mFUNCTION(mUI_Initilialize, mPtr<mHardwareWindow> &hardwareWindow)
{
  mFUNCTION_SETUP();

  mERROR_IF(hardwareWindow == nullptr, mR_ArgumentNull);

  const char glsl_version[] = "#version 150"; IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO(); (void)io;

  SDL_Window *pWindow = nullptr;
  mERROR_CHECK(mHardwareWindow_GetSdlWindowPtr(hardwareWindow, &pWindow));

  SDL_GLContext glContext;
  mERROR_CHECK(mRenderParams_GetRenderContext(hardwareWindow, &glContext));

  ImGui_ImplSDL2_InitForOpenGL(pWindow, glContext);
  ImGui_ImplOpenGL3_Init(glsl_version);

  ImGui::StyleColorsDark();

  mRETURN_SUCCESS();
}

mFUNCTION(mUI_StartFrame, mPtr<mHardwareWindow> &hardwareWindow)
{
  mFUNCTION_SETUP();

  mERROR_IF(hardwareWindow == nullptr, mR_ArgumentNull);

  ImGui_ImplOpenGL3_NewFrame();

  SDL_Window *pWindow = nullptr;
  mERROR_CHECK(mHardwareWindow_GetSdlWindowPtr(hardwareWindow, &pWindow));

  ImGui_ImplSDL2_NewFrame(pWindow);
  ImGui::NewFrame();

  mRETURN_SUCCESS();
}

mFUNCTION(mUI_Shutdown)
{
  mFUNCTION_SETUP();

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplSDL2_Shutdown();
  ImGui::DestroyContext();

  mRETURN_SUCCESS();
}

mFUNCTION(mUI_Bake, mPtr<mHardwareWindow>& hardwareWindow)
{
  mFUNCTION_SETUP();

  mERROR_IF(hardwareWindow == nullptr, mR_ArgumentNull);

  bool open = true;
  ImGui::ShowDemoWindow(&open);

  ImGui::Render();

  mRenderParams_CurrentRenderContext = (mRenderContextId)-1;
  mERROR_CHECK(mHardwareWindow_SetAsActiveRenderTarget(hardwareWindow));

  mRETURN_SUCCESS();
}

mFUNCTION(mUI_Render)
{
  mFUNCTION_SETUP();

  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

  mRETURN_SUCCESS();
}
