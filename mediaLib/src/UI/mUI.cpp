// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#include "mUI.h"

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

ImGuiIO mUI_ImguiIO;
ImFont *pFont = nullptr;
ImFont *pHeadline = nullptr;
ImFont *pMonospacedFont = nullptr;

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mUI_Initilialize, mPtr<mHardwareWindow> &hardwareWindow)
{
  mFUNCTION_SETUP();

  mERROR_IF(hardwareWindow == nullptr, mR_ArgumentNull);

  const char glsl_version[] = "#version 150"; IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  mUI_ImguiIO = ImGui::GetIO();

  SDL_Window *pWindow = nullptr;
  mERROR_CHECK(mHardwareWindow_GetSdlWindowPtr(hardwareWindow, &pWindow));

  SDL_GLContext glContext;
  mERROR_CHECK(mRenderParams_GetRenderContext(hardwareWindow, &glContext));

  ImGui_ImplSDL2_InitForOpenGL(pWindow, glContext);
  ImGui_ImplOpenGL3_Init(glsl_version);

  ImGui::StyleColorsLight();

  pFont = mUI_ImguiIO.Fonts->AddFontFromFileTTF("C:/Windows/Fonts/segoeui.ttf", 17.0f);
  mERROR_IF(pFont == nullptr, mR_InternalError);

  pMonospacedFont = mUI_ImguiIO.Fonts->AddFontFromFileTTF("C:/Windows/Fonts/consola.ttf", 13.0f);

  pHeadline = mUI_ImguiIO.Fonts->AddFontFromFileTTF("C:/Windows/Fonts/seguibl.ttf", 52.0f);

  mERROR_CHECK(mHardwareWindow_AddOnAnyEvent(hardwareWindow, mUI_ProcessEvent));

  ImVec4 *pColors = ImGui::GetStyle().Colors;
  pColors[ImGuiCol_Text] = ImVec4(0.12f, 0.12f, 0.12f, 1.00f);
  pColors[ImGuiCol_TextDisabled] = ImVec4(0.60f, 0.60f, 0.60f, 1.00f);
  pColors[ImGuiCol_WindowBg] = ImVec4(0.94f, 0.94f, 0.94f, 1.00f);
  pColors[ImGuiCol_ChildBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
  pColors[ImGuiCol_PopupBg] = ImVec4(1.00f, 1.00f, 1.00f, 0.98f);
  pColors[ImGuiCol_Border] = ImVec4(0.00f, 0.00f, 0.00f, 0.30f);
  pColors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
  pColors[ImGuiCol_FrameBg] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
  pColors[ImGuiCol_FrameBgHovered] = ImVec4(0.83f, 0.83f, 0.83f, 0.40f);
  pColors[ImGuiCol_FrameBgActive] = ImVec4(0.90f, 0.90f, 0.90f, 0.67f);
  pColors[ImGuiCol_TitleBg] = ImVec4(0.96f, 0.96f, 0.96f, 1.00f);
  pColors[ImGuiCol_TitleBgActive] = ImVec4(0.82f, 0.82f, 0.82f, 1.00f);
  pColors[ImGuiCol_TitleBgCollapsed] = ImVec4(1.00f, 1.00f, 1.00f, 0.51f);
  pColors[ImGuiCol_MenuBarBg] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
  pColors[ImGuiCol_ScrollbarBg] = ImVec4(0.98f, 0.98f, 0.98f, 0.00f);
  pColors[ImGuiCol_ScrollbarGrab] = ImVec4(0.80f, 0.80f, 0.80f, 0.80f);
  pColors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.75f, 0.75f, 0.75f, 0.80f);
  pColors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.58f, 0.58f, 0.58f, 1.00f);
  pColors[ImGuiCol_CheckMark] = ImVec4(0.16f, 0.16f, 0.16f, 1.00f);
  pColors[ImGuiCol_SliderGrab] = ImVec4(0.25f, 0.25f, 0.25f, 0.78f);
  pColors[ImGuiCol_SliderGrabActive] = ImVec4(0.47f, 0.47f, 0.47f, 0.60f);
  pColors[ImGuiCol_Button] = ImVec4(0.87f, 0.87f, 0.87f, 1.00f);
  pColors[ImGuiCol_ButtonHovered] = ImVec4(0.85f, 0.85f, 0.85f, 1.00f);
  pColors[ImGuiCol_ButtonActive] = ImVec4(0.75f, 0.75f, 0.75f, 1.00f);
  pColors[ImGuiCol_Header] = ImVec4(0.46f, 0.46f, 0.46f, 0.31f);
  pColors[ImGuiCol_HeaderHovered] = ImVec4(0.63f, 0.63f, 0.63f, 0.80f);
  pColors[ImGuiCol_HeaderActive] = ImVec4(0.82f, 0.82f, 0.82f, 1.00f);
  pColors[ImGuiCol_Separator] = ImVec4(0.39f, 0.39f, 0.39f, 1.00f);
  pColors[ImGuiCol_SeparatorHovered] = ImVec4(0.39f, 0.39f, 0.39f, 1.00f);
  pColors[ImGuiCol_SeparatorActive] = ImVec4(0.39f, 0.39f, 0.39f, 1.00f);
  pColors[ImGuiCol_ResizeGrip] = ImVec4(0.80f, 0.80f, 0.80f, 0.56f);
  pColors[ImGuiCol_ResizeGripHovered] = ImVec4(0.59f, 0.59f, 0.59f, 0.67f);
  pColors[ImGuiCol_ResizeGripActive] = ImVec4(0.37f, 0.37f, 0.37f, 0.95f);
  pColors[ImGuiCol_PlotLines] = ImVec4(0.39f, 0.39f, 0.39f, 1.00f);
  pColors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
  pColors[ImGuiCol_PlotHistogram] = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
  pColors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.45f, 0.00f, 1.00f);
  pColors[ImGuiCol_TextSelectedBg] = ImVec4(0.61f, 0.61f, 0.61f, 0.35f);
  pColors[ImGuiCol_DragDropTarget] = ImVec4(0.33f, 0.33f, 0.33f, 0.95f);
  pColors[ImGuiCol_NavHighlight] = ImVec4(0.81f, 0.81f, 0.81f, 0.80f);
  pColors[ImGuiCol_NavWindowingHighlight] = ImVec4(0.70f, 0.70f, 0.70f, 0.70f);
  pColors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.20f, 0.20f, 0.20f, 0.20f);
  pColors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.20f, 0.20f, 0.20f, 0.35f);

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

mFUNCTION(mUI_ProcessEvent, IN SDL_Event *pEvent)
{
  mFUNCTION_SETUP();

  ImGui_ImplSDL2_ProcessEvent(pEvent);

  mRETURN_SUCCESS();
}

mFUNCTION(mUI_PushMonospacedFont)
{
  mFUNCTION_SETUP();

  if (pMonospacedFont != nullptr)
    ImGui::PushFont(pMonospacedFont);

  mRETURN_SUCCESS();
}

mFUNCTION(mUI_PopMonospacedFont)
{
  mFUNCTION_SETUP();

  if (pMonospacedFont != nullptr)
    ImGui::PopFont();

  mRETURN_SUCCESS();
}

mFUNCTION(mUI_PushHeadlineFont)
{
  mFUNCTION_SETUP();

  if (pHeadline != nullptr)
    ImGui::PushFont(pHeadline);

  mRETURN_SUCCESS();
}

mFUNCTION(mUI_PopHeadlineFont)
{
  mFUNCTION_SETUP();

  if (pHeadline != nullptr)
    ImGui::PopFont();

  mRETURN_SUCCESS();
}
