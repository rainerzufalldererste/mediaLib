#include "mUI.h"

#include "mShader.h"
#include "mProfiler.h"

#if defined(mRENDERER_OPENGL)
#define IMGUI_IMPL_OPENGL_LOADER_GLEW
#define DECLSPEC
#include "imgui/include/backends/imgui_impl_sdl.h"
#include "imgui/include/backends/imgui_impl_opengl3.h"
#include "imgui/include/imgui_internal.h"
#undef DECLSPEC
#endif

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "FwItiCW9M4ymsopGJPX/Jho263tGovZ5P67oWtztNtdM6FLCPBYXOciUDwgZ6OKErj+1WGdP5jWXBdur"
#endif

ImGuiIO *mUI_pImguiIO = nullptr;
ImFont *pFont = nullptr;
ImFont *pHeadline = nullptr;
ImFont *pSlimHeadline = nullptr;
ImFont *pSubHeadline = nullptr;
ImFont *pMonospacedFont = nullptr;
ImFont *pBold = nullptr;

bool mUI_AutoUpdateMousePosition = true;
bool mUI_FirstFrameStarted = false;

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mUI_InitializeActiveContext, mPtr<mHardwareWindow> &hardwareWindow, const bool addUpdateCallback)
{
  mFUNCTION_SETUP();

  const char glsl_version[] = "#version 150";

  SDL_Window *pWindow = nullptr;
  mERROR_CHECK(mHardwareWindow_GetSdlWindowPtr(hardwareWindow, &pWindow));

  SDL_GLContext glContext;
  mERROR_CHECK(mRenderParams_GetRenderContext(hardwareWindow, &glContext));

  ImGui_ImplSDL2_InitForOpenGL(pWindow, glContext);
  ImGui_ImplOpenGL3_Init(glsl_version);

  if (addUpdateCallback)
    mERROR_CHECK(mHardwareWindow_AddOnAnyEvent(hardwareWindow, mUI_ProcessEvent));

  mRETURN_SUCCESS();
}

mFUNCTION(mUI_Initilialize, mPtr<mHardwareWindow> &hardwareWindow, const bool addUpdateCallback /* = true */)
{
  mFUNCTION_SETUP();

  mERROR_IF(hardwareWindow == nullptr, mR_ArgumentNull);

  IMGUI_CHECKVERSION();

  ImGui::CreateContext();

  mERROR_CHECK(mUI_InitializeActiveContext(hardwareWindow, addUpdateCallback));

  mUI_pImguiIO = &ImGui::GetIO();

  ImGui::StyleColorsLight();

  pFont = mUI_pImguiIO->Fonts->AddFontFromFileTTF("C:/Windows/Fonts/segoeui.ttf", 17.0f, nullptr, mUI_pImguiIO->Fonts->GetGlyphRangesDefault());
  mERROR_IF(pFont == nullptr, mR_InternalError);

  pMonospacedFont = mUI_pImguiIO->Fonts->AddFontFromFileTTF("C:/Windows/Fonts/consola.ttf", 13.0f, nullptr, mUI_pImguiIO->Fonts->GetGlyphRangesDefault());

  pHeadline = mUI_pImguiIO->Fonts->AddFontFromFileTTF("C:/Windows/Fonts/seguibl.ttf", 52.0f, nullptr, mUI_pImguiIO->Fonts->GetGlyphRangesDefault());
  pSlimHeadline = mUI_pImguiIO->Fonts->AddFontFromFileTTF("C:/Windows/Fonts/segoeuisl.ttf", 42.0f, nullptr, mUI_pImguiIO->Fonts->GetGlyphRangesDefault());
  pSubHeadline = mUI_pImguiIO->Fonts->AddFontFromFileTTF("C:/Windows/Fonts/seguisb.ttf", 28.0f, nullptr, mUI_pImguiIO->Fonts->GetGlyphRangesDefault());
  pBold = mUI_pImguiIO->Fonts->AddFontFromFileTTF("C:/Windows/Fonts/seguisb.ttf", 18.0f, nullptr, mUI_pImguiIO->Fonts->GetGlyphRangesDefault());

  ImGui::GetStyle().WindowRounding = 10.f;

  mERROR_CHECK(mUI_SetLightColourScheme());

  mRETURN_SUCCESS();
}

mFUNCTION(mUI_SetIniFilePath, const mString &path)
{
  mFUNCTION_SETUP();

  mERROR_IF(mUI_pImguiIO == nullptr, mR_ResourceStateInvalid);
  mERROR_IF(path.bytes <= 1, mR_InvalidParameter);
  mERROR_IF(mUI_FirstFrameStarted, mR_ResourceStateInvalid);

  static char iniFilePath[MAX_PATH + 1] = "";
  mERROR_IF(path.bytes > sizeof(iniFilePath), mR_ArgumentOutOfBounds);

  mERROR_CHECK(mMemcpy(iniFilePath, path.c_str(), path.bytes));

  mUI_pImguiIO->IniFilename = iniFilePath;

  mRETURN_SUCCESS();
}

mFUNCTION(mUI_StartFrame, mPtr<mHardwareWindow> &hardwareWindow)
{
  mFUNCTION_SETUP();

  mPROFILE_SCOPED("mUI_StartFrame");

  mERROR_IF(mUI_pImguiIO == nullptr, mR_ResourceStateInvalid);
  mERROR_IF(hardwareWindow == nullptr, mR_ArgumentNull);

  mUI_FirstFrameStarted = true;

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

  mUI_pImguiIO = nullptr;

  mRETURN_SUCCESS();
}

mFUNCTION(mUI_Bake)
{
  mFUNCTION_SETUP();

  mPROFILE_SCOPED("mUI_Bake");

  mERROR_IF(mUI_pImguiIO == nullptr, mR_ResourceStateInvalid);

  ImGui::Render();

  mRETURN_SUCCESS();
}

mFUNCTION(mUI_Render)
{
  mFUNCTION_SETUP();

  mPROFILE_SCOPED("mUI_Render");

  mERROR_IF(mUI_pImguiIO == nullptr, mR_ResourceStateInvalid);

  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

  mERROR_CHECK(mShader_AfterForeign());

  mRETURN_SUCCESS();
}

mFUNCTION(mUI_ProcessEvent, IN SDL_Event *pEvent)
{
  mFUNCTION_SETUP();

  mERROR_IF(pEvent == nullptr, mR_ArgumentNull);
  mERROR_IF(mUI_pImguiIO == nullptr, mR_ResourceStateInvalid);

  ImGui_ImplSDL2_ProcessEvent(pEvent);

  mRETURN_SUCCESS();
}

mFUNCTION(mUI_GetIO, OUT ImGuiIO **ppIO)
{
  mFUNCTION_SETUP();

  mERROR_IF(mUI_pImguiIO == nullptr, mR_ResourceStateInvalid);
  mERROR_IF(ppIO == nullptr, mR_ArgumentNull);

  *ppIO = &ImGui::GetIO();

  mRETURN_SUCCESS();
}

mFUNCTION(mUI_SetCustomMousePosition, const mVec2f position)
{
  mFUNCTION_SETUP();

  mERROR_IF(mUI_pImguiIO == nullptr, mR_ResourceStateInvalid);

  ImGuiIO *pIO = nullptr;
  mERROR_CHECK(mUI_GetIO(&pIO));

  pIO->MousePos = position;

  mRETURN_SUCCESS();
}

mFUNCTION(mUI_ManuallyUpdateMousePosition, const bool enable)
{
  mFUNCTION_SETUP();

  mUI_AutoUpdateMousePosition = !enable;

  mRETURN_SUCCESS();
}

mFUNCTION(mUI_PushMonospacedFont)
{
  mFUNCTION_SETUP();

  mERROR_IF(mUI_pImguiIO == nullptr, mR_ResourceStateInvalid);

  if (pMonospacedFont != nullptr)
    ImGui::PushFont(pMonospacedFont);

  mRETURN_SUCCESS();
}

mFUNCTION(mUI_PopMonospacedFont)
{
  mFUNCTION_SETUP();

  mERROR_IF(mUI_pImguiIO == nullptr, mR_ResourceStateInvalid);

  if (pMonospacedFont != nullptr)
    ImGui::PopFont();

  mRETURN_SUCCESS();
}

mFUNCTION(mUI_PushHeadlineFont)
{
  mFUNCTION_SETUP();

  mERROR_IF(mUI_pImguiIO == nullptr, mR_ResourceStateInvalid);

  if (pHeadline != nullptr)
    ImGui::PushFont(pHeadline);

  mRETURN_SUCCESS();
}

mFUNCTION(mUI_PopHeadlineFont)
{
  mFUNCTION_SETUP();

  mERROR_IF(mUI_pImguiIO == nullptr, mR_ResourceStateInvalid);

  if (pHeadline != nullptr)
    ImGui::PopFont();

  mRETURN_SUCCESS();
}

mFUNCTION(mUI_PushSlimHeadlineFont)
{
  mFUNCTION_SETUP();

  mERROR_IF(mUI_pImguiIO == nullptr, mR_ResourceStateInvalid);

  if (pSlimHeadline != nullptr)
    ImGui::PushFont(pSlimHeadline);

  mRETURN_SUCCESS();
}

mFUNCTION(mUI_PopSlimHeadlineFont)
{
  mFUNCTION_SETUP();

  mERROR_IF(mUI_pImguiIO == nullptr, mR_ResourceStateInvalid);

  if (pSlimHeadline != nullptr)
    ImGui::PopFont();

  mRETURN_SUCCESS();
}

mFUNCTION(mUI_PushSubHeadlineFont)
{
  mFUNCTION_SETUP();

  mERROR_IF(mUI_pImguiIO == nullptr, mR_ResourceStateInvalid);

  if (pSubHeadline != nullptr)
    ImGui::PushFont(pSubHeadline);

  mRETURN_SUCCESS();
}

mFUNCTION(mUI_PopSubHeadlineFont)
{
  mFUNCTION_SETUP();

  mERROR_IF(mUI_pImguiIO == nullptr, mR_ResourceStateInvalid);

  if (pSubHeadline != nullptr)
    ImGui::PopFont();

  mRETURN_SUCCESS();
}

mFUNCTION(mUI_PushBoldFont)
{
  mFUNCTION_SETUP();

  mERROR_IF(mUI_pImguiIO == nullptr, mR_ResourceStateInvalid);

  if (pBold != nullptr)
    ImGui::PushFont(pBold);

  mRETURN_SUCCESS();
}

mFUNCTION(mUI_PopBoldFont)
{
  mFUNCTION_SETUP();

  mERROR_IF(mUI_pImguiIO == nullptr, mR_ResourceStateInvalid);

  if (pBold != nullptr)
    ImGui::PopFont();

  mRETURN_SUCCESS();
}

mFUNCTION(mUI_SetLightColourScheme)
{
  ImVec4 *pColors = ImGui::GetStyle().Colors;

  pColors[ImGuiCol_Text] = ImVec4(0.12f, 0.12f, 0.12f, 1.00f);
  pColors[ImGuiCol_TextDisabled] = ImVec4(0.60f, 0.60f, 0.60f, 1.00f);
  pColors[ImGuiCol_WindowBg] = ImVec4(0.94f, 0.94f, 0.94f, 1.00f);
  pColors[ImGuiCol_ChildBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
  pColors[ImGuiCol_PopupBg] = ImVec4(0.98f, 0.98f, 0.98f, 1);
  pColors[ImGuiCol_Border] = ImVec4(0.00f, 0.00f, 0.00f, 0.30f);
  pColors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
  pColors[ImGuiCol_FrameBg] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
  pColors[ImGuiCol_FrameBgHovered] = ImVec4(0.830f, 0.830f, 0.830f, 0.400f);
  pColors[ImGuiCol_FrameBgActive] = ImVec4(0.900f, 0.900f, 0.900f, 0.670f);
  pColors[ImGuiCol_TitleBg] = ImVec4(0.96f, 0.96f, 0.96f, 1.00f);
  pColors[ImGuiCol_TitleBgActive] = ImVec4(0.82f, 0.82f, 0.82f, 1.00f);
  pColors[ImGuiCol_TitleBgCollapsed] = ImVec4(1.00f, 1.00f, 1.00f, 0.51f);
  pColors[ImGuiCol_MenuBarBg] = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
  pColors[ImGuiCol_ScrollbarBg] = ImVec4(0.98f, 0.98f, 0.98f, 0.00f);
  pColors[ImGuiCol_ScrollbarGrab] = ImVec4(0.80f, 0.80f, 0.80f, 0.80f);
  pColors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.75f, 0.75f, 0.75f, 0.80f);
  pColors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.58f, 0.58f, 0.58f, 1.00f);
  pColors[ImGuiCol_CheckMark] = ImVec4(0.16f, 0.16f, 0.16f, 1.00f);
  pColors[ImGuiCol_SliderGrab] = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
  pColors[ImGuiCol_SliderGrabActive] = ImVec4(0.47f, 0.47f, 0.47f, 1.00f);
  pColors[ImGuiCol_Button] = ImVec4(0.28f, 0.28f, 0.28f, 0.11f);
  pColors[ImGuiCol_ButtonHovered] = ImVec4(0.24f, 0.24f, 0.24f, 0.15f);
  pColors[ImGuiCol_ButtonActive] = ImVec4(0.16f, 0.16f, 0.16f, 0.22f);
  pColors[ImGuiCol_Header] = ImVec4(0.46f, 0.46f, 0.46f, 0.31f);
  pColors[ImGuiCol_HeaderHovered] = ImVec4(0.63f, 0.63f, 0.63f, 0.80f);
  pColors[ImGuiCol_HeaderActive] = ImVec4(0.82f, 0.82f, 0.82f, 1.00f);
  pColors[ImGuiCol_Separator] = ImVec4(0.70f, 0.70f, 0.70f, 1.00f);
  pColors[ImGuiCol_SeparatorHovered] = ImVec4(0.70f, 0.70f, 0.70f, 1.00f);
  pColors[ImGuiCol_SeparatorActive] = ImVec4(0.70f, 0.70f, 0.70f, 1.00f);
  pColors[ImGuiCol_ResizeGrip] = ImVec4(0.80f, 0.80f, 0.80f, 0.56f);
  pColors[ImGuiCol_ResizeGripHovered] = ImVec4(0.59f, 0.59f, 0.59f, 0.67f);
  pColors[ImGuiCol_ResizeGripActive] = ImVec4(0.37f, 0.37f, 0.37f, 0.95f);
  pColors[ImGuiCol_PlotLines] = ImVec4(0.39f, 0.39f, 0.39f, 1.00f);
  pColors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
  pColors[ImGuiCol_PlotHistogram] = ImVec4(1.00f, 0.52f, 0.28f, 1.00f);
  pColors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.45f, 0.00f, 1.00f);
  pColors[ImGuiCol_TextSelectedBg] = ImVec4(0.61f, 0.61f, 0.61f, 0.35f);
  pColors[ImGuiCol_DragDropTarget] = ImVec4(0.33f, 0.33f, 0.33f, 0.95f);
  pColors[ImGuiCol_NavHighlight] = ImVec4(0.81f, 0.81f, 0.81f, 0.80f);
  pColors[ImGuiCol_NavWindowingHighlight] = ImVec4(0.70f, 0.70f, 0.70f, 0.70f);
  pColors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.20f, 0.20f, 0.20f, 0.20f);
  pColors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.20f, 0.20f, 0.20f, 0.35f);

  return mR_Success;
}

mFUNCTION(mUI_SetDarkColourScheme)
{
  ImVec4 *pColors = ImGui::GetStyle().Colors;

  pColors[ImGuiCol_Text] = ImVec4(0.88f, 0.88f, 0.88f, 1.00f);
  pColors[ImGuiCol_TextDisabled] = ImVec4(0.42f, 0.42f, 0.42f, 1.00f);
  pColors[ImGuiCol_WindowBg] = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
  pColors[ImGuiCol_ChildBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
  pColors[ImGuiCol_PopupBg] = ImVec4(0.22f, 0.22f, 0.22f, 1);
  pColors[ImGuiCol_Border] = ImVec4(0.403f, 0.403f, 0.403f, 0.276f);
  pColors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
  pColors[ImGuiCol_FrameBg] = ImVec4(0.22f, 0.22f, 0.22f, 1.00f);
  pColors[ImGuiCol_FrameBgHovered] = ImVec4(0.407f, 0.407f, 0.407f, 0.400f);
  pColors[ImGuiCol_FrameBgActive] = ImVec4(0.389f, 0.389f, 0.389f, 0.670f);
  pColors[ImGuiCol_TitleBg] = ImVec4(0.18f, 0.18f, 0.18f, 1.00f);
  pColors[ImGuiCol_TitleBgActive] = ImVec4(0.11f, 0.11f, 0.11f, 1.00f);
  pColors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.18f, 0.18f, 0.18f, 1.00f);
  pColors[ImGuiCol_MenuBarBg] = ImVec4(0.11f, 0.11f, 0.11f, 1.00f);
  pColors[ImGuiCol_ScrollbarBg] = ImVec4(0.98f, 0.98f, 0.98f, 0.00f);
  pColors[ImGuiCol_ScrollbarGrab] = ImVec4(0.33f, 0.33f, 0.33f, 0.80f);
  pColors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.43f, 0.43f, 0.43f, 0.80f);
  pColors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.53f, 0.53f, 0.53f, 1.00f);
  pColors[ImGuiCol_CheckMark] = ImVec4(0.45f, 0.45f, 0.45f, 1.00f);
  pColors[ImGuiCol_SliderGrab] = ImVec4(0.65f, 0.65f, 0.65f, 1.00f);
  pColors[ImGuiCol_SliderGrabActive] = ImVec4(0.86f, 0.86f, 0.86f, 1.00f);
  pColors[ImGuiCol_Button] = ImVec4(0.65f, 0.65f, 0.65f, 0.11f);
  pColors[ImGuiCol_ButtonHovered] = ImVec4(0.68f, 0.68f, 0.68f, 0.15f);
  pColors[ImGuiCol_ButtonActive] = ImVec4(0.70f, 0.70f, 0.70f, 0.22f);
  pColors[ImGuiCol_Header] = ImVec4(0.37f, 0.37f, 0.37f, 0.31f);
  pColors[ImGuiCol_HeaderHovered] = ImVec4(0.30f, 0.30f, 0.30f, 0.80f);
  pColors[ImGuiCol_HeaderActive] = ImVec4(0.35f, 0.35f, 0.35f, 1.00f);
  pColors[ImGuiCol_Separator] = ImVec4(0.29f, 0.29f, 0.29f, 1.00f);
  pColors[ImGuiCol_SeparatorHovered] = ImVec4(0.42f, 0.42f, 0.42f, 1.00f);
  pColors[ImGuiCol_SeparatorActive] = ImVec4(0.52f, 0.52f, 0.52f, 1.00f);
  pColors[ImGuiCol_ResizeGrip] = ImVec4(0.43f, 0.43f, 0.43f, 0.56f);
  pColors[ImGuiCol_ResizeGripHovered] = ImVec4(0.61f, 0.61f, 0.61f, 0.67f);
  pColors[ImGuiCol_ResizeGripActive] = ImVec4(0.53f, 0.53f, 0.53f, 0.95f);
  pColors[ImGuiCol_PlotLines] = ImVec4(0.39f, 0.39f, 0.39f, 1.00f);
  pColors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
  pColors[ImGuiCol_PlotHistogram] = ImVec4(0.86f, 0.42f, 0.25f, 1.00f);
  pColors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.45f, 0.00f, 1.00f);
  pColors[ImGuiCol_TextSelectedBg] = ImVec4(0.61f, 0.61f, 0.61f, 0.35f);
  pColors[ImGuiCol_DragDropTarget] = ImVec4(0.33f, 0.33f, 0.33f, 0.95f);
  pColors[ImGuiCol_NavHighlight] = ImVec4(0.81f, 0.81f, 0.81f, 0.80f);
  pColors[ImGuiCol_NavWindowingHighlight] = ImVec4(0.70f, 0.70f, 0.70f, 0.70f);
  pColors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.20f, 0.20f, 0.20f, 0.20f);
  pColors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.20f, 0.20f, 0.20f, 0.35f);

  return mR_Success;
}

mFUNCTION(mUI_SetDpiScalingFactor, const float_t scalingFactor)
{
  mFUNCTION_SETUP();

  mERROR_IF(mUI_pImguiIO == nullptr, mR_ResourceStateInvalid);

  ImGui::GetIO().FontGlobalScale = scalingFactor;

  mRETURN_SUCCESS();
}
