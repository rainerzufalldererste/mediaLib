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
#include "imgui/imgui_internal.h"
#endif

ImGuiIO mUI_ImguiIO;
ImFont *pFont = nullptr;
ImFont *pHeadline = nullptr;
ImFont *pSubHeadline = nullptr;
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

  pFont = mUI_ImguiIO.Fonts->AddFontFromFileTTF("C:/Windows/Fonts/segoeui.ttf", 17.0f, nullptr, mUI_ImguiIO.Fonts->GetGlyphRangesDefault());
  mERROR_IF(pFont == nullptr, mR_InternalError);

  pMonospacedFont = mUI_ImguiIO.Fonts->AddFontFromFileTTF("C:/Windows/Fonts/consola.ttf", 13.0f, nullptr, mUI_ImguiIO.Fonts->GetGlyphRangesDefault());

  pHeadline = mUI_ImguiIO.Fonts->AddFontFromFileTTF("C:/Windows/Fonts/seguibl.ttf", 52.0f, nullptr, mUI_ImguiIO.Fonts->GetGlyphRangesDefault());

  pSubHeadline = mUI_ImguiIO.Fonts->AddFontFromFileTTF("C:/Windows/Fonts/seguisb.ttf", 28.0f, nullptr, mUI_ImguiIO.Fonts->GetGlyphRangesDefault());

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
  pColors[ImGuiCol_Button] = ImVec4(0.28f, 0.28f, 0.28f, 0.11f);
  pColors[ImGuiCol_ButtonHovered] = ImVec4(0.24f, 0.24f, 0.24f, 0.15f);
  pColors[ImGuiCol_ButtonActive] = ImVec4(0.16f, 0.16f, 0.16f, 0.22f);
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

mFUNCTION(mUI_Bake)
{
  mFUNCTION_SETUP();

  ImGui::Render();

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

mFUNCTION(mUI_GetIO, OUT ImGuiIO **ppIO)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppIO == nullptr, mR_ArgumentNull);

  *ppIO = &mUI_ImguiIO;

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

mFUNCTION(mUI_PushSubHeadlineFont)
{
  mFUNCTION_SETUP();

  if (pSubHeadline != nullptr)
    ImGui::PushFont(pSubHeadline);

  mRETURN_SUCCESS();
}

mFUNCTION(mUI_PopSubHeadlineFont)
{
  mFUNCTION_SETUP();

  if (pSubHeadline != nullptr)
    ImGui::PopFont();

  mRETURN_SUCCESS();
}

namespace ImGui
{
  // See: https://github.com/ocornut/imgui/issues/1901
  bool BufferingBar(const char *label, float_t value, const ImVec2 &size_arg, const ImU32 &bg_col, const ImU32 &fg_col)
  {
    ImGuiWindow* window = GetCurrentWindow();
    if (window->SkipItems)
      return false;

    ImGuiContext& g = *GImGui;
    const ImGuiStyle& style = g.Style;
    const ImGuiID id = window->GetID(label);

    ImVec2 pos = window->DC.CursorPos;
    ImVec2 size = size_arg;
    size.x -= style.FramePadding.x * 2;

    const ImRect bb(pos, ImVec2(pos.x + size.x, pos.y + size.y));
    ItemSize(bb, style.FramePadding.y);

    if (!ItemAdd(bb, id))
      return false;

    // Render
    const double_t circleStart = size.x * 0.7f;
    const double_t circleEnd = size.x;
    const double_t circleWidth = circleEnd - circleStart;

    window->DrawList->AddRectFilled(bb.Min, ImVec2(pos.x + (float_t)circleStart, bb.Max.y), bg_col);
    window->DrawList->AddRectFilled(bb.Min, ImVec2(pos.x + (float_t)circleStart * value, bb.Max.y), fg_col);

    const double_t t = g.Time;
    const double_t r = size.y / 2;
    const double_t speed = 1.5f;

    const double_t a = speed * 0;
    const double_t b = speed * 0.333f;
    const double_t c = speed * 0.666f;

    const double_t o1 = (circleWidth + r) * (t + a - speed * (int32_t)((t + a) / speed)) / speed;
    const double_t o2 = (circleWidth + r) * (t + b - speed * (int32_t)((t + b) / speed)) / speed;
    const double_t o3 = (circleWidth + r) * (t + c - speed * (int32_t)((t + c) / speed)) / speed;

    window->DrawList->AddCircleFilled(ImVec2(pos.x + (float_t)circleEnd - (float_t)o1, bb.Min.y + (float_t)r), (float_t)r, bg_col);
    window->DrawList->AddCircleFilled(ImVec2(pos.x + (float_t)circleEnd - (float_t)o2, bb.Min.y + (float_t)r), (float_t)r, bg_col);
    window->DrawList->AddCircleFilled(ImVec2(pos.x + (float_t)circleEnd - (float_t)o3, bb.Min.y + (float_t)r), (float_t)r, bg_col);
    
    return true;
  }

  // See: https://github.com/ocornut/imgui/issues/1901
  bool Spinner(const char *label, float_t radius, float_t thickness, const ImU32 &color)
  {
    ImGuiWindow* window = GetCurrentWindow();
    if (window->SkipItems)
      return false;

    ImGuiContext& g = *GImGui;
    const ImGuiStyle& style = g.Style;
    const ImGuiID id = window->GetID(label);

    ImVec2 pos = window->DC.CursorPos;
    ImVec2 size((radius) * 2, (radius + style.FramePadding.y) * 2);

    const ImRect bb(pos, ImVec2(pos.x + size.x, pos.y + size.y));
    ItemSize(bb, style.FramePadding.y);

    if (!ItemAdd(bb, id))
      return false;

    // Render
    window->DrawList->PathClear();

    int32_t num_segments = 30;
    int32_t start = (int32_t)abs(ImSin((float_t)g.Time * 1.8f) * (num_segments - 5));

    const float_t a_min = mTWOPIf * ((float_t)start) / (float_t)num_segments;
    const float_t a_max = mTWOPIf * ((float_t)num_segments - 3) / (float_t)num_segments;

    const ImVec2 centre = ImVec2(pos.x + radius, pos.y + radius + style.FramePadding.y);

    for (int i = 0; i < num_segments; i++)
    {
      const float_t a = a_min + ((float_t)i / (float_t)num_segments) * (a_max - a_min);

      window->DrawList->PathLineTo(ImVec2(centre.x + ImCos(a + (float_t)g.Time * 8) * radius, centre.y + ImSin(a + (float_t)g.Time * 8) * radius));
    }

    window->DrawList->PathStroke(color, false, thickness);

    return true;
  }

  size_t TabBar(const std::initializer_list<const char *> &names, const size_t activeBefore)
  {
    size_t returnedIndex = activeBefore;
    
    if (returnedIndex >= names.size())
      returnedIndex = 0;

    size_t index = 0;

    for (auto &name : names)
    {
      if (returnedIndex == index)
      {
        ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyle().Colors[ImGuiCol_ChildBg]);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImGui::GetStyle().Colors[ImGuiCol_ChildBg]);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImGui::GetStyle().Colors[ImGuiCol_ChildBg]);
        ImGui::Button(name);
        ImGui::PopStyleColor(3);
      }
      else
      {
        if (ImGui::Button(name))
          returnedIndex = index;
      }

      index++;

      if (index != names.size())
        ImGui::SameLine();
    }

    return returnedIndex;
  }
}
