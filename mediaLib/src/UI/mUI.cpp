#include "mUI.h"

#if defined(mRENDERER_OPENGL)
#define IMGUI_IMPL_OPENGL_LOADER_GLEW
#define DECLSPEC
#include "imgui/include/examples/imgui_impl_sdl.h"
//#include "imgui/include/examples/imgui_impl_opengl3.h"

#include "imgui/include/examples/imgui_impl_sdl.cpp"
#include "imgui/include/examples/imgui_impl_opengl3.cpp"
#include "imgui/include/imgui.cpp"
#include "imgui/include/imgui_demo.cpp"
#include "imgui/include/imgui_draw.cpp"
#include "imgui/include/imgui_widgets.cpp"
#include "imgui/include/imgui_internal.h"
#undef DECLSPEC
#endif

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "FwItiCW9M4ymsopGJPX/Jho263tGovZ5P67oWtztNtdM6FLCPBYXOciUDwgZ6OKErj+1WGdP5jWXBdur"
#endif

ImGuiIO *mUI_pImguiIO;
ImFont *pFont = nullptr;
ImFont *pHeadline = nullptr;
ImFont *pSlimHeadline = nullptr;
ImFont *pSubHeadline = nullptr;
ImFont *pMonospacedFont = nullptr;
ImFont *pBold = nullptr;

bool mUI_AutoUpdateMousePosition = true;

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mUI_Initilialize, mPtr<mHardwareWindow> &hardwareWindow, const bool addUpdateCallback /* = true */)
{
  mFUNCTION_SETUP();

  mERROR_IF(hardwareWindow == nullptr, mR_ArgumentNull);

  const char glsl_version[] = "#version 150"; IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  mUI_pImguiIO = &ImGui::GetIO();

  SDL_Window *pWindow = nullptr;
  mERROR_CHECK(mHardwareWindow_GetSdlWindowPtr(hardwareWindow, &pWindow));

  SDL_GLContext glContext;
  mERROR_CHECK(mRenderParams_GetRenderContext(hardwareWindow, &glContext));

  ImGui_ImplSDL2_InitForOpenGL(pWindow, glContext);
  ImGui_ImplOpenGL3_Init(glsl_version);

  ImGui::StyleColorsLight();

  pFont = mUI_pImguiIO->Fonts->AddFontFromFileTTF("C:/Windows/Fonts/segoeui.ttf", 17.0f, nullptr, mUI_pImguiIO->Fonts->GetGlyphRangesDefault());
  mERROR_IF(pFont == nullptr, mR_InternalError);

  pMonospacedFont = mUI_pImguiIO->Fonts->AddFontFromFileTTF("C:/Windows/Fonts/consola.ttf", 13.0f, nullptr, mUI_pImguiIO->Fonts->GetGlyphRangesDefault());

  pHeadline = mUI_pImguiIO->Fonts->AddFontFromFileTTF("C:/Windows/Fonts/seguibl.ttf", 52.0f, nullptr, mUI_pImguiIO->Fonts->GetGlyphRangesDefault());
  pSlimHeadline = mUI_pImguiIO->Fonts->AddFontFromFileTTF("C:/Windows/Fonts/segoeuisl.ttf", 42.0f, nullptr, mUI_pImguiIO->Fonts->GetGlyphRangesDefault());
  pSubHeadline = mUI_pImguiIO->Fonts->AddFontFromFileTTF("C:/Windows/Fonts/seguisb.ttf", 28.0f, nullptr, mUI_pImguiIO->Fonts->GetGlyphRangesDefault());
  pBold = mUI_pImguiIO->Fonts->AddFontFromFileTTF("C:/Windows/Fonts/seguisb.ttf", 18.0f, nullptr, mUI_pImguiIO->Fonts->GetGlyphRangesDefault());

  if (addUpdateCallback)
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
  pColors[ImGuiCol_SliderGrab] = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
  pColors[ImGuiCol_SliderGrabActive] = ImVec4(0.47f, 0.47f, 0.47f, 1.00f);
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

  *ppIO = &ImGui::GetIO();

  mRETURN_SUCCESS();
}

mFUNCTION(mUI_SetCustomMousePosition, const mVec2f position)
{
  mFUNCTION_SETUP();

  ImGuiIO *pIO = nullptr;
  mERROR_CHECK(mUI_GetIO(&pIO));

  pIO->MousePos = ToImVec(position);

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

mFUNCTION(mUI_PushSlimHeadlineFont)
{
  mFUNCTION_SETUP();

  if (pSlimHeadline != nullptr)
    ImGui::PushFont(pSlimHeadline);

  mRETURN_SUCCESS();
}

mFUNCTION(mUI_PopSlimHeadlineFont)
{
  mFUNCTION_SETUP();

  if (pSlimHeadline != nullptr)
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

mFUNCTION(mUI_PushBoldFont)
{
  mFUNCTION_SETUP();

  if (pBold != nullptr)
    ImGui::PushFont(pBold);

  mRETURN_SUCCESS();
}

mFUNCTION(mUI_PopBoldFont)
{
  mFUNCTION_SETUP();

  if (pBold != nullptr)
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

    for (int32_t i = 0; i < num_segments; i++)
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

  bool ImageButtonWithText(ImTextureID texId, const char *label, const ImVec2 &imageSize, const ImVec2 &buttonSize, const ImVec2 &uv0, const ImVec2 &uv1, const float_t frame_padding, const ImVec4 &bg_col, const ImVec4 &tint_col)
  {
    ImGuiWindow *pWindow = GetCurrentWindow();

    if (pWindow->SkipItems)
      return false;

    ImVec2 size = imageSize;

    if (size.x <= 0 && size.y <= 0)
    {
      size.x = size.y = ImGui::GetTextLineHeightWithSpacing();
    }
    else
    {
      if (size.x <= 0)
        size.x = size.y;
      else if (size.y <= 0)
        size.y = size.x;

      size *= pWindow->FontWindowScale * ImGui::GetIO().FontGlobalScale;
    }

    ImGuiContext &g = *GImGui;
    const ImGuiStyle &style = g.Style;

    const ImGuiID id = pWindow->GetID(label);
    const ImVec2 textSize = ImGui::CalcTextSize(label, NULL, true);
    const bool hasText = textSize.x > 0;

    const float_t innerSpacing = hasText ? ((frame_padding >= 0) ? (float_t)frame_padding : (style.ItemInnerSpacing.x)) : 0.f;
    const ImVec2 padding = (frame_padding >= 0) ? ImVec2((float_t)frame_padding, (float_t)frame_padding) : style.FramePadding;
    const ImVec2 totalSizeWithoutPadding(size.x + innerSpacing + textSize.x, size.y > textSize.y ? size.y : textSize.y);
    ImRect bb(pWindow->DC.CursorPos, pWindow->DC.CursorPos + totalSizeWithoutPadding + padding * 2);

    if (buttonSize.x != 0)
    {
      if (buttonSize.x < 0)
        bb.Max.x = bb.Min.x + GetContentRegionAvail().x + buttonSize.x;
      else if (buttonSize.x < 1)
        bb.Max.x = bb.Min.x + buttonSize.x * GetContentRegionAvail().x;
      else
        bb.Max.x = bb.Min.x + buttonSize.x;
    }

    if (buttonSize.y > 0)
    {
      if (buttonSize.y < 0)
        bb.Max.y = bb.Min.y + GetContentRegionAvail().y + buttonSize.y;
      else if (buttonSize.y < 1)
        bb.Max.y = bb.Min.y + buttonSize.y * GetContentRegionAvail().y;
      else
        bb.Max.y = bb.Min.y + buttonSize.y;
    }

    ImVec2 start(0, 0);
    start = pWindow->DC.CursorPos + padding;
    
    if (size.y < textSize.y)
      start.y += (textSize.y - size.y) * .5f;

    const ImRect image_bb(start, start + size);
    start = pWindow->DC.CursorPos + padding;start.x += size.x + innerSpacing;if (size.y > textSize.y) start.y += (size.y - textSize.y)*.5f;
    ItemSize(bb);

    if (!ItemAdd(bb, id))
      return false;

    bool hovered = false, held = false;
    bool pressed = ButtonBehavior(bb, id, &hovered, &held);

    // Render
    const ImU32 col = GetColorU32((hovered && held) ? ImGuiCol_ButtonActive : hovered ? ImGuiCol_ButtonHovered : ImGuiCol_Button);
    RenderFrame(bb.Min, bb.Max, col, true, ImClamp((float_t)ImMin(padding.x, padding.y), 0.0f, style.FrameRounding));
    
    if (bg_col.w > 0.0f)
      pWindow->DrawList->AddRectFilled(image_bb.Min, image_bb.Max, GetColorU32(bg_col));

    pWindow->DrawList->AddImage(texId, image_bb.Min, image_bb.Max, uv0, uv1, GetColorU32(tint_col));

    if (textSize.x > 0)
      ImGui::RenderText(start, label);
    
    return pressed;
  }
}
