#include "mediaLib.h"

#pragma warning (push, 0)
#define IMGUI_IMPL_OPENGL_LOADER_GLEW
#define DECLSPEC
#include "imgui/include/backends/imgui_impl_sdl.h"
//#include "imgui/include/backends/imgui_impl_opengl3.h"

#include "imgui/include/backends/imgui_impl_sdl.cpp"
#include "imgui/include/backends/imgui_impl_opengl3.cpp"
#include "imgui/include/imgui.cpp"
#include "imgui/include/imgui_demo.cpp"
#include "imgui/include/imgui_draw.cpp"
#include "imgui/include/imgui_widgets.cpp"
#include "imgui/include/imgui_internal.h"
#include "imgui/include/imgui_tables.cpp"
#undef DECLSPEC
#pragma warning(pop)

namespace ImGui
{
  // See: https://github.com/ocornut/imgui/issues/1901
  bool BufferingBar(const char *label, float_t value, const ImVec2 &size_arg, const ImU32 &bg_col, const ImU32 &fg_col)
  {
    ImGuiWindow *pWindow = GetCurrentWindow();

    if (pWindow == nullptr || pWindow->SkipItems)
      return false;

    ImGuiContext &g = *GImGui;
    const ImGuiStyle &style = g.Style;
    const ImGuiID id = pWindow->GetID(label);

    ImVec2 pos = pWindow->DC.CursorPos;
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

    pWindow->DrawList->AddRectFilled(bb.Min, ImVec2(pos.x + (float_t)circleStart, bb.Max.y), bg_col);
    pWindow->DrawList->AddRectFilled(bb.Min, ImVec2(pos.x + (float_t)circleStart * value, bb.Max.y), fg_col);

    const double_t t = g.Time;
    const double_t r = size.y / 2;
    const double_t speed = 1.5f;

    const double_t a = speed * 0;
    const double_t b = speed * 0.333f;
    const double_t c = speed * 0.666f;

    const double_t o1 = (circleWidth + r) * (t + a - speed * (int32_t)((t + a) / speed)) / speed;
    const double_t o2 = (circleWidth + r) * (t + b - speed * (int32_t)((t + b) / speed)) / speed;
    const double_t o3 = (circleWidth + r) * (t + c - speed * (int32_t)((t + c) / speed)) / speed;

    pWindow->DrawList->AddCircleFilled(ImVec2(pos.x + (float_t)circleEnd - (float_t)o1, bb.Min.y + (float_t)r), (float_t)r, bg_col);
    pWindow->DrawList->AddCircleFilled(ImVec2(pos.x + (float_t)circleEnd - (float_t)o2, bb.Min.y + (float_t)r), (float_t)r, bg_col);
    pWindow->DrawList->AddCircleFilled(ImVec2(pos.x + (float_t)circleEnd - (float_t)o3, bb.Min.y + (float_t)r), (float_t)r, bg_col);

    return true;
  }

  // See: https://github.com/ocornut/imgui/issues/1901
  bool Spinner(const char *label, float_t radius, float_t thickness, const ImU32 &color)
  {
    ImGuiWindow *pWindow = GetCurrentWindow();

    if (pWindow == nullptr || pWindow->SkipItems)
      return false;

    ImGuiContext &g = *GImGui;
    const ImGuiStyle &style = g.Style;
    const ImGuiID id = pWindow->GetID(label);

    ImVec2 pos = pWindow->DC.CursorPos;
    ImVec2 size((radius) * 2, (radius + style.FramePadding.y) * 2);

    const ImRect bb(pos, ImVec2(pos.x + size.x, pos.y + size.y));
    ItemSize(bb, style.FramePadding.y);

    if (!ItemAdd(bb, id))
      return false;

    // Render
    pWindow->DrawList->PathClear();

    int32_t num_segments = 30;
    int32_t start = (int32_t)abs(ImSin((float_t)g.Time * 1.8f) * (num_segments - 5));

    const float_t a_min = mTWOPIf * ((float_t)start) / (float_t)num_segments;
    const float_t a_max = mTWOPIf * ((float_t)num_segments - 3) / (float_t)num_segments;

    const ImVec2 centre = ImVec2(pos.x + radius, pos.y + radius + style.FramePadding.y);

    for (int32_t i = 0; i < num_segments; i++)
    {
      const float_t a = a_min + ((float_t)i / (float_t)num_segments) * (a_max - a_min);

      pWindow->DrawList->PathLineTo(ImVec2(centre.x + ImCos(a + (float_t)g.Time * 8) * radius, centre.y + ImSin(a + (float_t)g.Time * 8) * radius));
    }

    pWindow->DrawList->PathStroke(color, false, thickness);

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

    if (pWindow == nullptr || pWindow->SkipItems)
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
    start = pWindow->DC.CursorPos + padding; start.x += size.x + innerSpacing; if (size.y > textSize.y) start.y += (size.y - textSize.y) * .5f;
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

  void DrawRectangle(const ImVec2 &position, const ImVec2 &size, const ImVec4 &color)
  {
    ImGuiWindow *pWindow = GetCurrentWindow();

    if (pWindow == nullptr || pWindow->SkipItems)
      return;

    if (color.w > 0.0f)
      pWindow->DrawList->AddRectFilled(position, position + size, GetColorU32(color));
  }

  void ProgressBarRounded(float_t fraction, const ImVec2 &size_arg /* = ImVec2(-FLT_MIN, 0) */, const char *overlay /* = nullptr */)
  {
    ImGuiWindow *window = GetCurrentWindow();
    if (window->SkipItems)
      return;

    ImGuiContext &g = *GImGui;
    const ImGuiStyle &style = g.Style;

    ImVec2 pos = window->DC.CursorPos;
    ImVec2 size;

    if (overlay != nullptr)
      size = CalcItemSize(size_arg, CalcItemWidth(), g.FontSize + style.FramePadding.y * 2.0f);
    else
      size = CalcItemSize(size_arg, CalcItemWidth(), mMax(g.FontSize * 0.25f, style.FramePadding.y * 2.0f));

    ImRect bb(pos, pos + size);
    ItemSize(size, style.FramePadding.y);
    if (!ItemAdd(bb, 0))
      return;

    const float_t rounding = bb.GetHeight() * 0.5f;

    // Render
    fraction = ImSaturate(fraction);
    RenderFrame(bb.Min, bb.Max, GetColorU32(ImGuiCol_FrameBg), true, rounding);
    bb.Expand(ImVec2(-style.FrameBorderSize, -style.FrameBorderSize));
    const ImVec2 fill_br = ImVec2(ImLerp(bb.Min.x, bb.Max.x, fraction), bb.Max.y);
    const float_t minX = mClamp(bb.Min.x + rounding * 2.f, bb.Min.x, bb.Max.x);
    RenderFrame(ImVec2(bb.Min.x, bb.Min.y), ImVec2(mLerp(minX, bb.Max.x, fraction), bb.Max.y), GetColorU32(ImGuiCol_PlotHistogram), false, rounding);

    if (overlay != nullptr)
    {
      ImVec2 overlay_size = CalcTextSize(overlay, nullptr);

      if (overlay_size.x > 0.0f)
        RenderTextClipped(bb.Min, bb.Max, overlay, nullptr, &overlay_size, ImVec2(0.5f, 0.5f), &bb);
    }
  }

  void TextUnformattedWrapped(const char *text)
  {
    ImGuiContext &g = *GImGui;
    const bool need_backup = (g.CurrentWindow->DC.TextWrapPos < 0.0f);  // Keep existing wrap position if one is already set

    if (need_backup)
      PushTextWrapPos(0.0f);

    TextEx(text, NULL, ImGuiTextFlags_NoWidthForLargeClippedText); // Skip formatting

    if (need_backup)
      PopTextWrapPos();
  }

  void SetTooltipUnformatted(const char *text)
  {
    BeginTooltipEx(0, ImGuiTooltipFlags_OverridePreviousTooltip);
    TextUnformatted(text);
    EndTooltip();
  }
}
