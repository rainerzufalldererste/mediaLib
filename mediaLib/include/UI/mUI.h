#ifndef mUI_h__
#define mUI_h__

#include "mRenderParams.h"
#include "mHardwareWindow.h"

#include "../3rdParty/imgui/include/imgui.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "ni+7LtuBuidVXi1zTeZoQl05ZnxvKxiU4irKh3Y7ujekV3wpc490mR7HJgQ2uJ8tOwSDWr3HEdTRlETV"
#endif

mFUNCTION(mUI_Initilialize, mPtr<mHardwareWindow> &hardwareWindow, const bool addUpdateCallback = true);
mFUNCTION(mUI_SetIniFilePath, const mString &path);
mFUNCTION(mUI_StartFrame, mPtr<mHardwareWindow> &hardwareWindow);
mFUNCTION(mUI_Shutdown);
mFUNCTION(mUI_Bake);
mFUNCTION(mUI_Render);
mFUNCTION(mUI_ProcessEvent, IN SDL_Event *pEvent);
mFUNCTION(mUI_GetIO, OUT ImGuiIO **ppIO);

// This should be called after `mUI_StartFrame`.
mFUNCTION(mUI_SetCustomMousePosition, const mVec2f position);
mFUNCTION(mUI_ManuallyUpdateMousePosition, const bool enable);

mFUNCTION(mUI_PushMonospacedFont);
mFUNCTION(mUI_PopMonospacedFont);

mFUNCTION(mUI_PushHeadlineFont);
mFUNCTION(mUI_PopHeadlineFont);

mFUNCTION(mUI_PushSlimHeadlineFont);
mFUNCTION(mUI_PopSlimHeadlineFont);

mFUNCTION(mUI_PushSubHeadlineFont);
mFUNCTION(mUI_PopSubHeadlineFont);

mFUNCTION(mUI_PushBoldFont);
mFUNCTION(mUI_PopBoldFont);

mFUNCTION(mUI_SetLightColourScheme);
mFUNCTION(mUI_SetDarkColourScheme);
mFUNCTION(mUI_SetDpiScalingFactor, const float_t scalingFactor);

// Imgui Extentions:
namespace ImGui
{
  bool BufferingBar(const char *label, float_t value, const ImVec2 &size_arg, const ImU32 &bg_col, const ImU32 &fg_col);
  bool Spinner(const char *label, float_t radius, float_t thickness, const ImU32 &color);
  size_t TabBar(const std::initializer_list<const char *> &names, const size_t activeBefore);
  bool ImageButtonWithText(ImTextureID texId, const char *label, const ImVec2 &imageSize = ImVec2(0, 0), const ImVec2 &buttonSize = ImVec2(0, 0), const ImVec2 &uv0 = ImVec2(0, 0), const ImVec2 &uv1 = ImVec2(1, 1), const float_t frame_padding = -1, const ImVec4 &bg_col = ImVec4(0, 0, 0, 0), const ImVec4 &tint_col = ImVec4(1, 1, 1, 1));
  void DrawRectangle(const ImVec2 &position, const ImVec2 &size, const ImVec4 &color);
  void ProgressBarRounded(float_t fraction, const ImVec2 &size_arg = ImVec2(-FLT_MIN, 0), const char *overlay = nullptr);
  void TextUnformattedWrapped(const char *text);
  void SetTooltipUnformatted(const char *text);
  bool IsItemHoveredManual(const mVec2f mousePosition);
}

#endif // mUI_h__
