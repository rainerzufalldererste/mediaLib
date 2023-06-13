#ifndef mUI_h__
#define mUI_h__

#include "mRenderParams.h"
#include "mHardwareWindow.h"

#include "../3rdParty/imgui/imgui.h"

mFUNCTION(mUI_Initilialize, mPtr<mHardwareWindow> &hardwareWindow);
mFUNCTION(mUI_StartFrame, mPtr<mHardwareWindow> &hardwareWindow);
mFUNCTION(mUI_Shutdown);
mFUNCTION(mUI_Bake);
mFUNCTION(mUI_Render);
mFUNCTION(mUI_ProcessEvent, IN SDL_Event *pEvent);

mFUNCTION(mUI_PushMonospacedFont);
mFUNCTION(mUI_PopMonospacedFont);

mFUNCTION(mUI_PushHeadlineFont);
mFUNCTION(mUI_PopHeadlineFont);

template <typename T>
inline ImVec2 cast(const mVec2t<T> &vec2t)
{
  ImVec2 v;

  v.x = (float_t)vec2t.x;
  v.y = (float_t)vec2t.y;

  return v;
}

template <typename T>
inline ImVec4 cast(const mVec4t<T> &vec4t)
{
  ImVec4 v;

  v.x = (float_t)vec4t.x;
  v.y = (float_t)vec4t.y;
  v.z = (float_t)vec4t.z;
  v.w = (float_t)vec4t.w;

  return v;
}

// Imgui Extentions:
namespace ImGui
{
  bool BufferingBar(const char *label, float_t value, const ImVec2 &size_arg, const ImU32 &bg_col, const ImU32 &fg_col);
  bool Spinner(const char *label, float_t radius, float_t thickness, const ImU32 &color);
}

#endif // mUI_h__
