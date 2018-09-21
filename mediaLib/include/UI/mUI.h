// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef mUI_h__
#define mUI_h__

#include "mRenderParams.h"
#include "mHardwareWindow.h"

#include "../3rdParty/imgui/imgui.h"

mFUNCTION(mUI_Initilialize, mPtr<mHardwareWindow> &hardwareWindow);
mFUNCTION(mUI_StartFrame, mPtr<mHardwareWindow> &hardwareWindow);
mFUNCTION(mUI_Shutdown);
mFUNCTION(mUI_Bake, mPtr<mHardwareWindow> &hardwareWindow);
mFUNCTION(mUI_Render);
mFUNCTION(mUI_ProcessEvent, IN SDL_Event *pEvent);

mFUNCTION(mUI_PushMonospacedFont);
mFUNCTION(mUI_PopMonospacedFont);

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

#endif // mUI_h__
