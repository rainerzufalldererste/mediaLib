// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef mSoftwareWindow_h__
#define mSoftwareWindow_h__

#include "default.h"
#include "mImageBuffer.h"
#include "mRenderParams.h"

enum mSoftwareWindow_DisplayMode
{
  mSW_DM_Windowed = 0,
  mSW_DM_Fullscreen = SDL_WINDOW_FULLSCREEN,
  mSW_DM_FullscreenDesktop = SDL_WINDOW_FULLSCREEN_DESKTOP,
};

struct mSoftwareWindow;

mFUNCTION(mSoftwareWindow_Create, OUT mPtr<mSoftwareWindow> *pWindow, IN mAllocator *pAllocator, const std::string &title, const mVec2s &size, const mSoftwareWindow_DisplayMode displaymode = mSW_DM_Windowed);
mFUNCTION(mSoftwareWindow_Destroy, IN_OUT mPtr<mSoftwareWindow> *pWindow);
mFUNCTION(mSoftwareWindow_GetSize, mPtr<mSoftwareWindow> &window, OUT mVec2s *pSize);
mFUNCTION(mSoftwareWindow_GetBuffer, mPtr<mSoftwareWindow> &window, OUT mPtr<mImageBuffer> *pBuffer);
mFUNCTION(mSoftwareWindow_Swap, mPtr<mSoftwareWindow> &window);
mFUNCTION(mSoftwareWindow_SetSize, mPtr<mSoftwareWindow> &window, const mVec2s &size);
mFUNCTION(mSoftwareWindow_GetSdlWindowPtr, mPtr<mSoftwareWindow> &window, OUT SDL_Window **ppSdlWindow);

#endif // mSoftwareWindow_h__
