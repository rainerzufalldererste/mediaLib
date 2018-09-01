// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "mHardwareWindow.h"
#include "mFramebuffer.h"

struct mHardwareWindow
{
  SDL_Window *pWindow;
  mRenderContextId renderContextID;
  bool no3dAvailableFlag;
};

mFUNCTION(mHardwareWindow_Create_Internal, IN mHardwareWindow *pWindow, const std::string & title, const mVec2s & size, const mHardwareWindow_DisplayMode displaymode);
mFUNCTION(mHardwareWindow_Destroy_Internal, IN mHardwareWindow *pWindow);

mFUNCTION(mHardwareWindow_Create, OUT mPtr<mHardwareWindow>* pWindow, IN mAllocator * pAllocator, const std::string & title, const mVec2s & size, const mHardwareWindow_DisplayMode displaymode /* = mHW_DM_Windowed */)
{
  mFUNCTION_SETUP();

  mERROR_IF(SDL_Init(SDL_INIT_EVERYTHING), mR_InternalError);

  mERROR_CHECK(mSharedPointer_Allocate(pWindow, pAllocator, (std::function<void(mHardwareWindow *)>)[](mHardwareWindow *pData) { mHardwareWindow_Destroy_Internal(pData); }, 1));

  mERROR_CHECK(mHardwareWindow_Create_Internal(pWindow->GetPointer(), title, size, displaymode));

  mERROR_CHECK(mRenderParams_CreateRenderContext(&(*pWindow)->renderContextID, *pWindow));

  mGL_ERROR_CHECK();

  mERROR_CHECK(mHardwareWindow_SetAsActiveRenderTarget(*pWindow));

  mPRINT("GPU: %s %s\nDriver Version: %s\nGLSL Version: %s\n", glGetString(GL_VENDOR), glGetString(GL_RENDERER), glGetString(GL_VERSION), glGetString(GL_SHADING_LANGUAGE_VERSION));

  mGL_ERROR_CHECK();

  mRETURN_SUCCESS();
}

mFUNCTION(mHardwareWindow_Destroy, IN_OUT mPtr<mHardwareWindow> *pWindow)
{
  mFUNCTION_SETUP();

  mERROR_IF(pWindow == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Destroy(pWindow));

  mRETURN_SUCCESS();
}

mFUNCTION(mHardwareWindow_GetSize, mPtr<mHardwareWindow> &window, OUT mVec2s *pSize)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr || pSize == nullptr, mR_ArgumentNull);

  int w, h;
  SDL_GetWindowSize(window->pWindow, &w, &h);

  *pSize = mVec2s(w, h);

  mRETURN_SUCCESS();
}

mFUNCTION(mHardwareWindow_Swap, mPtr<mHardwareWindow>& window)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr, mR_ArgumentNull);

#if defined(mRENDERER_OPENGL)
  SDL_GL_SwapWindow(window->pWindow);
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mHardwareWindow_SetSize, mPtr<mHardwareWindow>& window, const mVec2s & size)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr, mR_ArgumentNull);
  mERROR_IF(size.x > INT_MAX || size.y > INT_MAX, mR_ArgumentOutOfBounds);

  SDL_SetWindowSize(window->pWindow, (int)size.x, (int)size.y);

  mRETURN_SUCCESS();
}

mFUNCTION(mHardwareWindow_SetAsActiveRenderTarget, mPtr<mHardwareWindow>& window)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mFramebuffer_Unbind());
  mERROR_CHECK(mRenderParams_ActivateRenderContext(window, window->renderContextID));

  mVec2s resolution;
  mERROR_CHECK(mHardwareWindow_GetSize(window, &resolution));
  mERROR_CHECK(mRenderParams_SetCurrentRenderResolution(resolution));

  mRETURN_SUCCESS();
}

mFUNCTION(mHardwareWindow_GetSdlWindowPtr, mPtr<mHardwareWindow>& window, OUT SDL_Window ** ppSdlWindow)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr || ppSdlWindow == nullptr, mR_ArgumentNull);

  *ppSdlWindow = window->pWindow;

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mHardwareWindow_Create_Internal, IN mHardwareWindow *pWindow, const std::string &title, const mVec2s &size, const mHardwareWindow_DisplayMode displaymode)
{
  mFUNCTION_SETUP();
  mUnused(displaymode);

  mERROR_IF(size.x > INT_MAX || size.y > INT_MAX, mR_ArgumentOutOfBounds);

  bool try3d = true;
retry_without_3d:
  SDL_GL_SetAttribute(SDL_GL_STEREO, try3d ? 1 : 0);

  pWindow->pWindow = SDL_CreateWindow(title.c_str(), SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, (int)size.x, (int)size.y
    , 
#if defined(mRENDERER_OPENGL)
    SDL_WINDOW_OPENGL |
#endif
    displaymode);

  if (pWindow->pWindow == nullptr && try3d)
  {
    try3d = false;
    goto retry_without_3d;
  }

  mERROR_IF(pWindow->pWindow == nullptr, mR_InternalError);

  mRETURN_SUCCESS();
}

mFUNCTION(mHardwareWindow_Destroy_Internal, IN mHardwareWindow * pWindow)
{
  mFUNCTION_SETUP();

  mERROR_IF(pWindow == nullptr, mR_ArgumentNull);
  
  mERROR_CHECK(mRenderParams_DestroyRenderContext(&pWindow->renderContextID));

  SDL_DestroyWindow(pWindow->pWindow);
  pWindow->pWindow = nullptr;

  mRETURN_SUCCESS();
}
