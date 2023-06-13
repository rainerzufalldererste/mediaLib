#ifndef mSoftwareWindow_h__
#define mSoftwareWindow_h__

#include "mRenderParams.h"
#include "mImageBuffer.h"

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
