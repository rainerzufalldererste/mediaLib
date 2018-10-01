#ifndef mHardwareWindow_h__
#define mHardwareWindow_h__

#include "mRenderParams.h"

enum mHardwareWindow_DisplayMode
{
  mHW_DM_Windowed = 0,
  mHW_DM_Fullscreen = SDL_WINDOW_FULLSCREEN,
  mHW_DM_FullscreenDesktop = SDL_WINDOW_FULLSCREEN_DESKTOP,
  mHW_DM_Resizeable = SDL_WINDOW_RESIZABLE,
  mHW_DM_Maximized = SDL_WINDOW_MAXIMIZED,
  mHW_DM_Minimized = SDL_WINDOW_MINIMIZED,
  mHW_DM_Hidden = SDL_WINDOW_HIDDEN,
  mHW_DM_NotInTaskbar = SDL_WINDOW_SKIP_TASKBAR,
};

struct mHardwareWindow;

mFUNCTION(mHardwareWindow_Create, OUT mPtr<mHardwareWindow> *pWindow, IN mAllocator *pAllocator, const mString &title, const mVec2s &size, const mHardwareWindow_DisplayMode displaymode = mHW_DM_Windowed, const bool stereo3dIfAvailable = false);
mFUNCTION(mHardwareWindow_Destroy, IN_OUT mPtr<mHardwareWindow> *pWindow);
mFUNCTION(mHardwareWindow_GetSize, mPtr<mHardwareWindow> &window, OUT mVec2s *pSize);
mFUNCTION(mHardwareWindow_Swap, mPtr<mHardwareWindow> &window);
mFUNCTION(mHardwareWindow_SetSize, mPtr<mHardwareWindow> &window, const mVec2s &size);
mFUNCTION(mHardwareWindow_SetAsActiveRenderTarget, mPtr<mHardwareWindow> &window);
mFUNCTION(mHardwareWindow_GetSdlWindowPtr, mPtr<mHardwareWindow> &window, OUT SDL_Window **ppSdlWindow);
mFUNCTION(mHardwareWindow_GetRenderContextId, mPtr<mHardwareWindow> &window, OUT mRenderContextId *pRenderContextId);

mFUNCTION(mHardwareWindow_AddOnResizeEvent, mPtr<mHardwareWindow> &window, const std::function<mResult(const mVec2s &)> &callback);
mFUNCTION(mHardwareWindow_AddOnCloseEvent, mPtr<mHardwareWindow> &window, const std::function<mResult(void)> &callback);
mFUNCTION(mHardwareWindow_AddOnAnyEvent, mPtr<mHardwareWindow> &window, const std::function<mResult(IN SDL_Event *)> &callback);

#endif // mHardwareWindow_h__
