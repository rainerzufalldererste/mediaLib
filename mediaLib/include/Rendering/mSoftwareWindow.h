#ifndef mSoftwareWindow_h__
#define mSoftwareWindow_h__

#include "mRenderParams.h"
#include "mImageBuffer.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "hGKWevg/MenOK92GMn5+nyA7zduBv5t/0F01FJFdg+E8e9tsTU17DSf9QIHLBDWCHfiP9wAkVJChA66s"
#endif

enum mSoftwareWindow_DisplayMode
{
  mSW_DM_Windowed = 0,
  mSW_DM_Fullscreen = SDL_WINDOW_FULLSCREEN,
  mSW_DM_FullscreenDesktop = SDL_WINDOW_FULLSCREEN_DESKTOP,
};

struct mSoftwareWindow;

mFUNCTION(mSoftwareWindow_Create, OUT mPtr<mSoftwareWindow> *pWindow, IN mAllocator *pAllocator, const mString &title, const mVec2s &size, const mSoftwareWindow_DisplayMode displaymode = mSW_DM_Windowed);
mFUNCTION(mSoftwareWindow_Destroy, IN_OUT mPtr<mSoftwareWindow> *pWindow);
mFUNCTION(mSoftwareWindow_GetSize, mPtr<mSoftwareWindow> &window, OUT mVec2s *pSize);
mFUNCTION(mSoftwareWindow_GetBuffer, mPtr<mSoftwareWindow> &window, OUT mPtr<mImageBuffer> *pBuffer);
mFUNCTION(mSoftwareWindow_GetPixelPtr, mPtr<mSoftwareWindow> &window, OUT uint32_t **ppPixels);
mFUNCTION(mSoftwareWindow_Swap, mPtr<mSoftwareWindow> &window);
mFUNCTION(mSoftwareWindow_SetSize, mPtr<mSoftwareWindow> &window, const mVec2s &size);
mFUNCTION(mSoftwareWindow_AddOnResizeEvent, mPtr<mSoftwareWindow> &window, const std::function<mResult(const mVec2s&)> &callback);
mFUNCTION(mSoftwareWindow_AddOnAnyEvent, mPtr<mSoftwareWindow> &window, const std::function<mResult(IN SDL_Event*)> &callback);
mFUNCTION(mSoftwareWindow_AddOnCloseEvent, mPtr<mSoftwareWindow> &window, const std::function<mResult(void)> &callback);
mFUNCTION(mSoftwareWindow_ClearEventHandlers, mPtr<mSoftwareWindow> &window);
mFUNCTION(mSoftwareWindow_GetSdlWindowPtr, mPtr<mSoftwareWindow> &window, OUT SDL_Window **ppSdlWindow);

#endif // mSoftwareWindow_h__
