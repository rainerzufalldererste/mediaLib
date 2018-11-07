#include "mHardwareWindow.h"
#include "mFramebuffer.h"
#include "mChunkedArray.h"

struct mHardwareWindow
{
  SDL_Window *pWindow;
  mRenderContextId renderContextID;
  mPtr<mQueue<std::function<mResult (IN SDL_Event *)>>> onEventCallbacks;
  mPtr<mQueue<std::function<mResult (void)>>> onExitCallbacks;
  mPtr<mQueue<std::function<mResult(const mVec2s &)>>> onResizeCallbacks;
};

mFUNCTION(mHardwareWindow_Create_Internal, IN_OUT mHardwareWindow *pWindow, IN mAllocator *pAllocator, const mString & title, const mVec2s & size, const mHardwareWindow_DisplayMode displaymode, const bool stereo3dIfAvailable);
mFUNCTION(mHardwareWindow_Destroy_Internal, IN mHardwareWindow *pWindow);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mHardwareWindow_Create, OUT mPtr<mHardwareWindow> *pWindow, IN mAllocator *pAllocator, const mString &title, const mVec2s &size, const mHardwareWindow_DisplayMode displaymode /* = mHW_DM_Windowed */, const bool stereo3dIfAvailable /* = false */)
{
  mFUNCTION_SETUP();

  mERROR_IF(SDL_Init(SDL_INIT_EVERYTHING), mR_InternalError);

  mERROR_CHECK(mSharedPointer_Allocate(pWindow, pAllocator, (std::function<void(mHardwareWindow *)>)[](mHardwareWindow *pData) { mHardwareWindow_Destroy_Internal(pData); }, 1));

  mERROR_CHECK(mHardwareWindow_Create_Internal(pWindow->GetPointer(), pAllocator, title, size, displaymode, stereo3dIfAvailable));

  mERROR_CHECK(mRenderParams_CreateRenderContext(&(*pWindow)->renderContextID, *pWindow));

  mGL_ERROR_CHECK();

  mERROR_CHECK(mHardwareWindow_SetAsActiveRenderTarget(*pWindow));

  mDebugOut("GPU: %s %s\nDriver Version: %s\nGLSL Version: %s\n", glGetString(GL_VENDOR), glGetString(GL_RENDERER), glGetString(GL_VERSION), glGetString(GL_SHADING_LANGUAGE_VERSION));

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

  SDL_Event _event;
  while (SDL_PollEvent(&_event))
  {
    size_t onEventCount = 0;
    mERROR_CHECK(mQueue_GetCount(window->onEventCallbacks, &onEventCount));

    for (size_t i = 0; i < onEventCount; ++i)
    {
      std::function<mResult(IN SDL_Event *)> *pFunction = nullptr;
      mERROR_CHECK(mQueue_PointerAt(window->onEventCallbacks, i, &pFunction));

      mResult result = ((*pFunction)(&_event));

      if (mFAILED(result))
      {
        if (result == mR_Break)
          break;
        else
          mRETURN_RESULT(result);
      }
    }

    if (_event.type == SDL_QUIT)
    {
      size_t onExitCallbackCount = 0;
      mERROR_CHECK(mQueue_GetCount(window->onExitCallbacks, &onExitCallbackCount));

      for (size_t i = 0; i < onExitCallbackCount; ++i)
      {
        std::function<mResult(void)> *pFunction = nullptr;
        mERROR_CHECK(mQueue_PointerAt(window->onExitCallbacks, i, &pFunction));

        mResult result = ((*pFunction)());

        if (mFAILED(result))
        {
          if (result == mR_Break)
            break;
          else
            mRETURN_RESULT(result);
        }
      }
    }
    else if (_event.type == SDL_WINDOWEVENT && (_event.window.event == SDL_WINDOWEVENT_MAXIMIZED || _event.window.event == SDL_WINDOWEVENT_RESIZED || _event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED || _event.window.event == SDL_WINDOWEVENT_RESTORED))
    {
      mVec2s size;
      mERROR_CHECK(mHardwareWindow_GetSize(window, &size));

      mERROR_CHECK(mHardwareWindow_SetAsActiveRenderTarget(window));

      size_t onResizeCallbackCount = 0;
      mERROR_CHECK(mQueue_GetCount(window->onResizeCallbacks, &onResizeCallbackCount));

      for (size_t i = 0; i < onResizeCallbackCount; ++i)
      {
        std::function<mResult(const mVec2s &)> *pFunction = nullptr;
        mERROR_CHECK(mQueue_PointerAt(window->onResizeCallbacks, i, &pFunction));

        mResult result = ((*pFunction)(size));

        if (mFAILED(result))
        {
          if (result == mR_Break)
            break;
          else
            mRETURN_RESULT(result);
        }
      }
    }
  }

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

mFUNCTION(mHardwareWindow_SetAsActiveRenderTarget, mPtr<mHardwareWindow> &window)
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

mFUNCTION(mHardwareWindow_GetSdlWindowPtr, mPtr<mHardwareWindow> &window, OUT SDL_Window **ppSdlWindow)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr || ppSdlWindow == nullptr, mR_ArgumentNull);

  *ppSdlWindow = window->pWindow;

  mRETURN_SUCCESS();
}

mFUNCTION(mHardwareWindow_GetRenderContextId, mPtr<mHardwareWindow> &window, OUT mRenderContextId *pRenderContextId)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr || pRenderContextId == nullptr, mR_ArgumentNull);

  *pRenderContextId = window->renderContextID;

  mRETURN_SUCCESS();
}

mFUNCTION(mHardwareWindow_AddOnResizeEvent, mPtr<mHardwareWindow>& window, const std::function<mResult(const mVec2s&)> &callback)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr || callback == nullptr, mR_ArgumentNull);

  std::function<mResult(const mVec2s&)> function = callback;

  mERROR_CHECK(mQueue_PushBack(window->onResizeCallbacks, &function));

  mRETURN_SUCCESS();
}

mFUNCTION(mHardwareWindow_AddOnCloseEvent, mPtr<mHardwareWindow>& window, const std::function<mResult(void)> &callback)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr || callback == nullptr, mR_ArgumentNull);

  std::function<mResult(void)> function = callback;

  mERROR_CHECK(mQueue_PushBack(window->onExitCallbacks, &function));

  mRETURN_SUCCESS();
}

mFUNCTION(mHardwareWindow_AddOnAnyEvent, mPtr<mHardwareWindow>& window, const std::function<mResult(IN SDL_Event*)> &callback)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr || callback == nullptr, mR_ArgumentNull);

  std::function<mResult(IN SDL_Event*)> function = callback;

  mERROR_CHECK(mQueue_PushBack(window->onEventCallbacks, &function));

  mRETURN_SUCCESS();
}

mFUNCTION(mHardwareWindow_ClearEventHandlers, mPtr<mHardwareWindow> &window)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mQueue_Clear(window->onEventCallbacks));
  mERROR_CHECK(mQueue_Clear(window->onExitCallbacks));
  mERROR_CHECK(mQueue_Clear(window->onResizeCallbacks));

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mHardwareWindow_Create_Internal, IN_OUT mHardwareWindow *pWindow, IN mAllocator *pAllocator, const mString &title, const mVec2s &size, const mHardwareWindow_DisplayMode displaymode, const bool stereo3dIfAvailable)
{
  mFUNCTION_SETUP();
  mUnused(displaymode);

  mERROR_IF(size.x > INT_MAX || size.y > INT_MAX, mR_ArgumentOutOfBounds);

  bool try3d = stereo3dIfAvailable;
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

  mERROR_CHECK(mQueue_Create(&pWindow->onEventCallbacks, pAllocator));
  mERROR_CHECK(mQueue_Create(&pWindow->onResizeCallbacks, pAllocator));
  mERROR_CHECK(mQueue_Create(&pWindow->onExitCallbacks, pAllocator));

  mRETURN_SUCCESS();
}

mFUNCTION(mHardwareWindow_Destroy_Internal, IN mHardwareWindow * pWindow)
{
  mFUNCTION_SETUP();

  mERROR_IF(pWindow == nullptr, mR_ArgumentNull);
  
  mERROR_CHECK(mRenderParams_DestroyRenderContext(&pWindow->renderContextID));

  SDL_DestroyWindow(pWindow->pWindow);
  pWindow->pWindow = nullptr;

  mERROR_CHECK(mQueue_Destroy(&pWindow->onEventCallbacks));
  mERROR_CHECK(mQueue_Destroy(&pWindow->onResizeCallbacks));
  mERROR_CHECK(mQueue_Destroy(&pWindow->onExitCallbacks));

  mRETURN_SUCCESS();
}
