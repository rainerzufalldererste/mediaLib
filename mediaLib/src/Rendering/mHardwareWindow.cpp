#include "mHardwareWindow.h"

#include "mFramebuffer.h"
#include "mChunkedArray.h"
#include "mSystemInfo.h"
#include "mProfiler.h"

#define DECLSPEC
#include "SDL_syswm.h"
#undef DECLSPEC

#include <dwmapi.h>
#pragma comment(lib, "dwmapi.lib")

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "uH9cesumuC4LRBXN9mdn+OhJb7RQbWTxufxBomi9I0ZjgSTunXjoEOf8H6Azc0p7O0xvz68h9B4SzPx6"
#endif

struct mHardwareWindow
{
  SDL_Window *pWindow;
  mRenderContextId renderContextID;
  mPtr<mQueue<std::function<mResult (IN SDL_Event *)>>> onEventCallbacks;
  mPtr<mQueue<std::function<mResult (void)>>> onExitCallbacks;
  mPtr<mQueue<std::function<mResult(const mVec2s &)>>> onResizeCallbacks;
  mPtr<mQueue<std::function<mResult(const bool)>>> onDarkModeChanged;
  mUniqueContainer<mFramebuffer> fakeFramebuffer;
  bool respectDarkMode, isDarkMode, processMessageQueueExplicit;
};

static mFUNCTION(mHardwareWindow_Create_Internal, IN_OUT mHardwareWindow *pWindow, IN mAllocator *pAllocator, const mString &title, const mVec2i &position, const mVec2s &size, const mHardwareWindow_DisplayMode displaymode, const bool stereo3dIfAvailable);
static mFUNCTION(mHardwareWindow_Destroy_Internal, IN mHardwareWindow *pWindow);
static mFUNCTION(mHardwareWindow_UpdateWindowTitleColour_Internal, mPtr<mHardwareWindow> &window);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mHardwareWindow_Create, OUT mPtr<mHardwareWindow> *pWindow, IN mAllocator *pAllocator, const mString &title, const mVec2s &size, const mHardwareWindow_DisplayMode displaymode /* = mHW_DM_Windowed */, const bool stereo3dIfAvailable /* = false */)
{
  return mHardwareWindow_Create(pWindow, pAllocator, title, mVec2i(SDL_WINDOWPOS_CENTERED), size, displaymode, stereo3dIfAvailable);
}

mFUNCTION(mHardwareWindow_Create, OUT mPtr<mHardwareWindow> *pWindow, IN mAllocator *pAllocator, const mString &title, const mVec2i &position, const mVec2s &size, const mHardwareWindow_DisplayMode displaymode /* = mHW_DM_Windowed */, const bool stereo3dIfAvailable /* = false */)
{
  mFUNCTION_SETUP();

  mERROR_IF(SDL_Init(SDL_INIT_EVERYTHING), mR_InternalError);

  mERROR_CHECK(mSharedPointer_Allocate(pWindow, pAllocator, (std::function<void(mHardwareWindow *)>)[](mHardwareWindow *pData) { mHardwareWindow_Destroy_Internal(pData); }, 1));

  mERROR_CHECK(mHardwareWindow_Create_Internal(pWindow->GetPointer(), pAllocator, title, position, size, displaymode, stereo3dIfAvailable));

  mERROR_CHECK(mRenderParams_CreateRenderContext(&(*pWindow)->renderContextID, *pWindow));

  mRenderParams_PrintRenderState(false, true);

  mGL_ERROR_CHECK();

  mUniqueContainer<mFramebuffer>::CreateWithCleanupFunction(&(*pWindow)->fakeFramebuffer, nullptr);
  (*pWindow)->fakeFramebuffer->pixelFormat = mPF_R8G8B8;
  (*pWindow)->fakeFramebuffer->sampleCount = 1;

  mERROR_CHECK(mHardwareWindow_SetAsActiveRenderTarget(*pWindow));

  mPRINT("GPU: ", reinterpret_cast<const char *>(glGetString(GL_VENDOR)), " ", reinterpret_cast<const char *>(glGetString(GL_RENDERER)),"\nDriver Version: ", reinterpret_cast<const char *>(glGetString(GL_VERSION)), "\nGLSL Version: ", reinterpret_cast<const char *>(glGetString(GL_SHADING_LANGUAGE_VERSION)), "\n");

  mGL_ERROR_CHECK();

  // Update Back Buffer Resolution.
  {
    mVec2s _size;
    mERROR_CHECK(mHardwareWindow_GetSize(*pWindow, &_size));

    mRenderParams_BackBufferResolution = _size;
    mRenderParams_BackBufferResolutionF = mVec2f(_size);
  }

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

  int32_t width, height;
  SDL_GetWindowSize(window->pWindow, &width, &height);

  *pSize = mVec2s(width, height);

  mRETURN_SUCCESS();
}

mFUNCTION(mHardwareWindow_SetProcessMessageQueueOnSwap, mPtr<mHardwareWindow> &window, const bool processMessageQueueOnSwap)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr, mR_ArgumentNull);

  window->processMessageQueueExplicit = !processMessageQueueOnSwap;

  mRETURN_SUCCESS();
}

mFUNCTION(mHardwareWindow_ProcessMessageQueue, mPtr<mHardwareWindow> &window)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr, mR_ArgumentNull);

  mPROFILE_SCOPED("mHardwareWindow_ProcessMessageQueue");

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

      mRenderParams_BackBufferResolution = size;
      mRenderParams_BackBufferResolutionF = mVec2f(size);

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
    else if (_event.type == SDL_SYSWMEVENT)
    {
#if defined (mPLATFORM_WINDOWS)
      if (_event.syswm.msg != nullptr && _event.syswm.msg->subsystem == SDL_SYSWM_WINDOWS)
      {
        switch (_event.syswm.msg->msg.win.msg)
        {
        case WM_SETTINGCHANGE:
        case WM_THEMECHANGED:
        {
          if (window->respectDarkMode)
          {
            const bool isDarkMode = mSystemInfo_IsDarkMode();

            if (window->isDarkMode != isDarkMode)
            {
              window->isDarkMode = isDarkMode;

              /* for now, we don't care if this succeeds. */ mHardwareWindow_UpdateWindowTitleColour_Internal(window);

              size_t onDarkModeChangedCount = 0;
              mERROR_CHECK(mQueue_GetCount(window->onDarkModeChanged, &onDarkModeChangedCount));

              for (size_t i = 0; i < onDarkModeChangedCount; ++i)
              {
                std::function<mResult(const bool)> *pFunction = nullptr;
                mERROR_CHECK(mQueue_PointerAt(window->onDarkModeChanged, i, &pFunction));

                mResult result = ((*pFunction)(window->isDarkMode));

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

          break;
        }
        }
      }
#endif
    }
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mHardwareWindow_Swap, mPtr<mHardwareWindow> &window)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr, mR_ArgumentNull);

  mPROFILE_SCOPED("mHardwareWindow_Swap (Swap)");

#if defined(mRENDERER_OPENGL)
  SDL_GL_SwapWindow(window->pWindow);
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  if (!window->processMessageQueueExplicit)
    mERROR_CHECK(mHardwareWindow_ProcessMessageQueue(window));

  mRETURN_SUCCESS();
}

mFUNCTION(mHardwareWindow_SetSize, mPtr<mHardwareWindow> &window, const mVec2s &size)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr, mR_ArgumentNull);
  mERROR_IF(size.x > INT_MAX || size.y > INT_MAX, mR_ArgumentOutOfBounds);

  SDL_SetWindowSize(window->pWindow, (int32_t)size.x, (int32_t)size.y);

  // Update Back Buffer Resolution.
  {
    mVec2s _size;
    mERROR_CHECK(mHardwareWindow_GetSize(window, &_size));

    mRenderParams_BackBufferResolution = _size;
    mRenderParams_BackBufferResolutionF = mVec2f(_size);
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mHardwareWindow_SetPosition, mPtr<mHardwareWindow> &window, const mVec2i &position)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr, mR_ArgumentNull);
  mERROR_IF(position.x > INT_MAX || position.y > INT_MAX || position.x < INT_MIN || position.y < INT_MIN, mR_ArgumentOutOfBounds);

  SDL_SetWindowPosition(window->pWindow, (int32_t)position.x, (int32_t)position.y);

  mRETURN_SUCCESS();
}

mFUNCTION(mHardwareWindow_SetAsActiveRenderTarget, mPtr<mHardwareWindow> &window)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mFramebuffer_Unbind());
  mERROR_CHECK(mRenderParams_ActivateRenderContext(window, window->renderContextID));

  mERROR_CHECK(mHardwareWindow_GetSize(window, &window->fakeFramebuffer->size));
  mERROR_CHECK(mFramebuffer_Push(window->fakeFramebuffer));

  mRETURN_SUCCESS();
}

mFUNCTION(mHardwareWindow_GetSdlWindowPtr, mPtr<mHardwareWindow> &window, OUT SDL_Window **ppSdlWindow)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr || ppSdlWindow == nullptr, mR_ArgumentNull);

  *ppSdlWindow = window->pWindow;

  mRETURN_SUCCESS();
}

#if defined(mPLATFORM_WINDOWS)
mFUNCTION(mHardwareWindow_GetWindowHandle, mPtr<mHardwareWindow> &window, OUT HWND *pHWND)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr || pHWND == nullptr, mR_ArgumentNull);

  SDL_SysWMinfo wmInfo;
  SDL_VERSION(&wmInfo.version);
  mERROR_IF(SDL_FALSE == SDL_GetWindowWMInfo(window->pWindow, &wmInfo), mR_NotSupported);

  HWND hwnd = wmInfo.info.win.window;
  mERROR_IF(hwnd == nullptr, mR_NotSupported);

  *pHWND = hwnd;

  mRETURN_SUCCESS();
}
#endif

mFUNCTION(mHardwareWindow_GetRenderContextId, mPtr<mHardwareWindow> &window, OUT mRenderContextId *pRenderContextId)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr || pRenderContextId == nullptr, mR_ArgumentNull);

  *pRenderContextId = window->renderContextID;

  mRETURN_SUCCESS();
}

mFUNCTION(mHardwareWindow_SetFullscreenMode, mPtr<mHardwareWindow> &window, const mHardwareWindow_DisplayMode displayMode)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr, mR_ArgumentNull);

  mERROR_IF(0 != SDL_SetWindowFullscreen(window->pWindow, displayMode), mR_NotSupported);

  mRETURN_SUCCESS();
}

mFUNCTION(mHardwareWindow_MaximizeWindow, mPtr<mHardwareWindow> &window)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr, mR_ArgumentNull);

  SDL_MaximizeWindow(window->pWindow);

  mRETURN_SUCCESS();
}

mFUNCTION(mHardwareWindow_MinimizeWindow, mPtr<mHardwareWindow> &window)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr, mR_ArgumentNull);

  SDL_MinimizeWindow(window->pWindow);

  mRETURN_SUCCESS();
}

mFUNCTION(mHardwareWindow_RestoreWindow, mPtr<mHardwareWindow> &window)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr, mR_ArgumentNull);

  SDL_RestoreWindow(window->pWindow);

  mRETURN_SUCCESS();
}

mFUNCTION(mHardwareWindow_SetResizable, mPtr<mHardwareWindow> &window, const bool resizable)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr, mR_ArgumentNull);

  SDL_SetWindowResizable(window->pWindow, resizable ? SDL_TRUE : SDL_FALSE);

  mRETURN_SUCCESS();
}

mFUNCTION(mHardwareWindow_AddOnResizeEvent, mPtr<mHardwareWindow> &window, const std::function<mResult(const mVec2s&)> &callback)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr || callback == nullptr, mR_ArgumentNull);

  std::function<mResult(const mVec2s&)> function = callback;

  mERROR_CHECK(mQueue_PushBack(window->onResizeCallbacks, &function));

  mRETURN_SUCCESS();
}

mFUNCTION(mHardwareWindow_AddOnCloseEvent, mPtr<mHardwareWindow> &window, const std::function<mResult(void)> &callback)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr || callback == nullptr, mR_ArgumentNull);

  std::function<mResult(void)> function = callback;

  mERROR_CHECK(mQueue_PushBack(window->onExitCallbacks, &function));

  mRETURN_SUCCESS();
}

mFUNCTION(mHardwareWindow_AddOnAnyEvent, mPtr<mHardwareWindow> &window, const std::function<mResult(IN SDL_Event*)> &callback)
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
  mERROR_CHECK(mQueue_Clear(window->onDarkModeChanged));

  mRETURN_SUCCESS();
}

mFUNCTION(mHardwareWindow_SetActive, mPtr<mHardwareWindow> &window)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr, mR_ArgumentNull);

  SDL_RaiseWindow(window->pWindow);

  mRETURN_SUCCESS();
}

mFUNCTION(mHardwareWindow_IsActive, const mPtr<mHardwareWindow> &window, OUT bool *pIsActive)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr || pIsActive == nullptr, mR_ArgumentNull);

  const uint32_t flags = SDL_GetWindowFlags(window->pWindow);

  *pIsActive = !!(flags & SDL_WINDOW_INPUT_FOCUS);

  mRETURN_SUCCESS();
}

mFUNCTION(mHardwareWindow_IsResizable, const mPtr<mHardwareWindow> &window, OUT bool *pIsResizable)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr || pIsResizable == nullptr, mR_ArgumentNull);

  const uint32_t flags = SDL_GetWindowFlags(window->pWindow);

  *pIsResizable = !!(flags & SDL_WINDOW_RESIZABLE);

  mRETURN_SUCCESS();
}

mFUNCTION(mHardwareWindow_IsMinimized, const mPtr<mHardwareWindow> &window, OUT bool *pIsMinimized)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr || pIsMinimized == nullptr, mR_ArgumentNull);

  const uint32_t flags = SDL_GetWindowFlags(window->pWindow);

  *pIsMinimized = !!(flags & SDL_WINDOW_MINIMIZED);

  mRETURN_SUCCESS();
}

mFUNCTION(mHardwareWindow_IsMaximized, const mPtr<mHardwareWindow> &window, OUT bool *pIsMaximized)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr || pIsMaximized == nullptr, mR_ArgumentNull);

  const uint32_t flags = SDL_GetWindowFlags(window->pWindow);

  *pIsMaximized = !!(flags & SDL_WINDOW_MAXIMIZED);

  mRETURN_SUCCESS();
}

// This function also has a sibling in `mSoftwareWindow_`. When fixing bugs or adding features, please add the respected changes to `mSoftwareWindow_` as well.
mFUNCTION(mHardwareWindow_RespectDarkModePreference, mPtr<mHardwareWindow> &window)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr, mR_ArgumentNull);

#if defined(mPLATFORM_WINDOWS)
  mERROR_IF(!mSystemInfo_IsWindows10OrGreater(17763), mR_NotSupported);

  window->isDarkMode = mSystemInfo_IsDarkMode();

  mERROR_CHECK(mHardwareWindow_UpdateWindowTitleColour_Internal(window));

  window->respectDarkMode = true;

  // HACK: Because this sometimes (!) leads to only the title text being black, we're minimizing and restoring the window in order to "refresh" the title screen.
  const uint32_t flags = SDL_GetWindowFlags(window->pWindow);

  if ((flags & SDL_WINDOW_MINIMIZED) || (flags & SDL_WINDOW_BORDERLESS) || (flags & SDL_WINDOW_FULLSCREEN) || (flags & SDL_WINDOW_FULLSCREEN_DESKTOP))
    mRETURN_SUCCESS();

  SDL_MinimizeWindow(window->pWindow);

  if (flags & SDL_WINDOW_MAXIMIZED)
    SDL_MaximizeWindow(window->pWindow);
  else
    SDL_RestoreWindow(window->pWindow);

  mRETURN_SUCCESS();
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif
}

mFUNCTION(mHardwareWindow_AddOnDarkModeChangedEvent, mPtr<mHardwareWindow> &window, const std::function<mResult(const bool isDarkMode)> &callback)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr || callback == nullptr, mR_ArgumentNull);

  std::function<mResult(const bool)> function = callback;

  mERROR_CHECK(mQueue_PushBack(window->onDarkModeChanged, &function));

  mRETURN_SUCCESS();
}

mFUNCTION(mHardwareWindow_GetWindowDPIScalingFactor, mPtr<mHardwareWindow> &window, OUT float_t *pScalingFactor)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr || pScalingFactor == nullptr, mR_ArgumentNull);

  HMODULE User32_dll = LoadLibraryW(L"User32.dll");
  mERROR_IF(User32_dll == INVALID_HANDLE_VALUE || User32_dll == NULL, mR_NotSupported);

  typedef UINT (* GetDpiForWindowFunc)(
    HWND hwnd
  );

  GetDpiForWindowFunc pGetDpiForWindow = reinterpret_cast<GetDpiForWindowFunc>(GetProcAddress(User32_dll, "GetDpiForWindow"));
  mERROR_IF(pGetDpiForWindow == nullptr, mR_NotSupported);

  HWND hwnd;
  mERROR_CHECK(mHardwareWindow_GetWindowHandle(window, &hwnd));

  const UINT dpi = pGetDpiForWindow(hwnd);

  *pScalingFactor = ((float_t)dpi / 96.f);

  mRETURN_SUCCESS();
}

mFUNCTION(mHardwareWindow_EnableDragAndDrop)
{
  mFUNCTION_SETUP();

  SDL_EventState(SDL_DROPFILE, SDL_ENABLE);
  SDL_EventState(SDL_DROPTEXT, SDL_ENABLE);

  // Now allow drag and drop if the application is privileged, but the user isn't Administrator.
  {
    ChangeWindowMessageFilter(WM_DROPFILES, MSGFLT_ADD);
    ChangeWindowMessageFilter(WM_COPYDATA, MSGFLT_ADD);
    ChangeWindowMessageFilter(0x0049, MSGFLT_ADD); // don't ask. but it doesn't work without this.
  }

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

static mFUNCTION(mHardwareWindow_Create_Internal, IN_OUT mHardwareWindow *pWindow, IN mAllocator *pAllocator, const mString &title, const mVec2i &position, const mVec2s &size, const mHardwareWindow_DisplayMode displaymode, const bool stereo3dIfAvailable)
{
  mFUNCTION_SETUP();

  mERROR_IF(size.x > INT_MAX || size.y > INT_MAX, mR_ArgumentOutOfBounds);
  mERROR_IF(position.x > INT_MAX || position.y > INT_MAX || position.x < INT_MIN || position.y < INT_MIN, mR_ArgumentOutOfBounds);

  const mHardwareWindow_DisplayMode innerDisplayMode = ((displaymode & SDL_WINDOW_FULLSCREEN_DESKTOP) == SDL_WINDOW_FULLSCREEN_DESKTOP) ? (mHardwareWindow_DisplayMode)((displaymode ^ SDL_WINDOW_FULLSCREEN_DESKTOP) | SDL_WINDOW_MAXIMIZED | SDL_WINDOW_BORDERLESS | SDL_WINDOW_RESIZABLE) : displaymode;

  bool try3d = stereo3dIfAvailable;
retry_without_3d:
  SDL_GL_SetAttribute(SDL_GL_STEREO, try3d ? 1 : 0);

  pWindow->pWindow = SDL_CreateWindow(title.c_str(), (int32_t)position.x, (int32_t)position.y, (int32_t)size.x, (int32_t)size.y
    , 
#if defined(mRENDERER_OPENGL)
    SDL_WINDOW_OPENGL |
#endif
    innerDisplayMode);

  if (pWindow->pWindow == nullptr && try3d)
  {
    try3d = false;
    goto retry_without_3d;
  }

  mERROR_IF(pWindow->pWindow == nullptr, mR_InternalError);

  if (displaymode != innerDisplayMode)
  {
    if ((displaymode & SDL_WINDOW_RESIZABLE) == 0)
      SDL_SetWindowResizable(pWindow->pWindow, SDL_FALSE);

    SDL_MaximizeWindow(pWindow->pWindow);
  }

  // Try to register for SDL_SYSWMEVENTs.
  SDL_EventState(SDL_SYSWMEVENT, SDL_ENABLE);

  if ((displaymode & (SDL_WINDOW_FULLSCREEN | SDL_WINDOW_FULLSCREEN_DESKTOP)) != 0)
    SDL_RaiseWindow(pWindow->pWindow);

  mERROR_CHECK(mQueue_Create(&pWindow->onEventCallbacks, pAllocator));
  mERROR_CHECK(mQueue_Create(&pWindow->onResizeCallbacks, pAllocator));
  mERROR_CHECK(mQueue_Create(&pWindow->onDarkModeChanged, pAllocator));
  mERROR_CHECK(mQueue_Create(&pWindow->onExitCallbacks, pAllocator));

  mRETURN_SUCCESS();
}

static mFUNCTION(mHardwareWindow_Destroy_Internal, IN mHardwareWindow *pWindow)
{
  mFUNCTION_SETUP();

  mERROR_IF(pWindow == nullptr, mR_ArgumentNull);
  
  mERROR_CHECK(mRenderParams_DestroyRenderContext(&pWindow->renderContextID));

  SDL_DestroyWindow(pWindow->pWindow);
  pWindow->pWindow = nullptr;

  mERROR_CHECK(mQueue_Destroy(&pWindow->onEventCallbacks));
  mERROR_CHECK(mQueue_Destroy(&pWindow->onResizeCallbacks));
  mERROR_CHECK(mQueue_Destroy(&pWindow->onDarkModeChanged));
  mERROR_CHECK(mQueue_Destroy(&pWindow->onExitCallbacks));

  mRETURN_SUCCESS();
}

static mFUNCTION(mHardwareWindow_UpdateWindowTitleColour_Internal, mPtr<mHardwareWindow> &window)
{
  mFUNCTION_SETUP();

  HWND hwnd;
  mERROR_CHECK(mHardwareWindow_GetWindowHandle(window, &hwnd));

  // Attention: This uses undocumented Windows features and may or may not break in the future. (There is no documented way to do this at the moment)
  enum
  {
    DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1 = 19,
    DWMWA_USE_IMMERSIVE_DARK_MODE = 20
  };

  const int32_t attribute = mSystemInfo_IsWindows10OrGreater(18985) ? DWMWA_USE_IMMERSIVE_DARK_MODE : DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_20H1;
  const int32_t setEnabled = (int32_t)mSystemInfo_IsDarkMode();

  mERROR_IF(FAILED(DwmSetWindowAttribute(hwnd, attribute, &setEnabled, sizeof(setEnabled))), mR_InternalError);

  mRETURN_SUCCESS();
}
