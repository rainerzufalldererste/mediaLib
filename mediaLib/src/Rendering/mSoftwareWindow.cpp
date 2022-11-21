#include "mSoftwareWindow.h"
#include "mSystemInfo.h"

#define DECLSPEC
#include "SDL_syswm.h"
#undef DECLSPEC

#include <dwmapi.h>
#pragma comment(lib, "dwmapi.lib")

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "JpqmGb49shQVGk07P9PO3bSxk270RBJCq3gdUJAuy/dSctxGkwd1ydrYVyqMI/XNT2u1MmVrGpdjWEVM"
#endif

struct mSoftwareWindow
{
  SDL_Window *pWindow;
  mPtr<mImageBuffer> buffer;
  mPtr<mQueue<std::function<mResult(IN SDL_Event *)>>> onEventCallbacks;
  mPtr<mQueue<std::function<mResult(void)>>> onExitCallbacks;
  mPtr<mQueue<std::function<mResult(const mVec2s &)>>> onResizeCallbacks;
  mPtr<mQueue<std::function<mResult(const bool)>>> onDarkModeChanged;
  bool respectDarkMode, isDarkMode;
};

static mFUNCTION(mSoftwareWindow_Create_Internal, IN mSoftwareWindow *pWindow, IN mAllocator *pAllocator, const mString &title, const mVec2i &position, const mVec2s &size, const mSoftwareWindow_DisplayMode displaymode);
static mFUNCTION(mSoftwareWindow_Destroy_Internal, IN mSoftwareWindow *pWindow);
static mFUNCTION(mSoftwareWindow_UpdateWindowTitleColour_Internal, mPtr<mSoftwareWindow> &window);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mSoftwareWindow_Create, OUT mPtr<mSoftwareWindow> *pWindow, IN mAllocator *pAllocator, const mString &title, const mVec2s &size, const mSoftwareWindow_DisplayMode displaymode /* = mSW_DM_Windowed */)
{
  return mSoftwareWindow_Create(pWindow, pAllocator, title, mVec2i(SDL_WINDOWPOS_CENTERED), size, displaymode);
}

mFUNCTION(mSoftwareWindow_Create, OUT mPtr<mSoftwareWindow> *pWindow, IN mAllocator *pAllocator, const mString &title, const mVec2i &position, const mVec2s &size, const mSoftwareWindow_DisplayMode displaymode /* = mSW_DM_Windowed */)
{
  mFUNCTION_SETUP();

  mERROR_IF(SDL_Init(SDL_INIT_EVERYTHING), mR_InternalError);

  mDEFER_CALL_ON_ERROR(pWindow, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Allocate(pWindow, pAllocator, (std::function<void (mSoftwareWindow *)>)[](mSoftwareWindow *pData) { mSoftwareWindow_Destroy_Internal(pData); }, 1));

  mERROR_CHECK(mSoftwareWindow_Create_Internal(pWindow->GetPointer(), pAllocator, title, position, size, displaymode));

  mRETURN_SUCCESS();
}

mFUNCTION(mSoftwareWindow_Destroy, IN_OUT mPtr<mSoftwareWindow> *pWindow)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mSharedPointer_Destroy(pWindow));

  mRETURN_SUCCESS();
}

mFUNCTION(mSoftwareWindow_GetSize, mPtr<mSoftwareWindow> &window, OUT mVec2s *pSize)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr || pSize == nullptr, mR_ArgumentNull);

  int w, h;
  SDL_GetWindowSize(window->pWindow, &w, &h);

  *pSize = mVec2s(w, h);

  mRETURN_SUCCESS();
}

mFUNCTION(mSoftwareWindow_GetBuffer, mPtr<mSoftwareWindow> &window, OUT mPtr<mImageBuffer> *pBuffer)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr || pBuffer == nullptr, mR_ArgumentNull);

  if (window->buffer == nullptr)
  {
    SDL_Surface *pSurface = SDL_GetWindowSurface(window->pWindow);
    mERROR_IF(pSurface == nullptr, mR_InternalError);

    mERROR_CHECK(mImageBuffer_Create(&window->buffer, nullptr, pSurface->pixels, mVec2s(pSurface->w, pSurface->h), mPF_B8G8R8A8));
  }

  *pBuffer = window->buffer;

  mRETURN_SUCCESS();
}

mFUNCTION(mSoftwareWindow_GetPixelPtr, mPtr<mSoftwareWindow> &window, OUT uint32_t **ppPixels)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr || ppPixels == nullptr, mR_ArgumentNull);

  *ppPixels = reinterpret_cast<uint32_t *>(SDL_GetWindowSurface(window->pWindow)->pixels);

  mRETURN_SUCCESS();
}

mFUNCTION(mSoftwareWindow_Swap, mPtr<mSoftwareWindow> &window)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr, mR_ArgumentNull);

  SDL_UpdateWindowSurface(window->pWindow);

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
      mERROR_CHECK(mSoftwareWindow_GetSize(window, &size));

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

              /* for now, we don't care if this succeeds. */ mSoftwareWindow_UpdateWindowTitleColour_Internal(window);

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

mFUNCTION(mSoftwareWindow_SetSize, mPtr<mSoftwareWindow> &window, const mVec2s &size)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr, mR_ArgumentNull);
  mERROR_IF(size.x > INT_MAX || size.y > INT_MAX, mR_ArgumentOutOfBounds);

  SDL_SetWindowSize(window->pWindow, (int)size.x , (int)size.y);

  if (window->buffer != nullptr) // update *all* image buffers.
  {
    SDL_Surface *pSurface = SDL_GetWindowSurface(window->pWindow);

    window->buffer->allocatedSize = pSurface->w * pSurface->h * sizeof(uint32_t);
    window->buffer->currentSize = mVec2s(pSurface->w, pSurface->h);
    window->buffer->lineStride = pSurface->w;
    window->buffer->pixelFormat = mPF_B8G8R8A8;
    window->buffer->pPixels = (uint8_t *)pSurface->pixels;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mSoftwareWindow_SetPosition, mPtr<mSoftwareWindow> &window, const mVec2i &position)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr, mR_ArgumentNull);
  mERROR_IF(position.x > INT_MAX || position.y > INT_MAX || position.x < INT_MIN || position.y < INT_MIN, mR_ArgumentOutOfBounds);

  SDL_SetWindowPosition(window->pWindow, (int32_t)position.x, (int32_t)position.y);

  mRETURN_SUCCESS();
}

mFUNCTION(mSoftwareWindow_GetSdlWindowPtr, mPtr<mSoftwareWindow> &window, OUT SDL_Window **ppSdlWindow)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr || ppSdlWindow == nullptr, mR_ArgumentNull);

  *ppSdlWindow = window->pWindow;

  mRETURN_SUCCESS();
}

#if defined(mPLATFORM_WINDOWS)
mFUNCTION(mSoftwareWindow_GetWindowHandle, mPtr<mSoftwareWindow> &window, OUT HWND *pHWND)
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

mFUNCTION(mSoftwareWindow_SetFullscreenMode, mPtr<mSoftwareWindow> &window, const mSoftwareWindow_DisplayMode displayMode)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr, mR_ArgumentNull);

  mERROR_IF(0 != SDL_SetWindowFullscreen(window->pWindow, displayMode), mR_NotSupported);

  mRETURN_SUCCESS();
}

mFUNCTION(mSoftwareWindow_MaximizeWindow, mPtr<mSoftwareWindow> &window)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr, mR_ArgumentNull);

  SDL_MaximizeWindow(window->pWindow);

  mRETURN_SUCCESS();
}

mFUNCTION(mSoftwareWindow_MinimizeWindow, mPtr<mSoftwareWindow> &window)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr, mR_ArgumentNull);

  SDL_MinimizeWindow(window->pWindow);

  mRETURN_SUCCESS();
}

mFUNCTION(mSoftwareWindow_RestoreWindow, mPtr<mSoftwareWindow> &window)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr, mR_ArgumentNull);

  SDL_RestoreWindow(window->pWindow);

  mRETURN_SUCCESS();
}

mFUNCTION(mSoftwareWindow_SetResizable, mPtr<mSoftwareWindow> &window, const bool resizable)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr, mR_ArgumentNull);

  SDL_SetWindowResizable(window->pWindow, resizable ? SDL_TRUE : SDL_FALSE);

  mRETURN_SUCCESS();
}

mFUNCTION(mSoftwareWindow_SetTitle, mPtr<mSoftwareWindow> &window, const mString &title)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr, mR_ArgumentNull);
  mERROR_IF(title.hasFailed, mR_InvalidParameter);

  SDL_SetWindowTitle(window->pWindow, title.c_str());

  mRETURN_SUCCESS();
}

mFUNCTION(mSoftwareWindow_AddOnResizeEvent, mPtr<mSoftwareWindow> &window, const std::function<mResult(const mVec2s&)> &callback)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr || callback == nullptr, mR_ArgumentNull);

  std::function<mResult(const mVec2s&)> function = callback;

  mERROR_CHECK(mQueue_PushBack(window->onResizeCallbacks, &function));

  mRETURN_SUCCESS();
}

mFUNCTION(mSoftwareWindow_AddOnCloseEvent, mPtr<mSoftwareWindow> &window, const std::function<mResult(void)> &callback)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr || callback == nullptr, mR_ArgumentNull);

  std::function<mResult(void)> function = callback;

  mERROR_CHECK(mQueue_PushBack(window->onExitCallbacks, &function));

  mRETURN_SUCCESS();
}

mFUNCTION(mSoftwareWindow_AddOnAnyEvent, mPtr<mSoftwareWindow> &window, const std::function<mResult(IN SDL_Event*)> &callback)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr || callback == nullptr, mR_ArgumentNull);

  std::function<mResult(IN SDL_Event*)> function = callback;

  mERROR_CHECK(mQueue_PushBack(window->onEventCallbacks, &function));

  mRETURN_SUCCESS();
}

mFUNCTION(mSoftwareWindow_ClearEventHandlers, mPtr<mSoftwareWindow> &window)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mQueue_Clear(window->onEventCallbacks));
  mERROR_CHECK(mQueue_Clear(window->onExitCallbacks));
  mERROR_CHECK(mQueue_Clear(window->onResizeCallbacks));
  mERROR_CHECK(mQueue_Clear(window->onDarkModeChanged));

  mRETURN_SUCCESS();
}

mFUNCTION(mSoftwareWindow_SetActive, mPtr<mSoftwareWindow> &window)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr, mR_ArgumentNull);

  SDL_RaiseWindow(window->pWindow);

  mRETURN_SUCCESS();
}

mFUNCTION(mSoftwareWindow_IsActive, const mPtr<mSoftwareWindow> &window, OUT bool *pIsActive)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr || pIsActive == nullptr, mR_ArgumentNull);

  const uint32_t flags = SDL_GetWindowFlags(window->pWindow);

  *pIsActive = !!(flags & SDL_WINDOW_INPUT_FOCUS);

  mRETURN_SUCCESS();
}

mFUNCTION(mSoftwareWindow_IsResizable, const mPtr<mSoftwareWindow> &window, OUT bool *pIsResizable)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr || pIsResizable == nullptr, mR_ArgumentNull);

  const uint32_t flags = SDL_GetWindowFlags(window->pWindow);

  *pIsResizable = !!(flags & SDL_WINDOW_RESIZABLE);

  mRETURN_SUCCESS();
}

mFUNCTION(mSoftwareWindow_IsMinimized, const mPtr<mSoftwareWindow> &window, OUT bool *pIsMinimized)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr || pIsMinimized == nullptr, mR_ArgumentNull);

  const uint32_t flags = SDL_GetWindowFlags(window->pWindow);

  *pIsMinimized = !!(flags & SDL_WINDOW_MINIMIZED);

  mRETURN_SUCCESS();
}

mFUNCTION(mSoftwareWindow_IsMaximized, const mPtr<mSoftwareWindow> &window, OUT bool *pIsMaximized)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr || pIsMaximized == nullptr, mR_ArgumentNull);

  const uint32_t flags = SDL_GetWindowFlags(window->pWindow);

  *pIsMaximized = !!(flags & SDL_WINDOW_MAXIMIZED);

  mRETURN_SUCCESS();
}

// This function also has a sibling in `mHardwareWindow`. When fixing bugs or adding features, please add the respected changes to `mHardwareWindow` as well.
mFUNCTION(mSoftwareWindow_RespectDarkModePreference, mPtr<mSoftwareWindow> &window)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr, mR_ArgumentNull);

#if defined(mPLATFORM_WINDOWS)
  mERROR_IF(!mSystemInfo_IsWindows10OrGreater(17763), mR_NotSupported);

  window->isDarkMode = mSystemInfo_IsDarkMode();

  mERROR_CHECK(mSoftwareWindow_UpdateWindowTitleColour_Internal(window));

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

mFUNCTION(mSoftwareWindow_AddOnDarkModeChangedEvent, mPtr<mSoftwareWindow> &window, const std::function<mResult(const bool isDarkMode)> &callback)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr || callback == nullptr, mR_ArgumentNull);

  std::function<mResult(const bool)> function = callback;

  mERROR_CHECK(mQueue_PushBack(window->onDarkModeChanged, &function));

  mRETURN_SUCCESS();
}

mFUNCTION(mSoftwareWindow_GetWindowDPIScalingFactor, mPtr<mSoftwareWindow> &window, OUT float_t *pScalingFactor)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr || pScalingFactor == nullptr, mR_ArgumentNull);

  HMODULE User32_dll = LoadLibraryW(L"User32.dll");
  mERROR_IF(User32_dll == INVALID_HANDLE_VALUE || User32_dll == NULL, mR_NotSupported);

  typedef UINT(*GetDpiForWindowFunc)(
    HWND hwnd
    );

  GetDpiForWindowFunc pGetDpiForWindow = reinterpret_cast<GetDpiForWindowFunc>(GetProcAddress(User32_dll, "GetDpiForWindow"));
  mERROR_IF(pGetDpiForWindow == nullptr, mR_NotSupported);

  HWND hwnd;
  mERROR_CHECK(mSoftwareWindow_GetWindowHandle(window, &hwnd));

  const UINT dpi = pGetDpiForWindow(hwnd);

  *pScalingFactor = ((float_t)dpi / 96.f);

  mRETURN_SUCCESS();
}

mFUNCTION(mSoftwareWindow_EnableDragAndDrop)
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

static mFUNCTION(mSoftwareWindow_Create_Internal, IN mSoftwareWindow *pWindow, IN mAllocator *pAllocator, const mString &title, const mVec2i &position, const mVec2s &size, const mSoftwareWindow_DisplayMode displaymode)
{
  mFUNCTION_SETUP();

  mERROR_IF(size.x > INT_MAX || size.y > INT_MAX, mR_ArgumentOutOfBounds);
  mERROR_IF(position.x > INT_MAX || position.y > INT_MAX || position.x < INT_MIN || position.y < INT_MIN, mR_ArgumentOutOfBounds);
  mERROR_IF((displaymode & SDL_WINDOW_OPENGL) != 0, mR_OperationNotSupported);

  const mSoftwareWindow_DisplayMode innerDisplayMode = ((displaymode & SDL_WINDOW_FULLSCREEN_DESKTOP) == SDL_WINDOW_FULLSCREEN_DESKTOP) ? (mSoftwareWindow_DisplayMode)((displaymode ^ SDL_WINDOW_FULLSCREEN_DESKTOP) | SDL_WINDOW_MAXIMIZED | SDL_WINDOW_BORDERLESS | SDL_WINDOW_RESIZABLE) : displaymode;

  pWindow->pWindow = SDL_CreateWindow(title.c_str(), (int32_t)position.x, (int32_t)position.y, (int32_t)size.x, (int32_t)size.y, innerDisplayMode);
  
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

static mFUNCTION(mSoftwareWindow_Destroy_Internal, IN mSoftwareWindow *pWindow)
{
  mFUNCTION_SETUP();

  mERROR_IF(pWindow == nullptr, mR_ArgumentNull);

  if (pWindow->buffer != nullptr)
    mERROR_CHECK(mImageBuffer_Destroy(&pWindow->buffer));

  SDL_DestroyWindow(pWindow->pWindow);
  pWindow->pWindow = nullptr;

  mERROR_CHECK(mQueue_Destroy(&pWindow->onEventCallbacks));
  mERROR_CHECK(mQueue_Destroy(&pWindow->onResizeCallbacks));
  mERROR_CHECK(mQueue_Destroy(&pWindow->onDarkModeChanged));
  mERROR_CHECK(mQueue_Destroy(&pWindow->onExitCallbacks));

  mRETURN_SUCCESS();
}

static mFUNCTION(mSoftwareWindow_UpdateWindowTitleColour_Internal, mPtr<mSoftwareWindow> &window)
{
  mFUNCTION_SETUP();

  HWND hwnd;
  mERROR_CHECK(mSoftwareWindow_GetWindowHandle(window, &hwnd));

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
