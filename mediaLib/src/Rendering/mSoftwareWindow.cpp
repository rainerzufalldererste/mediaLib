#include "mSoftwareWindow.h"

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
};

mFUNCTION(mSoftwareWindow_Create_Internal, IN mSoftwareWindow *pWindow, IN mAllocator *pAllocator, const mString &title, const mVec2s &size, const mSoftwareWindow_DisplayMode displaymode);
mFUNCTION(mSoftwareWindow_Destroy_Internal, IN mSoftwareWindow *pWindow);

mFUNCTION(mSoftwareWindow_Create, OUT mPtr<mSoftwareWindow> *pWindow, IN mAllocator *pAllocator, const mString &title, const mVec2s &size, const mSoftwareWindow_DisplayMode displaymode /* = mSW_DM_Windowed */)
{
  mFUNCTION_SETUP();

  mERROR_IF(SDL_Init(SDL_INIT_EVERYTHING), mR_InternalError);

  mERROR_CHECK(mSharedPointer_Allocate(pWindow, pAllocator, (std::function<void (mSoftwareWindow *)>)[](mSoftwareWindow *pData) { mSoftwareWindow_Destroy_Internal(pData); }, 1));

  mERROR_CHECK(mSoftwareWindow_Create_Internal(pWindow->GetPointer(), pAllocator, title, size, displaymode));

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

mFUNCTION(mSoftwareWindow_GetSdlWindowPtr, mPtr<mSoftwareWindow> &window, OUT SDL_Window **ppSdlWindow)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr || ppSdlWindow == nullptr, mR_ArgumentNull);

  *ppSdlWindow = window->pWindow;

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

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mSoftwareWindow_Create_Internal, IN mSoftwareWindow *pWindow, IN mAllocator *pAllocator, const mString &title, const mVec2s &size, const mSoftwareWindow_DisplayMode displaymode)
{
  mFUNCTION_SETUP();

  mERROR_IF(size.x > INT_MAX || size.y > INT_MAX, mR_ArgumentOutOfBounds);
  mERROR_IF((displaymode & SDL_WINDOW_OPENGL) != 0, mR_OperationNotSupported);

  pWindow->pWindow = SDL_CreateWindow(title.c_str(), SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, (int)size.x, (int)size.y, displaymode);
  
  mERROR_IF(pWindow->pWindow == nullptr, mR_InternalError);

  mERROR_CHECK(mQueue_Create(&pWindow->onEventCallbacks, pAllocator));
  mERROR_CHECK(mQueue_Create(&pWindow->onResizeCallbacks, pAllocator));
  mERROR_CHECK(mQueue_Create(&pWindow->onExitCallbacks, pAllocator));

  mRETURN_SUCCESS();
}

mFUNCTION(mSoftwareWindow_Destroy_Internal, IN mSoftwareWindow *pWindow)
{
  mFUNCTION_SETUP();

  mERROR_IF(pWindow == nullptr, mR_ArgumentNull);

  if (pWindow->buffer != nullptr)
    mERROR_CHECK(mImageBuffer_Destroy(&pWindow->buffer));

  SDL_DestroyWindow(pWindow->pWindow);
  pWindow->pWindow = nullptr;

  mERROR_CHECK(mQueue_Destroy(&pWindow->onEventCallbacks));
  mERROR_CHECK(mQueue_Destroy(&pWindow->onResizeCallbacks));
  mERROR_CHECK(mQueue_Destroy(&pWindow->onExitCallbacks));

  mRETURN_SUCCESS();
}
