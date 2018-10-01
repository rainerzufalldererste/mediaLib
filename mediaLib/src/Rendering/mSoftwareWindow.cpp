#include "mSoftwareWindow.h"

struct mSoftwareWindow
{
  SDL_Window *pWindow;
  mPtr<mImageBuffer> buffer;
};

mFUNCTION(mSoftwareWindow_Create_Internal, IN mSoftwareWindow *pWindow, const std::string & title, const mVec2s & size, const mSoftwareWindow_DisplayMode displaymode);
mFUNCTION(mSoftwareWindow_Destroy_Internal, IN mSoftwareWindow *pWindow);

mFUNCTION(mSoftwareWindow_Create, OUT mPtr<mSoftwareWindow>* pWindow, IN mAllocator *pAllocator, const std::string & title, const mVec2s & size, const mSoftwareWindow_DisplayMode displaymode /* = mSW_DM_Windowed */)
{
  mFUNCTION_SETUP();

  mERROR_IF(SDL_Init(SDL_INIT_EVERYTHING), mR_InternalError);

  mERROR_CHECK(mSharedPointer_Allocate(pWindow, pAllocator, (std::function<void (mSoftwareWindow *)>)[](mSoftwareWindow *pData) { mSoftwareWindow_Destroy_Internal(pData); }, 1));

  mERROR_CHECK(mSoftwareWindow_Create_Internal(pWindow->GetPointer(), title, size, displaymode));

  mRETURN_SUCCESS();
}

mFUNCTION(mSoftwareWindow_Destroy, IN_OUT mPtr<mSoftwareWindow>* pWindow)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mSharedPointer_Destroy(pWindow));

  mRETURN_SUCCESS();
}

mFUNCTION(mSoftwareWindow_GetSize, mPtr<mSoftwareWindow>& window, OUT mVec2s * pSize)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr || pSize == nullptr, mR_ArgumentNull);

  int w, h;
  SDL_GetWindowSize(window->pWindow, &w, &h);

  *pSize = mVec2s(w, h);

  mRETURN_SUCCESS();
}

mFUNCTION(mSoftwareWindow_GetBuffer, mPtr<mSoftwareWindow>& window, OUT mPtr<mImageBuffer>* pBuffer)
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

mFUNCTION(mSoftwareWindow_Swap, mPtr<mSoftwareWindow>& window)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr, mR_ArgumentNull);

  SDL_UpdateWindowSurface(window->pWindow);

  mRETURN_SUCCESS();
}

mFUNCTION(mSoftwareWindow_SetSize, mPtr<mSoftwareWindow>& window, const mVec2s & size)
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

mFUNCTION(mSoftwareWindow_GetSdlWindowPtr, mPtr<mSoftwareWindow>& window, OUT SDL_Window ** ppSdlWindow)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr || ppSdlWindow == nullptr, mR_ArgumentNull);

  *ppSdlWindow = window->pWindow;

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mSoftwareWindow_Create_Internal, IN mSoftwareWindow * pWindow, const std::string & title, const mVec2s & size, const mSoftwareWindow_DisplayMode displaymode)
{
  mFUNCTION_SETUP();

  mERROR_IF(size.x > INT_MAX || size.y > INT_MAX, mR_ArgumentOutOfBounds);
  mERROR_IF((displaymode & SDL_WINDOW_OPENGL) != 0, mR_OperationNotSupported);

  pWindow->pWindow = SDL_CreateWindow(title.c_str(), SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, (int)size.x, (int)size.y, displaymode);
    mERROR_IF(pWindow->pWindow == nullptr, mR_InternalError);

  mRETURN_SUCCESS();
}

mFUNCTION(mSoftwareWindow_Destroy_Internal, IN mSoftwareWindow * pWindow)
{
  mFUNCTION_SETUP();

  mERROR_IF(pWindow == nullptr, mR_ArgumentNull);

  if (pWindow->buffer != nullptr)
    mERROR_CHECK(mImageBuffer_Destroy(&pWindow->buffer));

  SDL_DestroyWindow(pWindow->pWindow);
  pWindow->pWindow = nullptr;

  mRETURN_SUCCESS();
}
