#include "default.h"
#include "mMediaFileInputHandler.h"
#include "mThreadPool.h"
#include "SDL.h"

SDL_Window *pWindow = nullptr;
uint32_t *pPixels = nullptr;

mFUNCTION(ResizeWindow, const mVec2s &newSize);

int main(int, char **)
{
  mFUNCTION_SETUP();

  mPtr<mThreadPool> threadPool = nullptr;
  g_mResult_breakOnError = true;
  mDEFER_DESTRUCTION(&threadPool, mThreadPool_Destroy);
  mERROR_CHECK(mThreadPool_Create(&threadPool, nullptr));

  mPtr<mMediaFileInputHandler> mediaFileHandler;
  mDEFER_DESTRUCTION(&mediaFileHandler, mMediaFileInputHandler_Destroy);
  mERROR_CHECK(mMediaFileInputHandler_Create(&mediaFileHandler, nullptr, L"C:/Users/cstiller/Videos/Converted.mp4", mMediaFileInputHandler_CreateFlags::mMMFIH_CF_VideoEnabled));

  mVec2s resolution;
  mERROR_CHECK(mMediaFileInputHandler_GetVideoStreamResolution(mediaFileHandler, &resolution));

  const size_t subScale = 5;
  mDEFER_DESTRUCTION(pWindow, SDL_DestroyWindow);
  pWindow = SDL_CreateWindow("VideoStream Renderer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, (int)resolution.x / subScale, (int)resolution.y / subScale, 0);
  mERROR_IF(pWindow == nullptr, mR_ArgumentNull);
  mERROR_CHECK(ResizeWindow(resolution / subScale));

  mPtr<mImageBuffer> bgraImageBuffer = nullptr;
  mDEFER_DESTRUCTION(&bgraImageBuffer, mImageBuffer_Destroy);
  mERROR_CHECK(mImageBuffer_Create(&bgraImageBuffer, nullptr, resolution));

  mPtr<mMediaFileInputIterator> iterator;
  mDEFER_DESTRUCTION(&iterator, mMediaFileInputIterator_Destroy);
  mERROR_CHECK(mMediaFileInputHandler_GetIterator(mediaFileHandler, &iterator, mMediaMajorType::mMMT_Video, 0));

  while (true)
  {
    mPtr<mImageBuffer> imageBuffer;
    mDEFER_DESTRUCTION(&imageBuffer, mImageBuffer_Destroy);

    const mResult result = mMediaFileInputIterator_GetNextVideoFrame(iterator, &imageBuffer, nullptr);

    if(result == mR_EndOfStream)
      break;
    
    mERROR_IF(mFAILED(result), result);

    if (imageBuffer->currentSize != bgraImageBuffer->currentSize)
    {
      ResizeWindow(imageBuffer->currentSize / subScale);
      mERROR_CHECK(mImageBuffer_AllocateBuffer(bgraImageBuffer, imageBuffer->currentSize, bgraImageBuffer->pixelFormat));
    }

    mERROR_CHECK(mPixelFormat_TransformBuffer(imageBuffer, bgraImageBuffer, threadPool));

    size_t i = 0;

    for (size_t y = 0; y < bgraImageBuffer->currentSize.y; y += subScale)
      for (size_t x = 0; x < bgraImageBuffer->currentSize.x; x += subScale)
        pPixels[i++] = ((uint32_t *)(bgraImageBuffer->pPixels))[x + y * bgraImageBuffer->lineStride];

    SDL_UpdateWindowSurface(pWindow);

    SDL_Event sdl_event;
    while (SDL_PollEvent(&sdl_event))
      ; // We don't care.
  }

  mRETURN_SUCCESS();
}

mFUNCTION(ResizeWindow, const mVec2s &newSize)
{
  mFUNCTION_SETUP();

  SDL_SetWindowSize(pWindow, (int)newSize.x, (int)newSize.y);
  pPixels = (uint32_t *)SDL_GetWindowSurface(pWindow)->pixels;
  mERROR_IF(pPixels == nullptr, mR_ArgumentNull);

  mRETURN_SUCCESS();
}
