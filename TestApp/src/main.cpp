#include "default.h"
#include "mVideoPlaybackEngine.h"
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

  mPtr<mVideoPlaybackEngine> playbackEngine;
  mDEFER_DESTRUCTION(&playbackEngine, mVideoPlaybackEngine_Destroy);
  mERROR_CHECK(mVideoPlaybackEngine_Create(&playbackEngine, nullptr, L"C:/Users/cstiller/Videos/Converted.mp4", threadPool));

  mVec2s resolution(1024, 1024);
  const size_t subScale = 5;
  mDEFER_DESTRUCTION(pWindow, SDL_DestroyWindow);
  pWindow = SDL_CreateWindow("VideoStream Renderer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, (int)resolution.x / subScale, (int)resolution.y / subScale, 0);
  mERROR_IF(pWindow == nullptr, mR_ArgumentNull);
  mERROR_CHECK(ResizeWindow(resolution / subScale));

  while (true)
  {
    mPtr<mImageBuffer> imageBuffer;
    mDEFER_DESTRUCTION(&imageBuffer, mImageBuffer_Destroy);

    bool isNewFrame = false;
    const mResult result = mVideoPlaybackEngine_GetCurrentFrame(playbackEngine, &imageBuffer, &isNewFrame);

    if (!isNewFrame)
    {
      mERROR_CHECK(mSleep(5));
      continue;
    }

    if(result == mR_EndOfStream)
      break;
    
    mERROR_IF(mFAILED(result), result);

    if (resolution != imageBuffer->currentSize)
    {
      resolution = imageBuffer->currentSize;
      ResizeWindow(imageBuffer->currentSize / subScale);
    }

    size_t i = 0;
    size_t height = 0;

    mVec2f corner0(20, 14), corner2(11, 712), corner1(720, 11), corner3(600, 600);
    mVec2s res = resolution / subScale;
    mVec2f resf = (mVec2f)res;

    for (float_t y = 0; y < 1; y += 1.0f / resf.y)
    {
      for (float_t x = 0; x < 1; x += 1.0f / resf.x)
      {
        const mVec2s pos = (mVec2s)(mInterpolateQuad(corner0, corner1, corner2, corner3, x, y));

        if(pos.x < res.x && pos.y < res.y)
          pPixels[pos.x + (res.x * pos.y)] = ((uint32_t *)(imageBuffer->pPixels))[height + (i += subScale)];
      }

      height += subScale * res.x;
      i = 0;
    }

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
