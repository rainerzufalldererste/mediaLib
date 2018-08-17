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
  mDEFER_DESTRUCTION(pWindow, SDL_DestroyWindow);
  pWindow = SDL_CreateWindow("VideoStream Renderer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, (int)resolution.x, (int)resolution.y, 0);
  mERROR_IF(pWindow == nullptr, mR_ArgumentNull);
  mERROR_CHECK(ResizeWindow(resolution));

  mPtr<mImageBuffer> imageBuffer;
  mDEFER_DESTRUCTION(&imageBuffer, mImageBuffer_Destroy);
  mERROR_CHECK(mImageBuffer_CreateFromFile(&imageBuffer, nullptr, "C:/Users/cstiller/Pictures/blom.png"));

  bool ppp = 0;

  while (true)
  {
    mERROR_CHECK(mMemset(pPixels, resolution.x * resolution.y, ppp * 0xFF));

    mVec2f corner0(200, 200), corner2(11, 912), corner1(920, 11), corner3(200, 800);
    mVec2f resf = (mVec2f)resolution;

    for (float_t y = 0; y < 1; y += 1.0f / resf.y)
    {
      for (float_t x = 0; x < 1; x += 1.0f / resf.x)
      {
        const mVec2s pos = (mVec2s)(mBiLerp(corner0, corner1, corner2, corner3, x, y));

        if(pos.x < resolution.x && pos.y < resolution.y)
          pPixels[pos.x + (resolution.x * pos.y)] = ((uint32_t *)(imageBuffer->pPixels))[(size_t)(x * imageBuffer->currentSize.x) + imageBuffer->currentSize.x * (size_t)(y * imageBuffer->currentSize.y)];
      }
    }

    ppp = !ppp;

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
