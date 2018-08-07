#include "default.h"
#include "mMediaFileInputHandler.h"
#include "SDL.h"

mVec2s resolution;
SDL_Window *pWindow = nullptr;
uint32_t *pPixels = nullptr;
const size_t subScale = 6;

mFUNCTION(OnVideoFramCallback, IN uint8_t *pBuffer, const mVideoStreamType &videoStreamType);

int main(int, char **)
{
  mFUNCTION_SETUP();

  g_mResult_breakOnError = true;
  mPtr<mMediaFileInputHandler> mediaFileHandler;
  mDEFER_DESTRUCTION(&mediaFileHandler, mMediaFileInputHandler_Destroy);
  mERROR_CHECK(mMediaFileInputHandler_Create(&mediaFileHandler, L"C:/Users/cstiller/Videos/Converted.MP4", mMediaFileInputHandler_CreateFlags::mMMFIH_CF_VideoEnabled));

  mERROR_CHECK(mMediaFileInputHandler_GetVideoStreamResolution(mediaFileHandler, &resolution));

  mDEFER_DESTRUCTION(pWindow, SDL_DestroyWindow);
  pWindow = SDL_CreateWindow("HoloRoom Software Render", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, (int)resolution.x / subScale, (int)resolution.y / subScale * 3 / 2, 0);
  mERROR_IF(pWindow == nullptr, mR_ArgumentNull);
  pPixels = (uint32_t *)SDL_GetWindowSurface(pWindow)->pixels;
  mERROR_IF(pPixels == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mMediaFileInputHandler_SetVideoCallback(mediaFileHandler, OnVideoFramCallback));
  mERROR_CHECK(mMediaFileInputHandler_Play(mediaFileHandler));

  mRETURN_SUCCESS();
}

mFUNCTION(OnVideoFramCallback, IN uint8_t *pBuffer, const mVideoStreamType &videoStreamType)
{
  mFUNCTION_SETUP();

  SDL_UpdateWindowSurface(pWindow);

  size_t i = 0;

  for (size_t y = 0; y < videoStreamType.resolution.y * 3 / 2; y += subScale)
    for (size_t x = 0; x < videoStreamType.resolution.x; x += subScale)
      pPixels[i++] = 0x010101 * (uint32_t)pBuffer[x + y * videoStreamType.resolution.x];

  SDL_Event sdl_event;
  while (SDL_PollEvent(&sdl_event))
    ; // We don't care.

  mRETURN_SUCCESS();
}
