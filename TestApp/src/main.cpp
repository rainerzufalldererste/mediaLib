#include "default.h"
#include "mMediaFileInputHandler.h"
#include "SDL.h"

mVec2s resulution;
SDL_Window *pWindow = nullptr;
uint32_t *pPixels = nullptr;

mFUNCTION(ProcessVideoBufferCallback, IN uint8_t *pBuffer, const mVideoStreamType &videoStreamType);

int main(int, char **)
{
  mFUNCTION_SETUP();

  mVector v(0, 1, 2);
  printf("%f, %f, %f\n", v.x, v.y, v.z);
  mVector w(4, 2, -2);
  printf("%f, %f, %f\n", w.x, w.y, w.z);

  mVector r = v + w;
  printf("%f, %f, %f\n", r.x, r.y, r.z);

  g_mResult_breakOnError = true;
  mPtr<mMediaFileInputHandler> videoInput;
  mDEFER_DESTRUCTION(&videoInput, mMediaFileInputHandler_Destroy);
  mERROR_CHECK(mMediaFileInputHandler_Create(&videoInput, L"C:\\Users\\cstiller\\Videos\\Original.MP4", mMediaFileInputHandler_CreateFlags::mMMFIH_CF_AllMediaTypesEnabled));
  
  mERROR_CHECK(mMediaFileInputHandler_GetVideoStreamResolution(videoInput, &resulution));

  SDL_Init(SDL_INIT_EVERYTHING);

  pWindow = SDL_CreateWindow("VideoFrames", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, (int)resulution.x, (int)resulution.y, 0);
  mDEFER_DESTRUCTION(pWindow, SDL_DestroyWindow);
  SDL_Surface *pSurface = SDL_GetWindowSurface(pWindow);
  pPixels = (uint32_t *)pSurface->pixels;

  mERROR_CHECK(mMediaFileInputHandler_SetVideoCallback(videoInput, &ProcessVideoBufferCallback));
  mERROR_CHECK(mMediaFileInputHandler_Play(videoInput));

  mRETURN_SUCCESS();
}

mFUNCTION(ProcessVideoBufferCallback, IN uint8_t *pBuffer, const mVideoStreamType &videoStreamType)
{
  mFUNCTION_SETUP();

  if (pBuffer && videoStreamType.mediaType == mMediaMajorType::mMMT_Video)
  {
    mERROR_CHECK(mMemcpy(pPixels, (uint32_t *)pBuffer, videoStreamType.resolution.x * videoStreamType.resolution.y));

    SDL_UpdateWindowSurface(pWindow);

    SDL_Event sdl_event;
    while (SDL_PollEvent(&sdl_event))
      ; // We don't care.
  }

  mRETURN_SUCCESS();
}
