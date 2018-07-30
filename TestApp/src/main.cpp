#include "default.h"
#include "mVideoFileInputHandler.h"
#include "SDL.h"

size_t sizeX = 0, sizeY = 0;
SDL_Window *pWindow = nullptr;
uint32_t *pPixels = nullptr;

mFUNCTION(ProcessBufferCallback, IN uint8_t *pBuffer);

int main(int, char **)
{
  mFUNCTION_SETUP();

  int k = 0;
  mPtr<int> kptr;
  mDEFER_DESTRUCTION(&kptr, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Create(&kptr, &k, mAT_ForeignRessource));

  uint32_t *pPtr = nullptr;
  mDEFER_DESTRUCTION(pPtr, mFree);
  mERROR_CHECK(mAlloc(&pPtr, 1));

  g_mResult_breakOnError = true;
  mPtr<mVideoFileInputHandler> videoInput;
  mDEFER_DESTRUCTION(&videoInput, mVideoFileInputHandler_Destroy);
  mERROR_CHECK(mVideoFileInputHandler_Create(&videoInput, L"C:\\Users\\cstiller\\Videos\\Original.MP4"));
  
  mERROR_CHECK(mVideoFileInputHandler_GetSize(videoInput, &sizeX, &sizeY));

  SDL_Init(SDL_INIT_EVERYTHING);

  pWindow = SDL_CreateWindow("VideoFrames", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, (int)sizeX, (int)sizeY, 0);
  mDEFER_DESTRUCTION(pWindow, SDL_DestroyWindow);
  SDL_Surface *pSurface = SDL_GetWindowSurface(pWindow);
  pPixels = (uint32_t *)pSurface->pixels;

  mERROR_CHECK(mVideoFileInputHandler_SetCallback(videoInput, ProcessBufferCallback));
  mERROR_CHECK(mVideoFileInputHandler_Play(videoInput));

  mRETURN_SUCCESS();
}

mFUNCTION(ProcessBufferCallback, IN uint8_t *pBuffer)
{
  mFUNCTION_SETUP();

  if (pBuffer)
  {
    mERROR_CHECK(mMemcpy(pPixels, (uint32_t *)pBuffer, sizeX * sizeY));

    SDL_UpdateWindowSurface(pWindow);

    SDL_Event sdl_event;
    while (SDL_PollEvent(&sdl_event))
      ; // We don't care.
  }

  mRETURN_SUCCESS();
}