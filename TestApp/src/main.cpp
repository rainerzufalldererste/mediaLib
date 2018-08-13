#include "default.h"
#include "mMediaFileInputHandler.h"
#include "mMediaFileWriter.h"
#include "mThreadPool.h"
#include "SDL.h"
#include <time.h>

mVec2s resolution;
SDL_Window *pWindow = nullptr;
uint32_t *pPixels = nullptr;
const size_t subScale = 5;
mPtr<mImageBuffer> bgraImageBuffer = nullptr;
mPtr<mThreadPool> threadPool = nullptr;
mPtr<mMediaFileWriter> mediaFileWriter = nullptr;

mFUNCTION(ResizeWindow, const mVec2s &newSize);
mFUNCTION(OnVideoFramCallback, mPtr<mImageBuffer> &, const mVideoStreamType &videoStreamType);

int main(int, char **)
{
  mFUNCTION_SETUP();

  g_mResult_breakOnError = true;
  mDEFER_DESTRUCTION(&threadPool, mThreadPool_Destroy);
  mERROR_CHECK(mThreadPool_Create(&threadPool, nullptr));

  mPtr<mMediaFileInputHandler> mediaFileHandler;
  mDEFER_DESTRUCTION(&mediaFileHandler, mMediaFileInputHandler_Destroy);
  mERROR_CHECK(mMediaFileInputHandler_Create(&mediaFileHandler, nullptr, L"C:/Users/cstiller/Videos/Converted.mp4", mMediaFileInputHandler_CreateFlags::mMMFIH_CF_VideoEnabled));

  mERROR_CHECK(mMediaFileInputHandler_GetVideoStreamResolution(mediaFileHandler, &resolution));

  mMediaFileInformation fileInfo = { 0 };
  fileInfo.averageBitrate = 50 * 1024 * 1024;
  fileInfo.frameRateFraction = mVec2s(30, 1);
  fileInfo.frameSize = resolution;
  fileInfo.pixelAspectRatioFraction = mVec2s(1, 1);

  mDEFER_DESTRUCTION(&mediaFileWriter, mMediaFileWriter_Destroy);
  mERROR_CHECK(mMediaFileWriter_Create(&mediaFileWriter, nullptr, L"C:/Users/cstiller/Videos/Export.mp4", &fileInfo));

  mDEFER_DESTRUCTION(pWindow, SDL_DestroyWindow);
  pWindow = SDL_CreateWindow("VideoStream Renderer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, (int)resolution.x / subScale, (int)resolution.y / subScale, 0);
  mERROR_IF(pWindow == nullptr, mR_ArgumentNull);
  mERROR_CHECK(ResizeWindow(resolution / subScale));

  mDEFER_DESTRUCTION(&bgraImageBuffer, mImageBuffer_Destroy);
  mERROR_CHECK(mImageBuffer_Create(&bgraImageBuffer, nullptr, resolution));

  mPtr<mImageBuffer> image;
  mDEFER_DESTRUCTION(&image, mImageBuffer_Destroy);
  mERROR_CHECK(mImageBuffer_CreateFromFile(&image, nullptr, "C:/Users/cstiller/Pictures/avatar.png"));
  mERROR_CHECK(mImageBuffer_SaveAsJpeg(image, "C:/Users/cstiller/Pictures/avatar.jpg"));
  mERROR_CHECK(mImageBuffer_AllocateBuffer(bgraImageBuffer, image->currentSize));
  mERROR_CHECK(mPixelFormat_TransformBuffer(image, bgraImageBuffer));
  mERROR_CHECK(ResizeWindow(image->currentSize));
  mERROR_CHECK(mMemcpy(pPixels, (uint32_t *)bgraImageBuffer->pPixels, bgraImageBuffer->currentSize.x * bgraImageBuffer->currentSize.y));

  SDL_UpdateWindowSurface(pWindow);

  mERROR_CHECK(mMediaFileInputHandler_SetVideoCallback(mediaFileHandler, OnVideoFramCallback));
  mERROR_CHECK(mMediaFileInputHandler_Play(mediaFileHandler));

  mERROR_CHECK(mMediaFileWriter_Finalize(mediaFileWriter));

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

mFUNCTION(OnVideoFramCallback, mPtr<mImageBuffer> &buffer, const mVideoStreamType &videoStreamType)
{
  mFUNCTION_SETUP();

  const clock_t now = clock();

  if (buffer->currentSize != bgraImageBuffer->currentSize)
  {
    ResizeWindow(buffer->currentSize / subScale);
    mERROR_CHECK(mImageBuffer_AllocateBuffer(bgraImageBuffer, buffer->currentSize, bgraImageBuffer->pixelFormat));
  }

  mERROR_CHECK(mPixelFormat_TransformBuffer(buffer, bgraImageBuffer, threadPool));
  mPRINT("frame time: %" PRIi32 "\n", clock() - now);

  mERROR_CHECK(mMediaFileWriter_AppendVideoFrame(mediaFileWriter, bgraImageBuffer));

  size_t i = 0;

  for (size_t y = 0; y < videoStreamType.resolution.y; y += subScale)
    for (size_t x = 0; x < videoStreamType.resolution.x; x += subScale)
      pPixels[i++] = ((uint32_t *)(bgraImageBuffer->pPixels))[x + y * videoStreamType.resolution.x];

  SDL_UpdateWindowSurface(pWindow);

  SDL_Event sdl_event;
  while (SDL_PollEvent(&sdl_event))
    ; // We don't care.

  mRETURN_SUCCESS();
}
