#include "default.h"
#include "mMediaFileInputHandler.h"
#include "mMediaFileWriter.h"
#include "mThreadPool.h"
#include "SDL.h"
#include <time.h>

#include "GL/glew.h"
#include "GL/wglew.h"

#pragma warning(push)
#pragma warning(disable: 4505)
#include "GL/freeglut.h"
#pragma warning(pop)

mVec2s resolution;
SDL_Window *pWindow = nullptr;
SDL_Surface *pSurface = nullptr;
const size_t subScale = 5;
mPtr<mImageBuffer> bgraImageBuffer = nullptr;
mPtr<mThreadPool> threadPool = nullptr;
bool is3dEnabled = false;

mFUNCTION(OnVideoFramCallback, mPtr<mImageBuffer> &, const mVideoStreamType &videoStreamType);
mFUNCTION(RenderFrame);

int main(int, char **)
{
  mFUNCTION_SETUP();

  g_mResult_breakOnError = true;
  mDEFER_DESTRUCTION(&threadPool, mThreadPool_Destroy);
  mERROR_CHECK(mThreadPool_Create(&threadPool, nullptr));

  mPtr<mMediaFileInputHandler> mediaFileHandler;
  mDEFER_DESTRUCTION(&mediaFileHandler, mMediaFileInputHandler_Destroy);
  mERROR_CHECK(mMediaFileInputHandler_Create(&mediaFileHandler, nullptr, L"C:/Users/cstiller/Videos/Converted.mp4", mMediaFileInputHandler_CreateFlags::mMMFIH_CF_VideoEnabled));

  GLenum glError = GL_NO_ERROR;
  mUnused(glError);

  SDL_Init(SDL_INIT_EVERYTHING);
  mERROR_IF(glewInit() != GL_TRUE, mR_InternalError);

  SDL_DisplayMode displayMode;
  SDL_GetCurrentDisplayMode(0, &displayMode);

  resolution.x = displayMode.w;
  resolution.y = displayMode.h;

  mDEFER_DESTRUCTION(pWindow, SDL_DestroyWindow);
  pWindow = SDL_CreateWindow("VideoStream Renderer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, (int)resolution.x, (int)resolution.y, SDL_WINDOW_FULLSCREEN | SDL_WINDOW_OPENGL);
  mERROR_IF(pWindow == nullptr, mR_ArgumentNull);

  if (SDL_GL_SetAttribute(SDL_GL_STEREO, 1) == 0)
  {
    is3dEnabled = true;
    mPRINT("3d enabled.");
  }

  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
  SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);

  pSurface = SDL_GetWindowSurface(pWindow);
  mERROR_IF(pSurface == nullptr, mR_ArgumentNull);

  mDEFER_DESTRUCTION(&bgraImageBuffer, mImageBuffer_Destroy);
  mERROR_CHECK(mImageBuffer_Create(&bgraImageBuffer, nullptr, resolution));

  mERROR_CHECK(mMediaFileInputHandler_SetVideoCallback(mediaFileHandler, OnVideoFramCallback));
  mERROR_CHECK(mMediaFileInputHandler_Play(mediaFileHandler));

  mRETURN_SUCCESS();
}

mFUNCTION(OnVideoFramCallback, mPtr<mImageBuffer> &buffer, const mVideoStreamType &videoStreamType)
{
  mFUNCTION_SETUP();

  mUnused(videoStreamType);

  if (buffer->currentSize != bgraImageBuffer->currentSize)
    mERROR_CHECK(mImageBuffer_AllocateBuffer(bgraImageBuffer, buffer->currentSize, bgraImageBuffer->pixelFormat));

  mERROR_CHECK(mPixelFormat_TransformBuffer(buffer, bgraImageBuffer, threadPool));

  SDL_Event sdl_event;
  while (SDL_PollEvent(&sdl_event))
    ; // We don't care.

  mERROR_CHECK(RenderFrame());

  mRETURN_SUCCESS();
}

size_t frame = 0;

mFUNCTION(RenderFrame)
{
  mFUNCTION_SETUP();

  glClearColor((frame & 0xFF) / 255.0f, 0.0f, 0.0f, 1.0f);
  glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

  SDL_UpdateWindowSurface(pWindow);

  mRETURN_SUCCESS();
}
