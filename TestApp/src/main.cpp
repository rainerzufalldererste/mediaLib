#include "default.h"
#include "mMediaFileInputHandler.h"
#include "mMediaFileWriter.h"
#include "mThreadPool.h"
#include "SDL.h"
#include <time.h>
#include "GL\glew.h"
#include "mHardwareWindow.h"
#include "mShader.h"
#include "mTexture.h"
#include "mMesh.h"
#include "mSpriteBatch.h"
#include "mResourceManager.h"
#include "mVideoPlaybackEngine.h"

const std::wstring videoFilename = L"N:/Data/video/PublicHologram.mp4";

int main(int, char **)
{
  mFUNCTION_SETUP();

  g_mResult_breakOnError = true;

  SDL_Init(SDL_INIT_EVERYTHING);

  SDL_DisplayMode displayMode;
  SDL_GetCurrentDisplayMode(0, &displayMode);

  mVec2s resolution;
  resolution.x = displayMode.w / 2;
  resolution.y = displayMode.h / 2;

  mPtr<mHardwareWindow> window = nullptr;
  mDEFER_DESTRUCTION(&window, mHardwareWindow_Destroy);
  mERROR_CHECK(mHardwareWindow_Create(&window, nullptr, "OpenGL Window", resolution));// , mHW_DM_FullscreenDesktop));
  mERROR_CHECK(mRenderParams_InitializeToDefault());

  mERROR_CHECK(mRenderParams_SetDoubleBuffering(true));
  mERROR_CHECK(mRenderParams_SetMultisampling(4));
  mERROR_CHECK(mRenderParams_SetVsync(false));

  mPtr<mThreadPool> threadPool;
  mPtr<mVideoPlaybackEngine> videoPlaybackEngine;
  mDEFER_DESTRUCTION(&videoPlaybackEngine, mVideoPlaybackEngine_Destroy);
  mERROR_CHECK(mVideoPlaybackEngine_Create(&videoPlaybackEngine, nullptr, videoFilename, threadPool, 0, mPF_YUV420));

  mPtr<mTexture> textureY;
  mDEFER_DESTRUCTION(&textureY, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Allocate(&textureY, nullptr, (std::function<void(mTexture *)>)[](mTexture *pTexture) {mTexture_Destroy(pTexture);}, 1));

  mPtr<mTexture> textureU;
  mDEFER_DESTRUCTION(&textureU, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Allocate(&textureU, nullptr, (std::function<void(mTexture *)>)[](mTexture *pTexture) {mTexture_Destroy(pTexture);}, 1));

  mPtr<mTexture> textureV;
  mDEFER_DESTRUCTION(&textureV, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Allocate(&textureV, nullptr, (std::function<void(mTexture *)>)[](mTexture *pTexture) {mTexture_Destroy(pTexture);}, 1));

  mPtr<mImageBuffer> currentFrame;
  mERROR_CHECK(mVideoPlaybackEngine_GetCurrentFrame(videoPlaybackEngine, &currentFrame));

  // Create Textures.
  size_t offsetY, offsetU, offsetV;
  mVec2s sizeY, sizeU, sizeV;
  uint8_t *pDataY, *pDataU, *pDataV;
  mERROR_CHECK(mPixelFormat_GetSubBufferOffset(currentFrame->pixelFormat, 0, currentFrame->currentSize, &offsetY));
  mERROR_CHECK(mPixelFormat_GetSubBufferOffset(currentFrame->pixelFormat, 1, currentFrame->currentSize, &offsetU));
  mERROR_CHECK(mPixelFormat_GetSubBufferOffset(currentFrame->pixelFormat, 2, currentFrame->currentSize, &offsetV));
  mERROR_CHECK(mPixelFormat_GetSubBufferSize(currentFrame->pixelFormat, 0, currentFrame->currentSize, &sizeY));
  mERROR_CHECK(mPixelFormat_GetSubBufferSize(currentFrame->pixelFormat, 1, currentFrame->currentSize, &sizeU));
  mERROR_CHECK(mPixelFormat_GetSubBufferSize(currentFrame->pixelFormat, 2, currentFrame->currentSize, &sizeV));

  pDataY = currentFrame->pPixels + offsetY;
  pDataU = currentFrame->pPixels + offsetU;
  pDataV = currentFrame->pPixels + offsetV;
  mERROR_CHECK(mTexture_Create(textureY.GetPointer(), pDataY, sizeY, mPF_Monochrome8));
  mERROR_CHECK(mTexture_Create(textureU.GetPointer(), pDataU, sizeU, mPF_Monochrome8));
  mERROR_CHECK(mTexture_Create(textureV.GetPointer(), pDataV, sizeV, mPF_Monochrome8));

  // Create Meshes.
  mPtr<mQueue<mPtr<mMesh>>> meshes;
  mDEFER_DESTRUCTION(&meshes, mQueue_Destroy);
  mERROR_CHECK(mQueue_Create(&meshes, nullptr));

  // Create individual meshes.
  {
    const size_t meshesCount = 1;

    mPtr<mQueue<mMeshFactory_AttributeInformation>> info;
    mDEFER_DESTRUCTION(&info, mQueue_Destroy);
    mERROR_CHECK(mQueue_Create(&info, nullptr));

    mERROR_CHECK(mQueue_PushBack(info, mMeshFactory_AttributeInformation(sizeof(mVec2f), 0, mMF_AIT_Attribute, GL_FLOAT, sizeof(float_t))));
    mERROR_CHECK(mQueue_PushBack(info, mMeshFactory_AttributeInformation(sizeof(mVec2f), 0, mMF_AIT_Attribute, GL_FLOAT, sizeof(float_t))));

    const char *vertexShader = mGLSL(
      in vec2 position0;
      in vec2 texCoord0;
      out vec2 _texCoord0;

      void main()
      {
        _texCoord0 = texCoord0;
        gl_Position = vec4(position0, 0, 1);
      }
    );

    const char *fragmentShader = mGLSL(
      out vec4 outColour;
      in vec2 _texCoord0;
      uniform sampler2D textureY;
      uniform sampler2D textureU;
      uniform sampler2D textureV;

      void main()
      {
        float y = texture(textureY, _texCoord0).x;
        float u = texture(textureU, _texCoord0).x;
        float v = texture(textureV, _texCoord0).x;

        outColour = vec4(y, u, v, 1);
      }
    );

    mPtr<mShader> shader;
    mDEFER_DESTRUCTION(&shader, mSharedPointer_Destroy);
    mERROR_CHECK(mSharedPointer_Allocate(&shader, nullptr, (std::function<void(mShader *)>)[](mShader *pData) {mShader_Destroy(pData);}, 1));
    mERROR_CHECK(mShader_Create(shader.GetPointer(), vertexShader, fragmentShader, "outColour"));

    mPtr<mQueue<mPtr<mTexture>>> textures;
    mDEFER_DESTRUCTION(&textures, mQueue_Destroy);
    mERROR_CHECK(mQueue_Create(&textures, nullptr));

    mERROR_CHECK(mQueue_PushBack(textures, textureY));
    mERROR_CHECK(mQueue_PushBack(textures, textureU));
    mERROR_CHECK(mQueue_PushBack(textures, textureV));

    for (size_t i = 0; i < meshesCount; ++i)
    {
      mPtr<mBinaryChunk> binaryChunk;
      mDEFER_DESTRUCTION(&binaryChunk, mBinaryChunk_Destroy);
      mERROR_CHECK(mBinaryChunk_Create(&binaryChunk, nullptr));

      mERROR_CHECK(mBinaryChunk_WriteData(binaryChunk, mVec2f(-1, -1)));  // position.
      mERROR_CHECK(mBinaryChunk_WriteData(binaryChunk, mVec2f(0 ,  1)));  // texCoord.
      mERROR_CHECK(mBinaryChunk_WriteData(binaryChunk, mVec2f(1 , -1)));  // position.
      mERROR_CHECK(mBinaryChunk_WriteData(binaryChunk, mVec2f(0 ,  0)));  // texCoord.
      mERROR_CHECK(mBinaryChunk_WriteData(binaryChunk, mVec2f(-1,  1)));  // position.
      mERROR_CHECK(mBinaryChunk_WriteData(binaryChunk, mVec2f(1 ,  1)));  // texCoord.
      mERROR_CHECK(mBinaryChunk_WriteData(binaryChunk, mVec2f(1 ,  1)));  // position.
      mERROR_CHECK(mBinaryChunk_WriteData(binaryChunk, mVec2f(1 ,  0)));  // texCoord.

      mPtr<mMesh> mesh;
      mDEFER_DESTRUCTION(&mesh, mMesh_Destroy);
      mERROR_CHECK(mMesh_Create(&mesh, nullptr, info, shader, binaryChunk, textures, mRP_RM_TriangleStrip));

      mERROR_CHECK(mQueue_PushBack(meshes, mesh));
    }
  }

  while (true)
  {
    mERROR_CHECK(mRenderParams_ClearTargetDepthAndColour());

    mTimeStamp before;
    mERROR_CHECK(mTimeStamp_Now(&before));

    bool isNewFrame = true;
    mResult result = mVideoPlaybackEngine_GetCurrentFrame(videoPlaybackEngine, &currentFrame, &isNewFrame);

    if (mFAILED(result))
    {
      if (result == mR_EndOfStream)
        mERROR_CHECK(mVideoPlaybackEngine_Create(&videoPlaybackEngine, nullptr, videoFilename, threadPool, 0, mPF_YUV420));
      else
        mERROR_CHECK(result);

      continue;
    }

    if (!isNewFrame)
      continue;

    // Change textures.
    pDataY = currentFrame->pPixels + offsetY;
    pDataU = currentFrame->pPixels + offsetU;
    pDataV = currentFrame->pPixels + offsetV;
    mERROR_CHECK(mTexture_SetTo(*textureY.GetPointer(), pDataY, sizeY, mPF_Monochrome8));
    mERROR_CHECK(mTexture_SetTo(*textureU.GetPointer(), pDataU, sizeU, mPF_Monochrome8));
    mERROR_CHECK(mTexture_SetTo(*textureV.GetPointer(), pDataV, sizeV, mPF_Monochrome8));

    // Render Scene.
    size_t meshCount = 0;
    mERROR_CHECK(mQueue_GetCount(meshes, &meshCount));

    for (size_t i = 0; i < meshCount; ++i)
    {
      mPtr<mMesh> *pMesh;
      mERROR_CHECK(mQueue_PointerAt(meshes, i, &pMesh));

      mERROR_CHECK(mMesh_Render(*pMesh));
    }

    mGL_DEBUG_ERROR_CHECK();

    // Swap.
    mERROR_CHECK(mHardwareWindow_Swap(window));

    mTimeStamp after;
    mERROR_CHECK(mTimeStamp_Now(&after));

    mPRINT("\rframes per second: %f, frame time: %f ms   ", (1.0 / (after - before).timePoint), (after - before).timePoint * 1000.0);

    SDL_Event _event;
    while (SDL_PollEvent(&_event))
      if (_event.type == SDL_QUIT || (_event.type == SDL_KEYDOWN && _event.key.keysym.sym == SDLK_ESCAPE))
        goto end;
  }

end:;

  mRETURN_SUCCESS();
}
