#include "default.h"
#include "SDL.h"
#include "mThreadPool.h"
#include "mHardwareWindow.h"
#include "mMesh.h"
#include "mResourceManager.h"
#include "mVideoPlaybackEngine.h"
#include "mTimeStamp.h"
#include "mFramebuffer.h"
#include "mScreenQuad.h"

mFUNCTION(MainLoop);

int main(int, char **)
{
  mFUNCTION_SETUP();

  mResult result = MainLoop();

  if (mFAILED(result))
  {
    mString resultToString;
    mERROR_CHECK(mResult_ToString(result, &resultToString));
    mPRINT("Application Failed with Error %" PRIi32 " (%s) in File '%s' (Line %" PRIu64 ").\n", result, resultToString.c_str(), g_mResult_lastErrorFile, g_mResult_lastErrorLine);
    getchar();
  }

  mRETURN_SUCCESS();
}

mFUNCTION(MainLoop)
{
  mFUNCTION_SETUP();

  g_mResult_breakOnError = true;

  SDL_Init(SDL_INIT_EVERYTHING);

  SDL_DisplayMode displayMode;
  SDL_GetCurrentDisplayMode(0, &displayMode);

  mVec2s resolution;
  resolution.x = displayMode.w / 2;
  resolution.y = displayMode.h / 2;

  mERROR_CHECK(mRenderParams_SetMultisampling(16));

  mPtr<mHardwareWindow> window = nullptr;
  mDEFER_DESTRUCTION(&window, mHardwareWindow_Destroy);
  mERROR_CHECK(mHardwareWindow_Create(&window, nullptr, "OpenGL Window", resolution));
  mERROR_CHECK(mRenderParams_InitializeToDefault());

  mERROR_CHECK(mRenderParams_SetDoubleBuffering(true));
  mERROR_CHECK(mRenderParams_SetVsync(false));
  mERROR_CHECK(mRenderParams_ShowCursor(false));

  bool supportsStereo = false;
  mERROR_CHECK(mRenderParams_SupportsStereo3d(&supportsStereo));

  mPRINT(supportsStereo ? "Graphics Device supports stereo.\n" : "Graphics Device does not support stereo.\n");

  if (supportsStereo)
  {
    mERROR_CHECK(mRenderParams_SetStereo3d(true));
    mERROR_CHECK(mRenderParams_SetVsync(true));
  }

  const std::wstring videoFilename = L"C:/Users/cstiller/Videos/Converted.mp4";

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

  mPtr<mFramebuffer> framebuffer;
  mDEFER_DESTRUCTION(&framebuffer, mFramebuffer_Destroy);
  mERROR_CHECK(mFramebuffer_Create(&framebuffer, nullptr, mRenderParams_CurrentRenderResolution));

  mPtr<mScreenQuad> screenQuad;
  mDEFER_DESTRUCTION(&screenQuad, mScreenQuad_Destroy);
  mERROR_CHECK(mScreenQuad_CreateFrom(&screenQuad, nullptr, L"../shaders/screenEffect.frag"));

  mPtr<mImageBuffer> currentFrame;
  mDEFER_DESTRUCTION(&currentFrame, mImageBuffer_Destroy);
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
  mERROR_CHECK(mTexture_Create(textureY.GetPointer(), pDataY, sizeY, mPF_Monochrome8, true, 0));
  mERROR_CHECK(mTexture_Create(textureU.GetPointer(), pDataU, sizeU, mPF_Monochrome8, true, 1));
  mERROR_CHECK(mTexture_Create(textureV.GetPointer(), pDataV, sizeV, mPF_Monochrome8, true, 2));

  // Create Meshes.
  mPtr<mQueue<mPtr<mMesh>>> meshes;
  mDEFER_DESTRUCTION(&meshes, mQueue_Destroy);
  mERROR_CHECK(mQueue_Create(&meshes, nullptr));

  mPtr<mShader> shader;
  mDEFER_DESTRUCTION(&shader, mSharedPointer_Destroy);

  const size_t quadCount = 1;

  // Create individual meshes.
  {
    mPtr<mMeshAttributeContainer> positionMeshAttribute;
    mDEFER_DESTRUCTION(&positionMeshAttribute, mMeshAttributeContainer_Destroy);
    mERROR_CHECK(mMeshAttributeContainer_Create<mVec2f>(&positionMeshAttribute, nullptr, "unused", { mVec2f(0), mVec2f(0), mVec2f(0), mVec2f(0) }));

    mERROR_CHECK(mSharedPointer_Allocate(&shader, nullptr, (std::function<void(mShader *)>)[](mShader *pData) { mShader_Destroy(pData); }, 1));
    mERROR_CHECK(mShader_CreateFromFile(shader.GetPointer(), L"../shaders/yuvQuad"));

    mGL_ERROR_CHECK();

    mGL_ERROR_CHECK();
    mPtr<mQueue<mKeyValuePair<mString, mPtr<mTexture>>>> textures;
    mDEFER_DESTRUCTION(&textures, mQueue_Destroy);
    mERROR_CHECK(mQueue_Create(&textures, nullptr));

    mERROR_CHECK(mQueue_PushBack(textures, mKeyValuePair<mString, mPtr<mTexture>>("textureY", textureY)));
    mERROR_CHECK(mQueue_PushBack(textures, mKeyValuePair<mString, mPtr<mTexture>>("textureU", textureU)));
    mERROR_CHECK(mQueue_PushBack(textures, mKeyValuePair<mString, mPtr<mTexture>>("textureV", textureV)));

    for (size_t i = 0; i < quadCount; ++i)
    {
      mPtr<mMesh> mesh;
      mDEFER_DESTRUCTION(&mesh, mMesh_Destroy);
      mERROR_CHECK(mMesh_Create(&mesh, nullptr, { positionMeshAttribute }, shader, textures, mRP_VRM_TriangleStrip));

      mERROR_CHECK(mQueue_PushBack(meshes, mesh));
    }
  }

  // Generate Point Offsets:
  mVec2f positions[quadCount * 4] = { { 0.15, 0.15 },{ 0, 1 },{ 0.6, 0.025 },{ 1, 1 } };

  size_t frames = 0;
  size_t decodedFrames = 0;
  mTimeStamp before;
  mERROR_CHECK(mTimeStamp_Now(&before));
  bool isFirstFrame = true;

  while (true)
  {
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

    if (isNewFrame)
      ++decodedFrames;

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

    bool leftEye = true;

  stereoEntryPoint:

    if (supportsStereo)
      mERROR_CHECK(mRenderParams_SetStereo3dBuffer(leftEye ? mRP_SRB_LeftEye : mRP_SRB_RightEye));

    mERROR_CHECK(mFramebuffer_Bind(framebuffer));

    mERROR_CHECK(mRenderParams_ClearTargetDepthAndColour());

    for (size_t i = 0; i < meshCount; ++i)
    {
      mPtr<mMesh> *pMesh;
      mERROR_CHECK(mQueue_PointerAt(meshes, i, &pMesh));

      mVec2f *pPositions = &positions[i * 4];

      for (size_t offsetIndex = 0; offsetIndex < 4; ++offsetIndex)
        pPositions[offsetIndex] += (mVec2f(rand() / (float_t)RAND_MAX, rand() / (float_t)RAND_MAX) - mVec2f(0.5f)) / 1000.0f;

      mERROR_CHECK(mShader_SetUniform((*pMesh)->shader, "offsets", pPositions, 4));
      mERROR_CHECK(mMesh_Render(*pMesh));
    }

    mERROR_CHECK(mFramebuffer_Unbind());
    mERROR_CHECK(mHardwareWindow_SetAsActiveRenderTarget(window));

    mERROR_CHECK(mRenderParams_ClearTargetDepthAndColour());

    mERROR_CHECK(mTexture_Bind(framebuffer));
    mERROR_CHECK(mShader_SetUniform(screenQuad->shader, "texture0", framebuffer));
    mERROR_CHECK(mScreenQuad_Render(screenQuad));

    if (isFirstFrame)
    {
      isFirstFrame = false;
      mPtr<mImageBuffer> imageBuffer;
      mDEFER_DESTRUCTION(&imageBuffer, mImageBuffer_Destroy);
      mERROR_CHECK(mFramebuffer_Download(framebuffer, &imageBuffer, nullptr));
      mERROR_CHECK(mImageBuffer_SaveAsJpeg(imageBuffer, "frame.jpg"));
    }

    if (supportsStereo && leftEye)
    {
      leftEye = false;
      goto stereoEntryPoint;
    }

    mGL_DEBUG_ERROR_CHECK();

    // Swap.
    mERROR_CHECK(mHardwareWindow_Swap(window));

    mTimeStamp after;
    mERROR_CHECK(mTimeStamp_Now(&after));

    ++frames;

    if ((after - before).timePoint > 1)
    {
      mPRINT("\rframes per second: %f (decoded %f video frames per second), frame time: %f ms   ", (float_t)frames / (after - before).timePoint, (float_t)decodedFrames / (after - before).timePoint, (after - before).timePoint * 1000.0 / (float_t)frames);

      frames = 0;
      decodedFrames = 0;
      mERROR_CHECK(mTimeStamp_Now(&before));
    }

    SDL_Event _event;
    while (SDL_PollEvent(&_event))
    {
      if (_event.type == SDL_QUIT || (_event.type == SDL_KEYDOWN && _event.key.keysym.sym == SDLK_ESCAPE))
        goto end;
      else if (_event.type == SDL_KEYDOWN && _event.key.keysym.sym == SDLK_BACKQUOTE)
        mERROR_CHECK(mShader_ReloadFromFile(shader));
    }
  }

end:;

  mRETURN_SUCCESS();
}
