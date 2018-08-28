#include "default.h"
#include "SDL.h"
#include "mThreadPool.h"
#include "mHardwareWindow.h"
#include "mMesh.h"
#include "mResourceManager.h"
#include "mVideoPlaybackEngine.h"
#include "mTimeStamp.h"
#include "SDL_opengl.h"

mFUNCTION(mainLoop);

int main(int, char **)
{
  mFUNCTION_SETUP();

  mResult result = mainLoop();

  if (mFAILED(result))
  {
    mString resultToString;
    mERROR_CHECK(mResult_ToString(result, &resultToString));
    mPRINT("Application Failed with Error %" PRIi32 " (%s) in File '%s' (Line %" PRIu64 ").\n", result, resultToString.c_str(), g_mResult_lastErrorFile, g_mResult_lastErrorLine);
    getchar();
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mainLoop)
{
  mFUNCTION_SETUP();

  g_mResult_breakOnError = true;

  SDL_Init(SDL_INIT_EVERYTHING);

  SDL_DisplayMode displayMode;
  SDL_GetCurrentDisplayMode(0, &displayMode);

  mVec2s resolution;
  resolution.x = displayMode.w;
  resolution.y = displayMode.h;

  mERROR_CHECK(mRenderParams_SetMultisampling(16));

  mPtr<mHardwareWindow> window = nullptr;
  mDEFER_DESTRUCTION(&window, mHardwareWindow_Destroy);
  mERROR_CHECK(mHardwareWindow_Create(&window, nullptr, "OpenGL Window", resolution, mHW_DM_Fullscreen));
  mERROR_CHECK(mRenderParams_InitializeToDefault());

  mERROR_CHECK(mRenderParams_SetDoubleBuffering(true));
  mERROR_CHECK(mRenderParams_SetVsync(false));
  mERROR_CHECK(mRenderParams_ShowCursor(false));

  GLboolean supportsStereo = false;
  glGetBooleanv(GL_STEREO, &supportsStereo);

  mPRINT(supportsStereo ? "Graphics Device supports stereo.\n" : "Graphics Device does not support stereo.\n");

  if (supportsStereo)
    mERROR_CHECK(mRenderParams_SetStereo3d(true));

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
  mERROR_CHECK(mTexture_Create(textureY.GetPointer(), pDataY, sizeY, mPF_Monochrome8, true, 0));
  mERROR_CHECK(mTexture_Create(textureU.GetPointer(), pDataU, sizeU, mPF_Monochrome8, true, 1));
  mERROR_CHECK(mTexture_Create(textureV.GetPointer(), pDataV, sizeV, mPF_Monochrome8, true, 2));

  // Create Meshes.
  mPtr<mQueue<mPtr<mMesh>>> meshes;
  mDEFER_DESTRUCTION(&meshes, mQueue_Destroy);
  mERROR_CHECK(mQueue_Create(&meshes, nullptr));

  const size_t quadCount = 1;

  // Create individual meshes.
  {
    mPtr<mMeshAttributeContainer> positionMeshAttribute;
    mDEFER_DESTRUCTION(&positionMeshAttribute, mMeshAttributeContainer_Destroy);
    mERROR_CHECK(mMeshAttributeContainer_Create<mVec2f>(&positionMeshAttribute, nullptr, "unused", { mVec2f(0), mVec2f(0), mVec2f(0), mVec2f(0) }));

    const char *vertexShader = mGLSL(
      in vec2 unused;
      out vec2 _texCoord0;
      uniform vec2 offsets[4];

    void main()
    {
      _texCoord0 = offsets[gl_VertexID];
      gl_Position = vec4(offsets[gl_VertexID] * 2 - 1 - unused, 0, 1);
    }
    );

    const char *fragmentShader = mGLSL(
      out vec4 outColour;
    in vec2 _texCoord0;
    uniform sampler2D textureY;
    uniform sampler2D textureU;
    uniform sampler2D textureV;
    uniform vec2 offsets[4];

    float cross(in vec2 a, in vec2 b) { return a.x*b.y - a.y*b.x; }

    vec2 invBilinear(in vec2 p, in vec2 a, in vec2 b, in vec2 c, in vec2 d)
    {
      vec2 e = b - a;
      vec2 f = d - a;
      vec2 g = a - b + c - d;
      vec2 h = p - a;

      float k2 = cross(g, f);
      float k1 = cross(e, f) + cross(h, g);
      float k0 = cross(h, e);

      float w = k1 * k1 - 4.0 * k0 * k2;

      if (w < 0.0)
        return vec2(-1.0);

      w = sqrt(w);

      float v1 = (-k1 - w) / (2.0 * k2);
      float u1 = (h.x - f.x * v1) / (e.x + g.x * v1);

      float v2 = (-k1 + w) / (2.0 * k2);
      float u2 = (h.x - f.x * v2) / (e.x + g.x * v2);

      float u = u1;
      float v = v1;

      if (v < 0.0 || v > 1.0 || u < 0.0 || u > 1.0)
      {
        u = u2;
        v = v2;
      }

      if (v < 0.0 || v > 1.0 || u < 0.0 || u > 1.0)
      {
        u = -1.0;
        v = -1.0;
      }

      return vec2(u, v);
    }

    void main()
    {
      vec2 pos = invBilinear(_texCoord0, offsets[1], offsets[3], offsets[2], offsets[0]);

      float y = texture(textureY, pos).x;
      float u = texture(textureU, pos).x;
      float v = texture(textureV, pos).x;

      vec3 col = vec3(
        clamp(y + 1.370705 * (v - 0.5), 0, 1),
        clamp(y - 0.698001 * (v - 0.5) - 0.337633 * (u - 0.5), 0, 1),
        clamp(y + 1.732446 * (u - 0.5), 0, 1));

      outColour = vec4(col, 1);
    }
    );

    mPtr<mShader> shader;
    mDEFER_DESTRUCTION(&shader, mSharedPointer_Destroy);
    mERROR_CHECK(mSharedPointer_Allocate(&shader, nullptr, (std::function<void(mShader *)>)[](mShader *pData) { mShader_Destroy(pData); }, 1));
    mERROR_CHECK(mShader_Create(shader.GetPointer(), vertexShader, fragmentShader, "outColour"));

    mGL_ERROR_CHECK();
    mERROR_CHECK(mShader_SetUniform(shader, "textureY", textureY));
    mERROR_CHECK(mShader_SetUniform(shader, "textureU", textureU));
    mERROR_CHECK(mShader_SetUniform(shader, "textureV", textureV));

    mGL_ERROR_CHECK();
    mPtr<mQueue<mPtr<mTexture>>> textures;
    mDEFER_DESTRUCTION(&textures, mQueue_Destroy);
    mERROR_CHECK(mQueue_Create(&textures, nullptr));

    mERROR_CHECK(mQueue_PushBack(textures, textureY));
    mERROR_CHECK(mQueue_PushBack(textures, textureU));
    mERROR_CHECK(mQueue_PushBack(textures, textureV));

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

    if(leftEye)
      mERROR_CHECK(mRenderParams_ClearTargetDepthAndColour());
    else
      mERROR_CHECK(mRenderParams_ClearTargetDepthAndColour(mVector(1, 0, 0)));

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
      if (_event.type == SDL_QUIT || (_event.type == SDL_KEYDOWN && _event.key.keysym.sym == SDLK_ESCAPE))
        goto end;
  }

end:;

  mRETURN_SUCCESS();
}
