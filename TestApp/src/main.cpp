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
#include "mUI.h"
#include "mFile.h"

mFUNCTION(MainLoop);
mFUNCTION(RenderShaderWindow);

bool shaderEditWindowShown = false;
mPtr<mShader> videoShader;
static char vertexShader[2048];
static char fragmentShader[2048];

//////////////////////////////////////////////////////////////////////////

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
  mERROR_CHECK(mHardwareWindow_Create(&window, nullptr, "OpenGL Window", resolution, (mHardwareWindow_DisplayMode)(mHW_DM_Windowed | mHW_DM_Resizeable), true));

  mERROR_CHECK(mUI_Initilialize(window));

  bool running = true;
  mERROR_CHECK(mHardwareWindow_AddOnCloseEvent(window, [&]() { running = false; return mR_Success; }));

  mERROR_CHECK(mHardwareWindow_AddOnAnyEvent(window,
    [&](IN SDL_Event *pEvent)
  {
    mFUNCTION_SETUP();

    if (pEvent->type == SDL_KEYDOWN)
    {
      if (pEvent->key.keysym.sym == SDLK_ESCAPE)
      {
        running = false;
      }
      else if (pEvent->key.keysym.sym == SDLK_BACKQUOTE)
      {
        shaderEditWindowShown = true;

        mERROR_IF(videoShader == nullptr, mR_ArgumentNull);

        std::string shaderText;
        mERROR_CHECK(mFile_ReadAllText(videoShader->vertexShader, nullptr, &shaderText));
        mERROR_CHECK(mAllocator_Copy(nullptr, vertexShader, shaderText.c_str(), mMin(shaderText.size(), mARRAYSIZE(vertexShader))));

        mERROR_CHECK(mFile_ReadAllText(videoShader->fragmentShader, nullptr, &shaderText));
        mERROR_CHECK(mAllocator_Copy(nullptr, fragmentShader, shaderText.c_str(), mMin(shaderText.size(), mARRAYSIZE(fragmentShader))));
      }
    }

    mRETURN_SUCCESS();
  }));

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

  const std::wstring videoFilename = L"N:/Data/video/PublicHologram.mp4";

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
  mDEFER_DESTRUCTION(&videoShader, mSharedPointer_Destroy);

  mPtr<mMeshFactory<mMesh2dPosition, mMeshTexcoord>> meshFactory;
  mDEFER_DESTRUCTION(&meshFactory, mMeshFactory_Destroy);
  mERROR_CHECK(mMeshFactory_Create(&meshFactory, nullptr));

  mERROR_CHECK(mMeshFactory_AppendData(meshFactory, mMesh2dPosition(-0.5, -0.5), mMeshTexcoord(0, 0)));
  mERROR_CHECK(mMeshFactory_AppendData(meshFactory, mMesh2dPosition(0.5, -0.5), mMeshTexcoord(1, 0)));
  mERROR_CHECK(mMeshFactory_AppendData(meshFactory, mMesh2dPosition(0, 0.5), mMeshTexcoord(1, 1)));

  mPtr<mMesh> meshTest;
  mDEFER_DESTRUCTION(&meshTest, mMesh_Destroy);
  mERROR_CHECK(mMeshFactory_CreateMesh(meshFactory, &meshTest, nullptr, textureY));

  const size_t quadCount = 1;

  // Create individual meshes.
  {
    mPtr<mMeshAttributeContainer> positionMeshAttribute;
    mDEFER_DESTRUCTION(&positionMeshAttribute, mMeshAttributeContainer_Destroy);
    mERROR_CHECK(mMeshAttributeContainer_Create<mVec2f>(&positionMeshAttribute, nullptr, "unused", { mVec2f(0), mVec2f(0), mVec2f(0), mVec2f(0) }));

    mERROR_CHECK(mSharedPointer_Allocate(&videoShader, nullptr, (std::function<void(mShader *)>)[](mShader *pData) { mShader_Destroy(pData); }, 1));
    mERROR_CHECK(mShader_CreateFromFile(videoShader.GetPointer(), L"../shaders/yuvQuad"));

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
      mERROR_CHECK(mMesh_Create(&mesh, nullptr, { positionMeshAttribute }, videoShader, textures, mRP_VRM_TriangleStrip));

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

  while (running)
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

mERROR_CHECK(mUI_StartFrame(window));
mERROR_CHECK(RenderShaderWindow());
mERROR_CHECK(mUI_Bake(window));

mERROR_CHECK(mRenderParams_ClearTargetDepthAndColour());

mERROR_CHECK(mTexture_Bind(framebuffer));
mERROR_CHECK(mShader_SetUniform(screenQuad->shader, "texture0", framebuffer));
mERROR_CHECK(mScreenQuad_Render(screenQuad));

mERROR_CHECK(mMesh_Render(meshTest));

mERROR_CHECK(mUI_Render());

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
  }

  mRETURN_SUCCESS();
}

mFUNCTION(RenderShaderWindow)
{
  mFUNCTION_SETUP();

  if (!shaderEditWindowShown)
    mRETURN_SUCCESS();

  if (ImGui::Begin("Shader Editor", &shaderEditWindowShown, ImVec2(0, 0), 1.0f, ImGuiWindowFlags_NoCollapse))
  {
    static bool vertSelected = true;
    static bool fragSelected = false;

    ImGui::Columns(2, nullptr, false);

    if (ImGui::Selectable("Vertex Shader", &vertSelected, 0, ImVec2(80, 20)))
    {
      vertSelected = true;
      fragSelected = false;
    }

    ImGui::SameLine(0);

    if (ImGui::Selectable("Fragment Shader", &fragSelected, 0, ImVec2(97, 20)))
    {
      fragSelected = true;
      vertSelected = false;
    }

    ImGui::Columns(1);
    ImGui::Separator();

    mERROR_CHECK(mUI_PushMonospacedFont());
    ImGui::InputTextMultiline("", vertSelected ? vertexShader : fragmentShader, mARRAYSIZE(vertexShader), ImVec2(-1, -30));
    mERROR_CHECK(mUI_PopMonospacedFont());

    if (ImGui::Button("Reload Shaders"))
    {
      mERROR_CHECK(mFile_WriteAllText(videoShader->vertexShader, vertexShader));
      mERROR_CHECK(mFile_WriteAllText(videoShader->fragmentShader, fragmentShader));

      mResult result = mShader_ReloadFromFile(videoShader);

      if (mFAILED(result))
        ImGui::OpenPopup("Shader Reloading Error");
    }

    if (ImGui::BeginPopupModal("Shader Reloading Error", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
    {
      ImGui::Text("There were compilation errors. Please check the command line for details.");

      if (ImGui::Button("Ok"))
        ImGui::CloseCurrentPopup();

      ImGui::EndPopup();
    }
  }

  ImGui::End();
  
  mRETURN_SUCCESS();
}
