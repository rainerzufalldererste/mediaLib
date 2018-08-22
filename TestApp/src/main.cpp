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

mPtr<mHardwareWindow> window = nullptr;
mPtr<mImageBuffer> image;
mPtr<mThreadPool> threadPool = nullptr;

int main(int, char **)
{
  mFUNCTION_SETUP();

  g_mResult_breakOnError = true;
  mDEFER_DESTRUCTION(&threadPool, mThreadPool_Destroy);
  mERROR_CHECK(mThreadPool_Create(&threadPool, nullptr));

  mDEFER_DESTRUCTION(&image, mImageBuffer_Destroy);
  mERROR_CHECK(mImageBuffer_CreateFromFile(&image, nullptr, "C:/data/avatar.jpg"));

  SDL_Init(SDL_INIT_EVERYTHING);

  SDL_DisplayMode displayMode;
  SDL_GetCurrentDisplayMode(0, &displayMode);

  mVec2s resolution;
  resolution.x = displayMode.w / 2;
  resolution.y = displayMode.h / 2;

  mDEFER_DESTRUCTION(&window, mHardwareWindow_Destroy);
  mERROR_CHECK(mHardwareWindow_Create(&window, nullptr, "OpenGL Window", resolution));
  mERROR_CHECK(mRenderParams_InitializeToDefault());

  mERROR_CHECK(mRenderParams_SetDoubleBuffering(true));
  mERROR_CHECK(mRenderParams_SetMultisampling(4));
  mERROR_CHECK(mRenderParams_SetVsync(false));

  mPtr<mMeshFactory<mMesh2dPosition, mMeshTexcoord, mMeshScale2dUniform>> meshFactory;
  mDEFER_DESTRUCTION(&meshFactory, mMeshFactory_Destroy);
  mERROR_CHECK(mMeshFactory_Create(&meshFactory, nullptr, mRP_RM_TriangleStrip));

  mERROR_CHECK(mMeshFactory_GrowBack(meshFactory, 4));
  mERROR_CHECK(mMeshFactory_AppendData(meshFactory, mMesh2dPosition(-1, -1), mMeshTexcoord(0, 1), mMeshScale2dUniform()));
  mERROR_CHECK(mMeshFactory_AppendData(meshFactory, mMesh2dPosition(-1,  1), mMeshTexcoord(0, 0), mMeshScale2dUniform()));
  mERROR_CHECK(mMeshFactory_AppendData(meshFactory, mMesh2dPosition( 1, -1), mMeshTexcoord(1, 1), mMeshScale2dUniform()));
  mERROR_CHECK(mMeshFactory_AppendData(meshFactory, mMesh2dPosition( 1,  1), mMeshTexcoord(1, 0), mMeshScale2dUniform()));

  mPtr<mMesh> mesh;
  mDEFER_DESTRUCTION(&mesh, mMesh_Destroy);
  mERROR_CHECK(mMeshFactory_CreateMesh(meshFactory, &mesh, nullptr, image));
  mERROR_CHECK(mMeshFactory_Clear(meshFactory));

  mERROR_CHECK(mMeshFactory_GrowBack(meshFactory, 5));
  mERROR_CHECK(mMeshFactory_AppendData(meshFactory, mMesh2dPosition(0, 0), mMeshTexcoord(1, 0), mMeshScale2dUniform()));
  mERROR_CHECK(mMeshFactory_AppendData(meshFactory, mMesh2dPosition(0, 4), mMeshTexcoord(1, 1), mMeshScale2dUniform()));
  mERROR_CHECK(mMeshFactory_AppendData(meshFactory, mMesh2dPosition(2, 0), mMeshTexcoord(0, 0), mMeshScale2dUniform()));
  mERROR_CHECK(mMeshFactory_AppendData(meshFactory, mMesh2dPosition(2, 2), mMeshTexcoord(0, 1), mMeshScale2dUniform()));
  mERROR_CHECK(mMeshFactory_AppendData(meshFactory, mMesh2dPosition(3, 3), mMeshTexcoord(1, 0), mMeshScale2dUniform()));

  mPtr<mMesh> mesh2;
  mDEFER_DESTRUCTION(&mesh2, mMesh_Destroy);
  mERROR_CHECK(mMeshFactory_CreateMesh(meshFactory, &mesh2, nullptr, image));

  mPtr<mSpriteBatch<mSBEColour, mSBERotation, mSBETextureFlip, mSBETextureCrop>> spriteBatch;
  mDEFER_DESTRUCTION(&spriteBatch, mSpriteBatch_Destroy);
  mERROR_CHECK(mSpriteBatch_Create(&spriteBatch, nullptr));

  mPtr<mTexture> texture;
  mDEFER_DESTRUCTION(&texture, mSharedPointer_Destroy);
  mERROR_CHECK(mResourceManager_GetResource(&texture, std::string("C:/data/transparent.png")));

  size_t frame = 0;

  while (true)
  {
    frame++;
    mRenderParams_ClearTargetDepthAndColour(mVector(mSin((frame) / 255.0f) / 4.0f + 0.25f, mSin((frame) / 255.0f) / 4.0f + 0.25f, mSin((frame) / 255.0f) / 4.0f + 0.25f, 1.0f));

    mTimeStamp before;
    mERROR_CHECK(mTimeStamp_Now(&before));

    mERROR_CHECK(mShader_SetUniform(mesh->shader, mMeshScale2dUniform::uniformName(), 1.5f * mesh->textures[0]->resolutionF / mRenderParams_CurrentRenderResolutionF));
    mERROR_CHECK(mMesh_Render(mesh));
    mERROR_CHECK(mShader_SetUniform(mesh2->shader, mMeshScale2dUniform::uniformName(), mesh->textures[0]->resolutionF / mRenderParams_CurrentRenderResolutionF));
    mERROR_CHECK(mMesh_Render(mesh2));
    mERROR_CHECK(mSpriteBatch_Begin(spriteBatch, mSB_SSM_FrontToBack, mSB_AM_AlphaBlend));
    mERROR_CHECK(mSpriteBatch_DrawWithDepth(spriteBatch, texture, mVec2f(200.0f + 100 * mSin(frame / 1490.0f), 200.0f + 100 * mSin(frame / 3490.0f)), 0.9f, mSBEColour(mSin(frame / 1123.0f) / 2 + 1, mSin(frame / 942.0f) / 2 + 1, mSin(frame / 1391.0f) / 2 + 1, mSin(frame / 1234.0f) / 2 + 1), mSBERotation((frame) / 2550.0f), mSBETextureFlip(frame % 2000 > 1000, frame % 4000 > 2000), mSBETextureCrop(mVec2f(mSin(frame / 1240.0f) / 4 + .5f, mSin(frame / 1402.0f) / 4 + .5f), mVec2f(1, 1))));
    mERROR_CHECK(mSpriteBatch_DrawWithDepth(spriteBatch, texture, mVec2f(200.0f + 100 * mCos(frame / 1241.0f), 200.0f + 100 * mCos(frame / 2490.0f)), 0.0f, mSBEColour(mSin(frame / 1123.0f) / 2 + 1, mSin(frame / 942.0f) / 2 + 1, mSin(frame / 1391.0f) / 2 + 1), mSBERotation((frame) / 2550.0f), mSBETextureFlip(), mSBETextureCrop()));
    mERROR_CHECK(mSpriteBatch_DrawWithDepth(spriteBatch, texture, mVec2f(200.0f + 100 * mCos(frame / 1241.0f), 200.0f + 100 * mSin(frame / 1230.0f)), 0.5f, mSBEColour(mCos(frame / 1123.0f) / 2 + 1, mCos(frame / 942.0f) / 2 + 1, mCos(frame / 1391.0f) / 2 + 1), mSBERotation((frame) / 2550.0f), mSBETextureFlip(), mSBETextureCrop(mVec2f(0.2f, 0.3f), mVec2f(0.5f, 0.9f))));
    mERROR_CHECK(mSpriteBatch_End(spriteBatch));

    mGL_ERROR_CHECK();

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
