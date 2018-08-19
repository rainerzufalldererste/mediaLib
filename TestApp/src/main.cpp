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

  mERROR_CHECK(mRenderParams_SetDoubleBuffering(true));
  mERROR_CHECK(mRenderParams_SetMultisampling(4));
  mERROR_CHECK(mRenderParams_SetVsync(true));

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

  mPtr<mMeshFactory<mMesh2dPosition, mMeshTexcoord, mMeshScale2dUniform>> meshFactory2;
  mDEFER_DESTRUCTION(&meshFactory2, mMeshFactory_Destroy);
  mERROR_CHECK(mMeshFactory_Create(&meshFactory2, nullptr, mRP_RM_TriangleStrip));

  mERROR_CHECK(mMeshFactory_GrowBack(meshFactory2, 4));
  mERROR_CHECK(mMeshFactory_AppendData(meshFactory2, mMesh2dPosition(0, 0), mMeshTexcoord(0, 1), mMeshScale2dUniform()));
  mERROR_CHECK(mMeshFactory_AppendData(meshFactory2, mMesh2dPosition(0, 4), mMeshTexcoord(0, 0), mMeshScale2dUniform()));
  mERROR_CHECK(mMeshFactory_AppendData(meshFactory2, mMesh2dPosition(2, 0), mMeshTexcoord(1, 1), mMeshScale2dUniform()));
  mERROR_CHECK(mMeshFactory_AppendData(meshFactory2, mMesh2dPosition(2, 2), mMeshTexcoord(1, 0), mMeshScale2dUniform()));

  mPtr<mMesh> mesh2;
  mDEFER_DESTRUCTION(&mesh2, mMesh_Destroy);
  mERROR_CHECK(mMeshFactory_CreateMesh(meshFactory2, &mesh2, nullptr, image));

  size_t frame = 0;

  mERROR_CHECK(mShader_SetUniform(mesh->shader, mMeshScale2dUniform::uniformName(), 1.5f * mesh->textures[0]->resolutionF / mRenderParams_CurrentRenderResolutionF));
  mERROR_CHECK(mShader_SetUniform(mesh2->shader, mMeshScale2dUniform::uniformName(), mesh->textures[0]->resolutionF / mRenderParams_CurrentRenderResolutionF));

  while (true)
  {
    frame++;
    mRenderParams_ClearTargetDepthAndColour(mVector(mSin((frame) / 255.0f) / 4.0f + 0.25f, mSin((frame) / 255.0f) / 4.0f + 0.25f, mSin((frame) / 255.0f) / 4.0f + 0.25f, 1.0f));

    mERROR_CHECK(mMesh_Render(mesh));
    mERROR_CHECK(mMesh_Render(mesh2));

    mGL_ERROR_CHECK();

    mERROR_CHECK(mHardwareWindow_Swap(window));

    SDL_Event _event;
    while (SDL_PollEvent(&_event))
      if (_event.type == SDL_QUIT || (_event.type == SDL_KEYDOWN && _event.key.keysym.sym == SDLK_ESCAPE))
        goto end;
  }

end:;

  mRETURN_SUCCESS();
}
