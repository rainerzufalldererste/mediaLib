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

  mPtr<mMeshFactory<mMesh2dPosition, mMeshTexcoord, mMeshScaleUniform>> meshFactory;
  mDEFER_DESTRUCTION(&meshFactory, mMeshFactory_Destroy);
  mERROR_CHECK(mMeshFactory_Create(&meshFactory, nullptr));

  mERROR_CHECK(mMeshFactory_GrowBack(meshFactory, 4));
  mERROR_CHECK(mMeshFactory_AppendData(meshFactory, mMesh2dPosition(-1, -1), mMeshTexcoord(0, 1), mMeshScaleUniform()));
  mERROR_CHECK(mMeshFactory_AppendData(meshFactory, mMesh2dPosition(-1,  1), mMeshTexcoord(0, 0), mMeshScaleUniform()));
  mERROR_CHECK(mMeshFactory_AppendData(meshFactory, mMesh2dPosition( 1, -1), mMeshTexcoord(1, 1), mMeshScaleUniform()));
  mERROR_CHECK(mMeshFactory_AppendData(meshFactory, mMesh2dPosition( 1,  1), mMeshTexcoord(1, 0), mMeshScaleUniform()));

  mPtr<mMesh> mesh;
  mDEFER_DESTRUCTION(&mesh, mMesh_Destroy);
  mERROR_CHECK(mMeshFactory_CreateMesh(meshFactory, &mesh, nullptr));

  GLuint vbo;
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);

  mVec2f vertices[] = { { -1, -1 },{ 0, 1 },{ -1, 1 },{ 0, 0 },{ 1, -1 },{ 1, 1 },{ 1, 1 },{ 1, 0 } };

  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  const char* vertexSource = mGLSL(
    in vec2 position;
  in vec2 texcoord;
  uniform vec2 screenSize;
  uniform vec2 scale;
  out vec2 Texcoord;

  void main() {
    Texcoord = texcoord;
    gl_Position = vec4((position / screenSize) * scale, 0.0, 1.0);
  }
  );

  const char* fragmentSource = mGLSL(
    out vec4 outColor;
  in vec2 Texcoord;
  uniform sampler2D tex0;

  void main() {
    outColor = texture(tex0, Texcoord);
  }
  );

  mShader shader;
  mDEFER_DESTRUCTION(&shader, mShader_Destroy);
  mERROR_CHECK(mShader_Create(&shader, vertexSource, fragmentSource));
  mERROR_CHECK(mShader_Bind(shader));

  mTexture texture;
  mDEFER_DESTRUCTION(&texture, mTexture_Destroy);
  mERROR_CHECK(mTexture_Create(&texture, image));
  mERROR_CHECK(mTexture_Bind(texture));

  mERROR_CHECK(mShader_SetUniform(shader, "tex0", texture));

  GLint posAttrib = glGetAttribLocation(shader.shaderProgram, "position");
  glEnableVertexAttribArray(posAttrib);
  glVertexAttribPointer(posAttrib, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(mVec2f), 0);
  mGL_ERROR_CHECK();

  GLint texAttrib = glGetAttribLocation(shader.shaderProgram, "texcoord");
  glEnableVertexAttribArray(texAttrib);
  glVertexAttribPointer(texAttrib, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(mVec2f), (void*)(sizeof(mVec2f)));
  mGL_ERROR_CHECK();

  GLint screenSizeAttrib = glGetUniformLocation(shader.shaderProgram, "screenSize");
  glUniform2f(screenSizeAttrib, mRenderParams_CurrentRenderResolutionF.x, mRenderParams_CurrentRenderResolutionF.y);
  mGL_ERROR_CHECK();

  size_t frame = 0;

  while (true)
  {
    mRenderParams_ClearTargetDepthAndColour(mVector(mSin((frame++) / 255.0f) / 4.0f + 0.25f, mSin((frame++) / 255.0f) / 4.0f + 0.25f, mSin((frame++) / 255.0f) / 4.0f + 0.25f, 1.0f));

    mERROR_CHECK(mShader_SetUniform(shader, "scale", mVec2f(image->currentSize) + 100 * mSin(frame / 1000.0f)));
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

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
