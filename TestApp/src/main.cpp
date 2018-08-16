#include "default.h"
#include "mMediaFileInputHandler.h"
#include "mMediaFileWriter.h"
#include "mThreadPool.h"
#include "SDL.h"
#include <time.h>
#include "GL\glew.h"

mVec2s resolution;
SDL_Window *pWindow = nullptr;
SDL_Surface *pSurface = nullptr;
const size_t subScale = 5;
mPtr<mImageBuffer> bgraImageBuffer = nullptr;
mPtr<mImageBuffer> image;
mPtr<mThreadPool> threadPool = nullptr;
SDL_GLContext glContext;
bool is3dEnabled = false;
GLenum glError = GL_NO_ERROR;

mFUNCTION(OnVideoFramCallback, mPtr<mImageBuffer> &, const mVideoStreamType &);

int main(int, char **)
{
  mFUNCTION_SETUP();

  g_mResult_breakOnError = true;
  mDEFER_DESTRUCTION(&threadPool, mThreadPool_Destroy);
  mERROR_CHECK(mThreadPool_Create(&threadPool, nullptr));

  mDEFER_DESTRUCTION(&bgraImageBuffer, mImageBuffer_Destroy);
  mERROR_CHECK(mImageBuffer_Create(&bgraImageBuffer, nullptr, resolution));

  mDEFER_DESTRUCTION(&image, mImageBuffer_Destroy);
  mERROR_CHECK(mImageBuffer_CreateFromFile(&image, nullptr, "C:/Users/cstiller/Pictures/avatar.png"));

  mPtr<mMediaFileInputHandler> mediaFileHandler;
  mDEFER_DESTRUCTION(&mediaFileHandler, mMediaFileInputHandler_Destroy);
  mERROR_CHECK(mMediaFileInputHandler_Create(&mediaFileHandler, nullptr, L"C:/Users/cstiller/Videos/Converted.mp4", mMediaFileInputHandler_CreateFlags::mMMFIH_CF_VideoEnabled));

  SDL_Init(SDL_INIT_EVERYTHING);

  SDL_DisplayMode displayMode;
  SDL_GetCurrentDisplayMode(0, &displayMode);

  resolution.x = displayMode.w / 2;
  resolution.y = displayMode.h / 2;

  mDEFER_DESTRUCTION(pWindow, SDL_DestroyWindow);
  pWindow = SDL_CreateWindow("VideoStream Renderer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, (int)resolution.x, (int)resolution.y, /*SDL_WINDOW_FULLSCREEN | */SDL_WINDOW_OPENGL);
  mERROR_IF(pWindow == nullptr, mR_ArgumentNull);

  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

  glContext = SDL_GL_CreateContext(pWindow);
  glewExperimental = GL_TRUE;
  mERROR_IF((glError = glewInit()) != GL_NO_ERROR, mR_InternalError);

  SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
  SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);
  SDL_GL_SetSwapInterval(1);

  //if (SDL_GL_SetAttribute(SDL_GL_STEREO, 1) == 0)
  //{
  //  is3dEnabled = true;
  //  mPRINT("3d enabled.");
  //}

  // Prepare GL Rendering
  {
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    mVec2f vertices[] = { { -1, -1 },{ 0, 1 },{ -1, 1 },{ 0, 0 },{ 1, -1 },{ 1, 1 },{ 1, 1 },{ 1, 0 } };

    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

#define GLSL(src) "#version 150 core\n" #src

    const char* vertexSource = GLSL(
      in vec2 position;
      in vec2 texcoord;
      out vec2 Texcoord;

    void main() {
      Texcoord = texcoord;
      gl_Position = vec4(position, 0.0, 1.0);
    }
    );

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexSource, NULL);
    glCompileShader(vertexShader);

    GLint status;
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &status);
    
    if (status != GL_TRUE)
    {
      char buffer[512];
      glGetShaderInfoLog(vertexShader, 512, NULL, buffer);
      mPRINT(buffer);
    }

    const char* fragmentSource = GLSL(
      out vec4 outColor;
      in vec2 Texcoord;
      uniform sampler2D tex0;

    void main() {
      outColor = texture(tex0, Texcoord);
    }
    );
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
    glCompileShader(fragmentShader);

    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &status);

    if (status != GL_TRUE)
    {
      char buffer[512];
      glGetShaderInfoLog(fragmentShader, 512, NULL, buffer);
      mPRINT(buffer);
    }


    // Link the vertex and fragment shader into a shader program
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glBindFragDataLocation(shaderProgram, 0, "outColor");
    glLinkProgram(shaderProgram);
    glUseProgram(shaderProgram);
    mASSERT((glError = glGetError()) == GL_NO_ERROR, "GL ERROR.");

    glActiveTexture(GL_TEXTURE0);
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    mASSERT((glError = glGetError()) == GL_NO_ERROR, "GL ERROR.");
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, (GLsizei)image->currentSize.x, (GLsizei)image->currentSize.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, image->pPixels);
    glUniform1i(glGetUniformLocation(shaderProgram, "tex0"), 0);
    mASSERT((glError = glGetError()) == GL_NO_ERROR, "GL ERROR.");

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    mASSERT((glError = glGetError()) == GL_NO_ERROR, "GL ERROR.");

    GLint posAttrib = glGetAttribLocation(shaderProgram, "position");
    glEnableVertexAttribArray(posAttrib);
    glVertexAttribPointer(posAttrib, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(mVec2f), 0);
    mASSERT((glError = glGetError()) == GL_NO_ERROR, "GL ERROR.");

    GLint texAttrib = glGetAttribLocation(shaderProgram, "texcoord");
    glEnableVertexAttribArray(texAttrib);
    glVertexAttribPointer(texAttrib, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(mVec2f), (void*)(sizeof(mVec2f)));
    mASSERT((glError = glGetError()) == GL_NO_ERROR, "GL ERROR.");
  }

  //mERROR_CHECK(mMediaFileInputHandler_SetVideoCallback(mediaFileHandler, OnVideoFramCallback));
  //mERROR_CHECK(mMediaFileInputHandler_Play(mediaFileHandler));


  size_t frame = 0;

  while (true)
  {
    glClearColor((frame++ & 0xFF) / 255.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    mASSERT((glError = glGetError()) == GL_NO_ERROR, "GL ERROR.");

    SDL_GL_SwapWindow(pWindow);

    //mERROR_CHECK(RenderFrame());
  }

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

  mRETURN_SUCCESS();
}
