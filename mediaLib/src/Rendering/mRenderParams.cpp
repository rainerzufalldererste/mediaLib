// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "mRenderParams.h"
#include "mHardwareWindow.h"

mVec2s mRenderParams_CurrentRenderResolution;
mVec2f mRenderParams_CurrentRenderResolutionF;

mRenderContextId mRenderParams_CurrentRenderContext;

#if defined(mRENDERER_OPENGL)
GLenum mRenderParams_GLError;
#endif

mRenderContext *mRenderParams_pRenderContexts;
size_t mRenderParams_RenderContextCount;

mFUNCTION(mRenderParams_InitializeToDefault)
{
  mFUNCTION_SETUP();

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  mRETURN_SUCCESS();
}

mFUNCTION(mRenderParams_CreateRenderContext, OUT mRenderContextId *pRenderContextId, mPtr<mHardwareWindow> &window)
{
  mFUNCTION_SETUP();

  mERROR_IF(pRenderContextId == nullptr || window == nullptr, mR_ArgumentNull);

  mRenderContext currentRenderContext;
  currentRenderContext.referenceCount = 1;

#if defined(mRENDERER_OPENGL)

  SDL_Window *pWindow;
  mERROR_CHECK(mHardwareWindow_GetSdlWindowPtr(window, &pWindow));

  SDL_GLContext glContext = SDL_GL_CreateContext(pWindow);
  mERROR_IF(glContext == nullptr, mR_InternalError);

  if (mRenderParams_RenderContextCount == 0)
  {
    glewExperimental = GL_TRUE;
    mERROR_IF((mRenderParams_GLError = glewInit()) != GL_NO_ERROR, mR_InternalError);
  }

  currentRenderContext.glContext = glContext;

#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  const mRenderContextId renderContextId = mRenderParams_RenderContextCount;

  ++mRenderParams_RenderContextCount;
  mERROR_CHECK(mAllocator_Reallocate(nullptr, &mRenderParams_pRenderContexts, mRenderParams_RenderContextCount));
  mERROR_CHECK(mAllocator_Move(nullptr, &mRenderParams_pRenderContexts[renderContextId], &currentRenderContext, 1));

  *pRenderContextId = renderContextId;

  mRETURN_SUCCESS();
}

mFUNCTION(mRenderParams_ActivateRenderContext, mPtr<mHardwareWindow> &window, const mRenderContextId renderContextId)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr, mR_ArgumentNull);
  mERROR_IF(renderContextId >= mRenderParams_RenderContextCount, mR_IndexOutOfBounds);

#if defined(mRENDERER_OPENGL)
  SDL_Window *pWindow;
  mERROR_CHECK(mHardwareWindow_GetSdlWindowPtr(window, &pWindow));

  int result = 0;
  mERROR_IF(0 != (result = SDL_GL_MakeCurrent(pWindow, mRenderParams_pRenderContexts[renderContextId].glContext)), mR_RenderingError);
  mRenderParams_CurrentRenderContext = renderContextId;
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mRenderParams_DestroyRenderContext, IN_OUT mRenderContextId *pRenderContextId)
{
  mFUNCTION_SETUP();

  mERROR_IF(pRenderContextId == nullptr, mR_ArgumentNull);
  mERROR_IF(*pRenderContextId >= mRenderParams_RenderContextCount, mR_IndexOutOfBounds);

  const size_t referenceCount = --mRenderParams_pRenderContexts[*pRenderContextId].referenceCount;

  if (referenceCount == 0)
  {
#if defined(mRENDERER_OPENGL)
    SDL_GL_DeleteContext(mRenderParams_pRenderContexts[*pRenderContextId].glContext);
    mRenderParams_pRenderContexts[*pRenderContextId].glContext = nullptr;
#else
    mRETURN_RESULT(mR_NotImplemented);
#endif
  }

  *pRenderContextId = (mRenderContextId)-1;

  mRETURN_SUCCESS();
}

mFUNCTION(mRenderParams_SetCurrentRenderResolution, const mVec2s &resolution)
{
  mFUNCTION_SETUP();

  mRenderParams_CurrentRenderResolution = resolution;
  mRenderParams_CurrentRenderResolutionF = (mVec2f)resolution;

  mRETURN_SUCCESS();
}

mFUNCTION(mRenderParams_SetVsync, const bool vsync)
{
  mFUNCTION_SETUP();

#if defined(mRENDERER_OPENGL)
  int result = 0;
  mERROR_IF(0 != (result = SDL_GL_SetSwapInterval(vsync ? 1 : 0)), mR_OperationNotSupported);
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif
  mRETURN_SUCCESS();
}

mFUNCTION(mRenderParams_SetDoubleBuffering, const bool doubleBuffering)
{
  mFUNCTION_SETUP();

#if defined(mRENDERER_OPENGL)
  int result = 0;
  mERROR_IF(0 != (result = SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, doubleBuffering ? 1 : 0)), mR_OperationNotSupported);
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mRenderParams_SetMultisampling, const size_t count)
{
  mFUNCTION_SETUP();

  mERROR_IF(count > INT_MAX, mR_ArgumentOutOfBounds);

  int result = 0;

  if (count > 0)
  {
#if defined(mRENDERER_OPENGL)
    mERROR_IF(0 != (result = SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1)), mR_OperationNotSupported);
    mERROR_IF(0 != (result = SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, (int)count)), mR_OperationNotSupported);
#else
    mRETURN_RESULT(mR_NotImplemented);
#endif
  }
  else
  {
#if defined(mRENDERER_OPENGL)
    mERROR_IF(0 != (result = SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 0)), mR_OperationNotSupported);
    mERROR_IF(0 != (result = SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 0)), mR_OperationNotSupported);
#else
    mRETURN_RESULT(mR_NotImplemented);
#endif
  }

  mRETURN_SUCCESS();
}

bool mRenderParams_IsStereo = false;

mFUNCTION(mRenderParams_SetStereo3d, const bool enabled)
{
  mFUNCTION_SETUP();

  GLboolean supportsStereo = false;
  glGetBooleanv(GL_STEREO, &supportsStereo);

  mERROR_IF(!supportsStereo && enabled, mR_OperationNotSupported);

  mERROR_IF(0 > SDL_GL_SetAttribute(SDL_GL_STEREO, 1), mR_InternalError);

  mRenderParams_IsStereo = true;

  mRETURN_SUCCESS();
}

mFUNCTION(mRenderParams_SetStereo3dBuffer, const mRenderParams_StereoRenderBuffer buffer)
{
  mFUNCTION_SETUP();

  mERROR_IF(!mRenderParams_IsStereo, mR_OperationNotSupported);

  switch (buffer)
  {
  case mRP_SRB_LeftEye:
    glDrawBuffer(GL_BACK_LEFT);
    break;

  case mRP_SRB_RightEye:
    glDrawBuffer(GL_BACK_RIGHT);
    break;

  default:
    mRETURN_RESULT(mR_OperationNotSupported);
    break;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mRenderParams_ClearTargetColour, const mVector & colour)
{
  mFUNCTION_SETUP();
#if defined(mRENDERER_OPENGL)
  glClearColor(colour.x, colour.y, colour.z, colour.w);
  glClear(GL_COLOR_BUFFER_BIT);
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif
  mRETURN_SUCCESS();
}

mFUNCTION(mRenderParams_ClearDepth)
{
  mFUNCTION_SETUP();
#if defined(mRENDERER_OPENGL)
  glClear(GL_DEPTH_BUFFER_BIT);
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif
  mRETURN_SUCCESS();
}

mFUNCTION(mRenderParams_ClearTargetDepthAndColour, const mVector & colour)
{
  mFUNCTION_SETUP();
#if defined(mRENDERER_OPENGL)
  glClearColor(colour.x, colour.y, colour.z, colour.w);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif
  mRETURN_SUCCESS();
}

bool mRenderParams_BlendingEnabled = false;

mFUNCTION(mRenderParams_SetBlendingEnabled, const bool enabled /*= true*/)
{
  mFUNCTION_SETUP();

  if (mRenderParams_BlendingEnabled == enabled)
    mRETURN_SUCCESS();

#if defined(mRENDERER_OPENGL)
  if (enabled)
    glEnable(GL_BLEND);
  else
    glDisable(GL_BLEND);

  mGL_DEBUG_ERROR_CHECK();

  mRenderParams_BlendingEnabled = enabled;

#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}
