#include "mRenderParams.h"
#include "mHardwareWindow.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "qgl523t3vQYdvPDcFGW8An9gs1fWNQGjMnAf2GapKD0ru9ZEb3bc8MpLfZdV2CJ8IQdzzsAZRoW4MvB4"
#endif

mVec2s mRenderParams_CurrentRenderResolution;
mVec2f mRenderParams_CurrentRenderResolutionF;

mVec2s mRenderParams_BackBufferResolution;
mVec2f mRenderParams_BackBufferResolutionF;

mRenderContextId mRenderParams_CurrentRenderContext;

#if defined(mRENDERER_OPENGL)
GLenum mRenderParams_GLError = GL_NO_ERROR;
#endif

mRenderContext *mRenderParams_pRenderContexts = nullptr;
size_t mRenderParams_RenderContextCount = 0;
size_t mRenderParams_InitializedRenderContextCount = 0;

mFUNCTION(mRenderParams_InitializeToDefault)
{
  mFUNCTION_SETUP();

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
  ++mRenderParams_InitializedRenderContextCount;
  mERROR_CHECK(mAllocator_Reallocate(&mDefaultAllocator, &mRenderParams_pRenderContexts, mRenderParams_RenderContextCount));
  mERROR_CHECK(mMemmove(&mRenderParams_pRenderContexts[renderContextId], &currentRenderContext, 1));

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
    --mRenderParams_InitializedRenderContextCount;
#else
    mRETURN_RESULT(mR_NotImplemented);
#endif
  }

  *pRenderContextId = (mRenderContextId)-1;

  mRETURN_SUCCESS();
}

#if defined(mRENDERER_OPENGL)
mFUNCTION(mRenderParams_GetRenderContext, mPtr<mHardwareWindow> &window, OUT SDL_GLContext *pContext)
{
  mFUNCTION_SETUP();

  mERROR_IF(window == nullptr || pContext == nullptr, mR_ArgumentNull);

  mRenderContextId renderContextId;
  mERROR_CHECK(mHardwareWindow_GetRenderContextId(window, &renderContextId));
  mERROR_IF(renderContextId >= mRenderParams_RenderContextCount, mR_IndexOutOfBounds);
  
  *pContext = mRenderParams_pRenderContexts[renderContextId].glContext;

  mRETURN_SUCCESS();
}
#endif

mFUNCTION(mRenderParams_SetCurrentRenderResolution, const mVec2s &resolution)
{
  mFUNCTION_SETUP();

  mRenderParams_CurrentRenderResolution = resolution;
  mRenderParams_CurrentRenderResolutionF = (mVec2f)resolution;

#if defined (mRENDERER_OPENGL)
  glViewport(0, 0, (GLsizei)resolution.x, (GLsizei)resolution.y);
#endif

  mGL_DEBUG_ERROR_CHECK();

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

mFUNCTION(mRenderParams_SupportsStereo3d, OUT bool *pSupportsStereo3d)
{
  mFUNCTION_SETUP();

  GLboolean supportsStereo;
  glGetBooleanv(GL_STEREO, &supportsStereo);

  *pSupportsStereo3d = supportsStereo != 0;

  mRETURN_SUCCESS();
}

bool mRenderParams_IsStereo = false;

mFUNCTION(mRenderParams_SetStereo3d, const bool enabled)
{
  mFUNCTION_SETUP();

  bool supportsStereo = false;
  mERROR_CHECK(mRenderParams_SupportsStereo3d(&supportsStereo));

  mERROR_IF(!supportsStereo && enabled, mR_OperationNotSupported);
  mERROR_IF(0 > SDL_GL_SetAttribute(SDL_GL_STEREO, 1), mR_InternalError);

  mRenderParams_IsStereo = true;

  mRETURN_SUCCESS();
}

mFUNCTION(mRenderParams_ShowCursor, const bool isVisible)
{
  mFUNCTION_SETUP();

  SDL_ShowCursor(isVisible ? SDL_ENABLE : SDL_DISABLE);

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

mFUNCTION(mRenderParams_ClearTargetColour, const mVector &colour)
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
  glDepthMask(GL_TRUE);
  glClear(GL_DEPTH_BUFFER_BIT);
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif
  mRETURN_SUCCESS();
}

mFUNCTION(mRenderParams_ClearTargetDepthAndColour, const mVector &colour)
{
  mFUNCTION_SETUP();
#if defined(mRENDERER_OPENGL)
  glDepthMask(GL_TRUE);
  glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
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

mFUNCTION(mRenderParams_SetDepthTestEnabled, const bool enabled /*= true*/)
{
  mFUNCTION_SETUP();

#if defined(mRENDERER_OPENGL)
  (enabled ? glEnable : glDisable)(GL_DEPTH_TEST);
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mRenderParams_SetDepthMaskEnabled, const bool enabled)
{
  mFUNCTION_SETUP();

#if defined(mRENDERER_OPENGL)
  glDepthMask(enabled ? GL_TRUE : GL_FALSE);
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}

mFUNCTION(mRenderParams_SetScissorTestEnabled, const bool enabled)
{
  mFUNCTION_SETUP();

#if defined(mRENDERER_OPENGL)
  (enabled ? glEnable : glDisable)(GL_SCISSOR_TEST);
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mRenderParams_GetMaxDepthPrecisionBits, OUT size_t *pDepth)
{
  mFUNCTION_SETUP();

  mERROR_IF(pDepth == nullptr, mR_ArgumentNull);

#if defined(mRENDERER_OPENGL)
  const HDC hdc = wglGetCurrentDC();
  int32_t pixelFormatCount = 1;
  size_t maxDepthBits = 0;
  
  for (size_t i = 1; i <= pixelFormatCount; i++)
  {
    PIXELFORMATDESCRIPTOR pixelFormat = { 0 };

    pixelFormatCount = DescribePixelFormat(hdc, (int32_t)i, sizeof(PIXELFORMATDESCRIPTOR), &pixelFormat);
    mERROR_IF(pixelFormatCount == 0, mR_OperationNotSupported);

    if (pixelFormat.cDepthBits > maxDepthBits && pixelFormat.cBlueBits >= 8 && pixelFormat.cRedBits >= 8 && pixelFormat.cGreenBits >= 8)
      maxDepthBits = (size_t)pixelFormat.cDepthBits;
  }

  *pDepth = maxDepthBits;
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mRenderParams_SetDepthPrecisionBits, const size_t bits)
{
  mFUNCTION_SETUP();

#if defined(mRENDERER_OPENGL)
  mERROR_IF(0 != SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, (int32_t)bits), mR_OperationNotSupported);
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mRenderParams_SetBlendFunc, const mRenderParam_BlendFunc blendFunc)
{
  mFUNCTION_SETUP();

  switch (blendFunc)
  {
  case mRP_BF_NoAlpha:
    mERROR_CHECK(mRenderParams_SetBlendingEnabled(false));
    break;

  case mRP_BF_Additive:
    mERROR_CHECK(mRenderParams_SetBlendingEnabled(true));
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glBlendEquation(GL_FUNC_ADD);
    break;

  case mRP_BF_AlphaBlend:
    mERROR_CHECK(mRenderParams_SetBlendingEnabled(true));
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBlendEquation(GL_FUNC_ADD);
    break;

  case mRP_BF_Premultiplied:
    mERROR_CHECK(mRenderParams_SetBlendingEnabled(true));
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glBlendEquation(GL_FUNC_ADD);
    break;

  case mRP_BF_Override:
    mERROR_CHECK(mRenderParams_SetBlendingEnabled(true));
    glBlendFunc(GL_ONE, GL_ZERO);
    glBlendEquation(GL_FUNC_ADD);
    break;

  case mRP_BF_AlphaMask:
    mERROR_CHECK(mRenderParams_SetBlendingEnabled(true));
    glBlendFuncSeparate(GL_ZERO, GL_DST_COLOR, GL_ZERO, GL_SRC_ALPHA);
    glBlendEquation(GL_FUNC_ADD);
    break;

  default:
    mRETURN_RESULT(mR_OperationNotSupported);
    break;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mRenderParams_SetDepthFunc, const mRenderParams_DepthFunc depthFunc)
{
  mFUNCTION_SETUP();

  switch (depthFunc)
  {
  case mRP_DF_NoDepth:
    mERROR_CHECK(mRenderParams_SetDepthTestEnabled(false));
    break;

  case mRP_DF_Less:
    mERROR_CHECK(mRenderParams_SetDepthTestEnabled(true));
    glDepthFunc(GL_LESS);
    break;

  case mRP_DF_Greater:
    mERROR_CHECK(mRenderParams_SetDepthTestEnabled(true));
    glDepthFunc(GL_GREATER);
    break;

  case mRP_DF_Equal:
    mERROR_CHECK(mRenderParams_SetDepthTestEnabled(true));
    glDepthFunc(GL_EQUAL);
    break;

  case mRP_DF_LessOrEqual:
    mERROR_CHECK(mRenderParams_SetDepthTestEnabled(true));
    glDepthFunc(GL_LEQUAL);
    break;

  case mRP_DF_GreaterOrEqual:
    mERROR_CHECK(mRenderParams_SetDepthTestEnabled(true));
    glDepthFunc(GL_GEQUAL);
    break;

  case mRP_DF_NotEqual:
    mERROR_CHECK(mRenderParams_SetDepthTestEnabled(true));
    glDepthFunc(GL_NOTEQUAL);
    break;

  case mRP_DF_Always:
    mERROR_CHECK(mRenderParams_SetDepthTestEnabled(true));
    glDepthFunc(GL_ALWAYS);
    break;


  default:
    mRETURN_RESULT(mR_OperationNotSupported);
    break;
  }

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}

mFUNCTION(mRenderParams_GetCurrentGLContext_HGLRC, HGLRC *pGLContext)
{
  mFUNCTION_SETUP();

  mERROR_IF(pGLContext == nullptr, mR_ArgumentNull);

  *pGLContext = wglGetCurrentContext();

  mRETURN_SUCCESS();
}

mFUNCTION(mRenderParams_GetCurrentGLContext_HDC, HDC *pGLDrawable)
{
  mFUNCTION_SETUP();

  mERROR_IF(pGLDrawable == nullptr, mR_ArgumentNull);

  *pGLDrawable = wglGetCurrentDC();

  mRETURN_SUCCESS();
}

mFUNCTION(mRenderParams_GetCurrentGLContext_HWND, HWND *pGLWindow)
{
  mFUNCTION_SETUP();

  mERROR_IF(pGLWindow == nullptr, mR_ArgumentNull);

  *pGLWindow = WindowFromDC(wglGetCurrentDC());

  mRETURN_SUCCESS();
}

mFUNCTION(mTexture2DParams_ApplyToBoundTexture, const mTexture2DParams &params, const bool isMultisampleTexture /* = false */)
{
  mFUNCTION_SETUP();

  mERROR_IF(params.minFilter == 0 || params.magFilter == 0 || params.wrapModeX == 0 || params.wrapModeY == 0, mR_InvalidParameter);

#ifdef mRENDERER_OPENGL
  if (!isMultisampleTexture)
  {
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, params.minFilter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, params.magFilter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, params.wrapModeX);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, params.wrapModeY);

    mGL_ERROR_CHECK();
  }
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mTexture3DParams_ApplyToBoundTexture, const mTexture3DParams &params, const bool isMultisampleTexture /* = false */)
{
  mFUNCTION_SETUP();

  mERROR_IF(params.minFilter == 0 || params.magFilter == 0 || params.wrapModeX == 0 || params.wrapModeY == 0 || params.wrapModeZ == 0, mR_InvalidParameter);

#ifdef mRENDERER_OPENGL
  if (!isMultisampleTexture)
  {
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, params.minFilter);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, params.magFilter);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, params.wrapModeX);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, params.wrapModeY);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, params.wrapModeZ);

    mGL_ERROR_CHECK();
  }
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

#ifdef _DEBUG
#define mPRINT_FUNC mPRINT_DEBUG
#else
#define mPRINT_FUNC mPRINT
#endif

mFUNCTION(mRenderParams_PrintRenderState, const bool onlyNewValues /* = false */, const bool onlyUpdateValues /* = false */)
{
  mFUNCTION_SETUP();

#ifdef mRENDERER_OPENGL

  if (!onlyUpdateValues)
    mPRINT_FUNC("GL STATE:\n================================================\n");

#define mGL_PRINT_PARAM(func, param, paramString, var, format) \
  do { static var mCONCAT_LITERALS(_var_, param)[4]; \
  var mCONCAT_LITERALS(_local_var_, param)[4]; \
  func(param, &mCONCAT_LITERALS(_local_var_, param)[0]); \
  if (!onlyUpdateValues) \
  { const bool valueIsNew = mCONCAT_LITERALS(_var_, param)[0] != mCONCAT_LITERALS(_local_var_, param)[0]; \
    if (onlyNewValues) \
    { if (valueIsNew) \
      { mPRINT_FUNC(paramString " = " format " [was " format "]\n", mCONCAT_LITERALS(_local_var_, param)[0], mCONCAT_LITERALS(_var_, param)[0]); \
      } \
    } \
    else \
    { mPRINT_FUNC("%s" paramString " = " format "\n", valueIsNew ? " * " : "   ", mCONCAT_LITERALS(_local_var_, param)[0]); \
    } \
  } \
  mCONCAT_LITERALS(_var_, param)[0] = mCONCAT_LITERALS(_local_var_, param)[0]; } while (0)

#define mGL_PRINT_VEC2_PARAM(func, param, paramString, var, format) \
  do { static var mCONCAT_LITERALS(_var_, param)[4]; \
  var mCONCAT_LITERALS(_local_var_, param)[4]; \
  func(param, &mCONCAT_LITERALS(_local_var_, param)[0]); \
  if (!onlyUpdateValues) \
  { const bool valueIsNew = mCONCAT_LITERALS(_var_, param)[0] != mCONCAT_LITERALS(_local_var_, param)[0] || mCONCAT_LITERALS(_var_, param)[1] != mCONCAT_LITERALS(_local_var_, param)[1]; \
    if (onlyNewValues) \
    { if (valueIsNew) \
      { mPRINT_FUNC(paramString " = (" format ", " format ") [was (" format ", " format ")]\n", mCONCAT_LITERALS(_local_var_, param)[0], mCONCAT_LITERALS(_local_var_, param)[1], mCONCAT_LITERALS(_var_, param)[0], mCONCAT_LITERALS(_var_, param)[1]); \
      } \
    } \
    else \
    { mPRINT_FUNC("%s" paramString " = (" format ", " format ")\n", valueIsNew ? " * " : "   ", mCONCAT_LITERALS(_local_var_, param)[0], mCONCAT_LITERALS(_local_var_, param)[1]); \
    } \
  } \
  mCONCAT_LITERALS(_var_, param)[0] = mCONCAT_LITERALS(_local_var_, param)[0]; \
  mCONCAT_LITERALS(_var_, param)[1] = mCONCAT_LITERALS(_local_var_, param)[1]; } while (0)

#define mGL_PRINT_VEC3_PARAM(func, param, paramString, var, format) \
  do { static var mCONCAT_LITERALS(_var_, param)[4]; \
  var mCONCAT_LITERALS(_local_var_, param)[4]; \
  func(param, &mCONCAT_LITERALS(_local_var_, param)[0]); \
  if (!onlyUpdateValues) \
  { const bool valueIsNew = mCONCAT_LITERALS(_var_, param)[0] != mCONCAT_LITERALS(_local_var_, param)[0] || mCONCAT_LITERALS(_var_, param)[1] != mCONCAT_LITERALS(_local_var_, param)[1] || mCONCAT_LITERALS(_var_, param)[2] != mCONCAT_LITERALS(_local_var_, param)[2]; \
    if (onlyNewValues) \
    { if (valueIsNew) \
      { mPRINT_FUNC(paramString " = (" format ", " format ", " format ") [was (" format ", " format ", " format ")]\n", mCONCAT_LITERALS(_local_var_, param)[0], mCONCAT_LITERALS(_local_var_, param)[1], mCONCAT_LITERALS(_local_var_, param)[2], mCONCAT_LITERALS(_var_, param)[0], mCONCAT_LITERALS(_var_, param)[1], mCONCAT_LITERALS(_var_, param)[2]); \
      } \
    } \
    else \
    {  mPRINT_FUNC("%s" paramString " = (" format ", " format ", " format ")\n", valueIsNew ? " * " : "   ", mCONCAT_LITERALS(_local_var_, param)[0], mCONCAT_LITERALS(_local_var_, param)[1], mCONCAT_LITERALS(_local_var_, param)[2]); \
    } \
  } \
  mCONCAT_LITERALS(_var_, param)[0] = mCONCAT_LITERALS(_local_var_, param)[0]; \
  mCONCAT_LITERALS(_var_, param)[1] = mCONCAT_LITERALS(_local_var_, param)[1]; \
  mCONCAT_LITERALS(_var_, param)[2] = mCONCAT_LITERALS(_local_var_, param)[2]; } while (0)

#define mGL_PRINT_VEC4_PARAM(func, param, paramString, var, format) \
  do { static var mCONCAT_LITERALS(_var_, param)[4]; \
  var mCONCAT_LITERALS(_local_var_, param)[4]; \
  func(param, &mCONCAT_LITERALS(_local_var_, param)[0]); \
  if (!onlyUpdateValues) \
  { const bool valueIsNew = mCONCAT_LITERALS(_var_, param)[0] != mCONCAT_LITERALS(_local_var_, param)[0] || mCONCAT_LITERALS(_var_, param)[1] != mCONCAT_LITERALS(_local_var_, param)[1] || mCONCAT_LITERALS(_var_, param)[2] != mCONCAT_LITERALS(_local_var_, param)[2] || mCONCAT_LITERALS(_var_, param)[3] != mCONCAT_LITERALS(_local_var_, param)[3]; \
    if (onlyNewValues) \
    { if (valueIsNew) \
      { mPRINT_FUNC(paramString " = (" format ", " format ", " format ", " format ") [was (" format ", " format ", " format ", " format ")]\n", mCONCAT_LITERALS(_local_var_, param)[0], mCONCAT_LITERALS(_local_var_, param)[1], mCONCAT_LITERALS(_local_var_, param)[2], mCONCAT_LITERALS(_local_var_, param)[3], mCONCAT_LITERALS(_var_, param)[0], mCONCAT_LITERALS(_var_, param)[1], mCONCAT_LITERALS(_var_, param)[2], mCONCAT_LITERALS(_var_, param)[3]); \
      } \
    } \
    else \
    {  mPRINT_FUNC("%s" paramString " = (" format ", " format ", " format ", " format ")\n", valueIsNew ? " * " : "   ", mCONCAT_LITERALS(_local_var_, param)[0], mCONCAT_LITERALS(_local_var_, param)[1], mCONCAT_LITERALS(_local_var_, param)[2], mCONCAT_LITERALS(_local_var_, param)[3]); \
    } \
  } \
  mCONCAT_LITERALS(_var_, param)[0] = mCONCAT_LITERALS(_local_var_, param)[0]; \
  mCONCAT_LITERALS(_var_, param)[1] = mCONCAT_LITERALS(_local_var_, param)[1]; \
  mCONCAT_LITERALS(_var_, param)[2] = mCONCAT_LITERALS(_local_var_, param)[2]; \
  mCONCAT_LITERALS(_var_, param)[3] = mCONCAT_LITERALS(_local_var_, param)[3]; } while (0)

#define mGL_PRINT_FRAMEBUFFER_ATTACHMENT_PARAM(target, attachment, param) \
  do \
  { static GLint __var__ = 0; \
    GLint __local_var; \
    glGetFramebufferAttachmentParameteriv(target, attachment, param, &__local_var); \
    if (!onlyUpdateValues) \
    { const bool valueIsNew = __local_var != __var__; \
      if (onlyNewValues) \
      { if (valueIsNew) \
        { mPRINT_FUNC(#target ": " #attachment " (" #param ") = %" PRIi32 " [was %" PRIi32 "]\n", __local_var, __var__); \
        } \
      } \
      else \
      { mPRINT_FUNC("%s" #target ": " #attachment " (" #param ")" " = %" PRIi32 "\n", valueIsNew ? " * " : "   ", __local_var, __var__); \
      } \
    } \
    __var__ = __local_var; \
  } while (0)

#define mGL_PRINT_DOUBLE_PARAM(param) \
  mGL_PRINT_PARAM(glGetDoublev, param, #param, double_t, "%f"); \
  mGL_ERROR_CHECK()

#define mGL_PRINT_DOUBLE_VEC2_PARAM(param) \
  mGL_PRINT_VEC2_PARAM(glGetDoublev, param, #param, double_t, "%f"); \
  mGL_ERROR_CHECK()

#define mGL_PRINT_DOUBLE_VEC3_PARAM(param) \
  mGL_PRINT_VEC3_PARAM(glGetDoublev, param, #param, double_t, "%f"); \
  mGL_ERROR_CHECK()

#define mGL_PRINT_DOUBLE_VEC4_PARAM(param) \
  mGL_PRINT_VEC4_PARAM(glGetDoublev, param, #param, double_t, "%f"); \
  mGL_ERROR_CHECK()

#define mGL_PRINT_INTEGER_PARAM(param) \
  mGL_PRINT_PARAM(glGetIntegerv, param, #param, int32_t, "%" PRIi32); \
  mGL_ERROR_CHECK()

#define mGL_PRINT_INTEGER_VEC2_PARAM(param) \
  mGL_PRINT_VEC2_PARAM(glGetIntegerv, param, #param, int32_t, "%" PRIi32); \
  mGL_ERROR_CHECK()

#define mGL_PRINT_INTEGER_VEC4_PARAM(param) \
  mGL_PRINT_VEC4_PARAM(glGetIntegerv, param, #param, int32_t, "%" PRIi32); \
  mGL_ERROR_CHECK()

#define mGL_PRINT_BOOL_PARAM(param) \
  mGL_PRINT_PARAM(glGetBooleanv, param, #param, GLboolean, "%" PRIu8); \
  mGL_ERROR_CHECK()

#define mGL_PRINT_BOOL_VEC4_PARAM(param) \
  mGL_PRINT_VEC4_PARAM(glGetBooleanv, param, #param, GLboolean, "%" PRIu8); \
  mGL_ERROR_CHECK()

  mGL_PRINT_INTEGER_PARAM(GL_ACCUM_ALPHA_BITS);
  mGL_PRINT_INTEGER_PARAM(GL_ACCUM_BLUE_BITS);
  mGL_PRINT_DOUBLE_VEC4_PARAM(GL_ACCUM_CLEAR_VALUE);
  mGL_PRINT_INTEGER_PARAM(GL_ACCUM_GREEN_BITS);
  mGL_PRINT_INTEGER_PARAM(GL_ACCUM_RED_BITS);
  mGL_PRINT_INTEGER_PARAM(GL_ACTIVE_TEXTURE);
  mGL_PRINT_INTEGER_PARAM(GL_ALPHA_BITS);
  mGL_PRINT_INTEGER_PARAM(GL_ALPHA_SCALE);
  mGL_PRINT_INTEGER_PARAM(GL_ALPHA_TEST);
  mGL_PRINT_INTEGER_PARAM(GL_ALPHA_TEST_FUNC);
  mGL_PRINT_DOUBLE_PARAM(GL_ALPHA_TEST_REF);
  mGL_PRINT_INTEGER_PARAM(GL_ARRAY_BUFFER_BINDING);
  mGL_PRINT_BOOL_PARAM(GL_AUTO_NORMAL);
  mGL_PRINT_INTEGER_PARAM(GL_AUX_BUFFERS);
  mGL_PRINT_BOOL_PARAM(GL_BLEND);
  mGL_PRINT_INTEGER_VEC4_PARAM(GL_BLEND_COLOR);
  mGL_PRINT_INTEGER_PARAM(GL_BLEND_DST_ALPHA);
  mGL_PRINT_INTEGER_PARAM(GL_BLEND_DST_RGB);
  mGL_PRINT_INTEGER_PARAM(GL_BLEND_EQUATION_RGB);
  mGL_PRINT_INTEGER_PARAM(GL_BLEND_EQUATION_ALPHA);
  mGL_PRINT_INTEGER_PARAM(GL_BLEND_SRC_ALPHA);
  mGL_PRINT_INTEGER_PARAM(GL_BLEND_SRC_RGB);
  mGL_PRINT_INTEGER_PARAM(GL_BLUE_BITS);
  mGL_PRINT_INTEGER_PARAM(GL_BLUE_SCALE);
  mGL_PRINT_INTEGER_PARAM(GL_CLIENT_ACTIVE_TEXTURE);
  mGL_PRINT_BOOL_PARAM(GL_CLIP_PLANE0);
  mGL_PRINT_BOOL_PARAM(GL_CLIP_PLANE1);
  mGL_PRINT_BOOL_PARAM(GL_CLIP_PLANE2);
  mGL_PRINT_BOOL_PARAM(GL_CLIP_PLANE3);
  mGL_PRINT_BOOL_PARAM(GL_CLIP_PLANE4);
  mGL_PRINT_BOOL_PARAM(GL_CLIP_PLANE5);
  mGL_PRINT_BOOL_PARAM(GL_COLOR_ARRAY);
  mGL_PRINT_INTEGER_PARAM(GL_COLOR_ARRAY_BUFFER_BINDING);
  mGL_PRINT_INTEGER_PARAM(GL_COLOR_ARRAY_SIZE);
  mGL_PRINT_INTEGER_PARAM(GL_COLOR_ARRAY_STRIDE);
  mGL_PRINT_INTEGER_PARAM(GL_COLOR_ARRAY_TYPE);
  mGL_PRINT_DOUBLE_VEC4_PARAM(GL_COLOR_CLEAR_VALUE);
  mGL_PRINT_BOOL_PARAM(GL_COLOR_LOGIC_OP);
  mGL_PRINT_BOOL_PARAM(GL_COLOR_MATERIAL);
  mGL_PRINT_BOOL_PARAM(GL_COLOR_LOGIC_OP);
  mGL_PRINT_BOOL_PARAM(GL_COLOR_MATERIAL);
  mGL_PRINT_INTEGER_PARAM(GL_COLOR_MATERIAL_FACE);
  mGL_PRINT_INTEGER_PARAM(GL_COLOR_MATERIAL_PARAMETER);
  mGL_PRINT_BOOL_PARAM(GL_COLOR_SUM);
  mGL_PRINT_BOOL_PARAM(GL_COLOR_TABLE);
  mGL_PRINT_BOOL_VEC4_PARAM(GL_COLOR_WRITEMASK);
  mGL_PRINT_BOOL_PARAM(GL_CONVOLUTION_1D);
  mGL_PRINT_BOOL_PARAM(GL_CONVOLUTION_2D);
  mGL_PRINT_BOOL_PARAM(GL_CULL_FACE);
  mGL_PRINT_INTEGER_PARAM(GL_CULL_FACE_MODE);
  mGL_PRINT_DOUBLE_VEC4_PARAM(GL_CURRENT_COLOR);
  mGL_PRINT_INTEGER_PARAM(GL_CURRENT_FOG_COORD);
  mGL_PRINT_INTEGER_PARAM(GL_CURRENT_INDEX);
  mGL_PRINT_DOUBLE_VEC3_PARAM(GL_CURRENT_NORMAL);
  mGL_PRINT_INTEGER_PARAM(GL_CURRENT_PROGRAM);
  mGL_PRINT_DOUBLE_VEC4_PARAM(GL_CURRENT_RASTER_COLOR);
  mGL_PRINT_INTEGER_PARAM(GL_CURRENT_RASTER_DISTANCE);
  mGL_PRINT_INTEGER_PARAM(GL_CURRENT_RASTER_INDEX);
  mGL_PRINT_INTEGER_PARAM(GL_CURRENT_RASTER_POSITION);
  mGL_PRINT_BOOL_PARAM(GL_CURRENT_RASTER_POSITION_VALID);
  mGL_PRINT_INTEGER_VEC4_PARAM(GL_CURRENT_RASTER_TEXTURE_COORDS);
  mGL_PRINT_DOUBLE_VEC4_PARAM(GL_CURRENT_SECONDARY_COLOR);
  mGL_PRINT_INTEGER_VEC4_PARAM(GL_CURRENT_TEXTURE_COORDS);
  mGL_PRINT_INTEGER_PARAM(GL_DEPTH_BITS);
  mGL_PRINT_DOUBLE_PARAM(GL_DEPTH_CLEAR_VALUE);
  mGL_PRINT_INTEGER_PARAM(GL_DEPTH_FUNC);
  mGL_PRINT_DOUBLE_VEC2_PARAM(GL_DEPTH_RANGE);
  mGL_PRINT_INTEGER_PARAM(GL_DEPTH_SCALE);
  mGL_PRINT_BOOL_PARAM(GL_DEPTH_TEST);
  mGL_PRINT_BOOL_PARAM(GL_DEPTH_WRITEMASK);
  mGL_PRINT_BOOL_PARAM(GL_DITHER);
  mGL_PRINT_BOOL_PARAM(GL_DOUBLEBUFFER);
  mGL_PRINT_INTEGER_PARAM(GL_DRAW_BUFFER);
  mGL_PRINT_INTEGER_PARAM(GL_DRAW_BUFFER0);
  mGL_PRINT_INTEGER_PARAM(GL_DRAW_BUFFER1);
  mGL_PRINT_INTEGER_PARAM(GL_DRAW_BUFFER2);
  mGL_PRINT_INTEGER_PARAM(GL_DRAW_BUFFER3);
  mGL_PRINT_INTEGER_PARAM(GL_DRAW_BUFFER4);
  mGL_PRINT_INTEGER_PARAM(GL_DRAW_BUFFER5);
  mGL_PRINT_INTEGER_PARAM(GL_DRAW_BUFFER6);
  mGL_PRINT_INTEGER_PARAM(GL_DRAW_FRAMEBUFFER_BINDING);
  mGL_PRINT_BOOL_PARAM(GL_EDGE_FLAG);
  mGL_PRINT_BOOL_PARAM(GL_EDGE_FLAG_ARRAY);
  mGL_PRINT_INTEGER_PARAM(GL_EDGE_FLAG_ARRAY_STRIDE);
  mGL_PRINT_INTEGER_PARAM(GL_ELEMENT_ARRAY_BUFFER_BINDING);
  mGL_PRINT_INTEGER_PARAM(GL_FEEDBACK_BUFFER_SIZE);
  mGL_PRINT_INTEGER_PARAM(GL_FEEDBACK_BUFFER_TYPE);
  mGL_PRINT_BOOL_PARAM(GL_FOG);
  mGL_PRINT_BOOL_PARAM(GL_FOG_COORD_ARRAY);
  mGL_PRINT_INTEGER_PARAM(GL_FOG_COORD_ARRAY_BUFFER_BINDING);
  mGL_PRINT_INTEGER_PARAM(GL_FOG_COORD_ARRAY_STRIDE);
  mGL_PRINT_INTEGER_PARAM(GL_FOG_COORD_ARRAY_TYPE);
  mGL_PRINT_INTEGER_PARAM(GL_FOG_COORD_SRC);
  mGL_PRINT_DOUBLE_VEC4_PARAM(GL_FOG_COLOR);
  mGL_PRINT_INTEGER_PARAM(GL_FOG_DENSITY);
  mGL_PRINT_INTEGER_PARAM(GL_FOG_END);
  mGL_PRINT_INTEGER_PARAM(GL_FOG_HINT);
  mGL_PRINT_INTEGER_PARAM(GL_FOG_INDEX);
  mGL_PRINT_INTEGER_PARAM(GL_FOG_INDEX);
  mGL_PRINT_INTEGER_PARAM(GL_FOG_MODE);
  mGL_PRINT_INTEGER_PARAM(GL_FOG_START);
  mGL_PRINT_INTEGER_PARAM(GL_FRAGMENT_SHADER_DERIVATIVE_HINT);
  mGL_PRINT_INTEGER_PARAM(GL_FRONT_FACE);
  mGL_PRINT_INTEGER_PARAM(GL_GENERATE_MIPMAP_HINT);
  mGL_PRINT_INTEGER_PARAM(GL_GREEN_BITS);
  mGL_PRINT_INTEGER_PARAM(GL_GREEN_SCALE);
  mGL_PRINT_BOOL_PARAM(GL_HISTOGRAM);
  mGL_PRINT_BOOL_PARAM(GL_INDEX_ARRAY);
  mGL_PRINT_INTEGER_PARAM(GL_INDEX_ARRAY_BUFFER_BINDING);
  mGL_PRINT_INTEGER_PARAM(GL_INDEX_ARRAY_STRIDE);
  mGL_PRINT_INTEGER_PARAM(GL_INDEX_ARRAY_TYPE);
  mGL_PRINT_INTEGER_PARAM(GL_INDEX_BITS);
  mGL_PRINT_INTEGER_PARAM(GL_INDEX_CLEAR_VALUE);
  mGL_PRINT_BOOL_PARAM(GL_INDEX_LOGIC_OP);
  mGL_PRINT_BOOL_PARAM(GL_INDEX_MODE);
  mGL_PRINT_INTEGER_PARAM(GL_INDEX_OFFSET);
  mGL_PRINT_INTEGER_PARAM(GL_INDEX_SHIFT);
  mGL_PRINT_INTEGER_PARAM(GL_INDEX_WRITEMASK);
  mGL_PRINT_INTEGER_PARAM(GL_LIGHT0);
  mGL_PRINT_INTEGER_PARAM(GL_LIGHT1);
  mGL_PRINT_INTEGER_PARAM(GL_LIGHT2);
  mGL_PRINT_INTEGER_PARAM(GL_LIGHT3);
  mGL_PRINT_INTEGER_PARAM(GL_LIGHT4);
  mGL_PRINT_INTEGER_PARAM(GL_LIGHT5);
  mGL_PRINT_INTEGER_PARAM(GL_LIGHT6);
  mGL_PRINT_INTEGER_PARAM(GL_LIGHT7);
  mGL_PRINT_BOOL_PARAM(GL_LIGHTING);
  mGL_PRINT_DOUBLE_VEC4_PARAM(GL_LIGHT_MODEL_AMBIENT);
  mGL_PRINT_INTEGER_PARAM(GL_LIGHT_MODEL_COLOR_CONTROL);
  mGL_PRINT_BOOL_PARAM(GL_LIGHT_MODEL_LOCAL_VIEWER);
  mGL_PRINT_BOOL_PARAM(GL_LIGHT_MODEL_TWO_SIDE);
  mGL_PRINT_BOOL_PARAM(GL_LINE_SMOOTH);
  mGL_PRINT_INTEGER_PARAM(GL_LINE_SMOOTH_HINT);
  mGL_PRINT_BOOL_PARAM(GL_LINE_STIPPLE);
  mGL_PRINT_INTEGER_PARAM(GL_LINE_STIPPLE_PATTERN);
  mGL_PRINT_INTEGER_PARAM(GL_LINE_STIPPLE_REPEAT);
  mGL_PRINT_INTEGER_PARAM(GL_LINE_WIDTH);
  mGL_PRINT_INTEGER_VEC4_PARAM(GL_LINE_WIDTH_RANGE);
  mGL_PRINT_INTEGER_PARAM(GL_LIST_BASE);
  mGL_PRINT_INTEGER_PARAM(GL_LIST_INDEX);
  mGL_PRINT_INTEGER_PARAM(GL_LIST_MODE);
  mGL_PRINT_INTEGER_PARAM(GL_LOGIC_OP_MODE);
  mGL_PRINT_BOOL_PARAM(GL_MAP1_COLOR_4);
  mGL_PRINT_INTEGER_VEC2_PARAM(GL_MAP1_GRID_DOMAIN);
  mGL_PRINT_INTEGER_PARAM(GL_MAP1_GRID_SEGMENTS);
  mGL_PRINT_BOOL_PARAM(GL_MAP1_INDEX);
  mGL_PRINT_BOOL_PARAM(GL_MAP1_NORMAL);
  mGL_PRINT_BOOL_PARAM(GL_MAP1_TEXTURE_COORD_1);
  mGL_PRINT_BOOL_PARAM(GL_MAP1_TEXTURE_COORD_2);
  mGL_PRINT_BOOL_PARAM(GL_MAP1_TEXTURE_COORD_3);
  mGL_PRINT_BOOL_PARAM(GL_MAP1_TEXTURE_COORD_4);
  mGL_PRINT_BOOL_PARAM(GL_MAP1_VERTEX_3);
  mGL_PRINT_BOOL_PARAM(GL_MAP1_VERTEX_4);
  mGL_PRINT_BOOL_PARAM(GL_MAP2_COLOR_4);
  mGL_PRINT_INTEGER_VEC4_PARAM(GL_MAP2_GRID_DOMAIN);
  mGL_PRINT_INTEGER_VEC2_PARAM(GL_MAP2_GRID_SEGMENTS);
  mGL_PRINT_BOOL_PARAM(GL_MAP2_INDEX);
  mGL_PRINT_BOOL_PARAM(GL_MAP2_NORMAL);
  mGL_PRINT_BOOL_PARAM(GL_MAP2_TEXTURE_COORD_1);
  mGL_PRINT_BOOL_PARAM(GL_MAP2_TEXTURE_COORD_2);
  mGL_PRINT_BOOL_PARAM(GL_MAP2_TEXTURE_COORD_3);
  mGL_PRINT_BOOL_PARAM(GL_MAP2_TEXTURE_COORD_4);
  mGL_PRINT_BOOL_PARAM(GL_MAP2_VERTEX_3);
  mGL_PRINT_BOOL_PARAM(GL_MAP2_VERTEX_4);
  mGL_PRINT_BOOL_PARAM(GL_MAP2_COLOR_4);
  mGL_PRINT_BOOL_PARAM(GL_MAP_COLOR);
  mGL_PRINT_BOOL_PARAM(GL_MAP_STENCIL);
  mGL_PRINT_BOOL_PARAM(GL_MINMAX);
  mGL_PRINT_BOOL_PARAM(GL_NORMAL_ARRAY);
  mGL_PRINT_INTEGER_PARAM(GL_NORMAL_ARRAY_BUFFER_BINDING);
  mGL_PRINT_INTEGER_PARAM(GL_NORMAL_ARRAY_STRIDE);
  mGL_PRINT_INTEGER_PARAM(GL_NORMAL_ARRAY_TYPE);
  mGL_PRINT_BOOL_PARAM(GL_NORMALIZE);
  mGL_PRINT_INTEGER_PARAM(GL_PACK_ALIGNMENT);
  mGL_PRINT_INTEGER_PARAM(GL_PACK_IMAGE_HEIGHT);
  mGL_PRINT_BOOL_PARAM(GL_PACK_LSB_FIRST);
  mGL_PRINT_INTEGER_PARAM(GL_PACK_ROW_LENGTH);
  mGL_PRINT_INTEGER_PARAM(GL_PACK_SKIP_IMAGES);
  mGL_PRINT_INTEGER_PARAM(GL_PACK_SKIP_PIXELS);
  mGL_PRINT_INTEGER_PARAM(GL_PACK_SKIP_ROWS);
  mGL_PRINT_BOOL_PARAM(GL_PACK_SWAP_BYTES);
  mGL_PRINT_INTEGER_PARAM(GL_PERSPECTIVE_CORRECTION_HINT);
  mGL_PRINT_INTEGER_PARAM(GL_PIXEL_PACK_BUFFER_BINDING);
  mGL_PRINT_INTEGER_PARAM(GL_PIXEL_UNPACK_BUFFER_BINDING);
  mGL_PRINT_DOUBLE_VEC3_PARAM(GL_POINT_DISTANCE_ATTENUATION);
  mGL_PRINT_INTEGER_PARAM(GL_POINT_FADE_THRESHOLD_SIZE);
  mGL_PRINT_INTEGER_PARAM(GL_POINT_SIZE);
  mGL_PRINT_BOOL_PARAM(GL_POINT_SMOOTH);
  mGL_PRINT_INTEGER_PARAM(GL_POINT_SMOOTH_HINT);
  mGL_PRINT_BOOL_PARAM(GL_POINT_SPRITE);
  mGL_PRINT_INTEGER_PARAM(GL_POLYGON_MODE);
  mGL_PRINT_INTEGER_PARAM(GL_POLYGON_OFFSET_FACTOR);
  mGL_PRINT_INTEGER_PARAM(GL_POLYGON_OFFSET_UNITS);
  mGL_PRINT_BOOL_PARAM(GL_POLYGON_OFFSET_FILL);
  mGL_PRINT_BOOL_PARAM(GL_POLYGON_OFFSET_LINE);
  mGL_PRINT_BOOL_PARAM(GL_POLYGON_OFFSET_POINT);
  mGL_PRINT_BOOL_PARAM(GL_POLYGON_SMOOTH);
  mGL_PRINT_INTEGER_PARAM(GL_POLYGON_SMOOTH_HINT);
  mGL_PRINT_BOOL_PARAM(GL_POLYGON_STIPPLE);
  mGL_PRINT_INTEGER_PARAM(GL_READ_FRAMEBUFFER_BINDING);
  mGL_PRINT_INTEGER_PARAM(GL_READ_BUFFER);
  mGL_PRINT_INTEGER_PARAM(GL_RED_BITS);
  mGL_PRINT_INTEGER_PARAM(GL_RED_SCALE);
  mGL_PRINT_INTEGER_PARAM(GL_RENDER_MODE);
  mGL_PRINT_INTEGER_PARAM(GL_RENDERBUFFER_BINDING);
  mGL_PRINT_BOOL_PARAM(GL_RESCALE_NORMAL);
  mGL_PRINT_BOOL_PARAM(GL_RGBA_MODE);
  mGL_PRINT_INTEGER_PARAM(GL_SAMPLE_BUFFERS);
  mGL_PRINT_DOUBLE_PARAM(GL_SAMPLE_COVERAGE_VALUE);
  mGL_PRINT_BOOL_PARAM(GL_SAMPLE_COVERAGE_INVERT);
  mGL_PRINT_INTEGER_PARAM(GL_SAMPLES);
  mGL_PRINT_INTEGER_PARAM(GL_SAMPLE_BUFFERS);
  mGL_PRINT_DOUBLE_PARAM(GL_SAMPLE_COVERAGE_VALUE);
  mGL_PRINT_BOOL_PARAM(GL_SAMPLE_COVERAGE_INVERT);
  mGL_PRINT_INTEGER_VEC4_PARAM(GL_SCISSOR_BOX);
  mGL_PRINT_BOOL_PARAM(GL_SCISSOR_TEST);
  mGL_PRINT_BOOL_PARAM(GL_SECONDARY_COLOR_ARRAY);
  mGL_PRINT_INTEGER_PARAM(GL_SECONDARY_COLOR_ARRAY_BUFFER_BINDING);
  mGL_PRINT_INTEGER_PARAM(GL_SECONDARY_COLOR_ARRAY_SIZE);
  mGL_PRINT_INTEGER_PARAM(GL_SECONDARY_COLOR_ARRAY_STRIDE);
  mGL_PRINT_INTEGER_PARAM(GL_SECONDARY_COLOR_ARRAY_TYPE);
  mGL_PRINT_INTEGER_PARAM(GL_SELECTION_BUFFER_SIZE);
  mGL_PRINT_BOOL_PARAM(GL_SEPARABLE_2D);
  mGL_PRINT_INTEGER_PARAM(GL_SHADE_MODEL);
  mGL_PRINT_BOOL_PARAM(GL_SHADER_COMPILER);
  mGL_PRINT_INTEGER_PARAM(GL_SHADER_STORAGE_BUFFER_BINDING);
  mGL_PRINT_INTEGER_PARAM(GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT);
  mGL_PRINT_INTEGER_PARAM(GL_STENCIL_BACK_FAIL);
  mGL_PRINT_INTEGER_PARAM(GL_STENCIL_BACK_FUNC);
  mGL_PRINT_INTEGER_PARAM(GL_STENCIL_BACK_PASS_DEPTH_FAIL);
  mGL_PRINT_INTEGER_PARAM(GL_STENCIL_BACK_PASS_DEPTH_PASS);
  mGL_PRINT_INTEGER_PARAM(GL_STENCIL_BACK_REF);
  mGL_PRINT_INTEGER_PARAM(GL_STENCIL_BACK_VALUE_MASK);
  mGL_PRINT_INTEGER_PARAM(GL_STENCIL_BACK_WRITEMASK);
  mGL_PRINT_INTEGER_PARAM(GL_STENCIL_BITS);
  mGL_PRINT_INTEGER_PARAM(GL_STENCIL_CLEAR_VALUE);
  mGL_PRINT_INTEGER_PARAM(GL_STENCIL_FAIL);
  mGL_PRINT_INTEGER_PARAM(GL_STENCIL_FUNC);
  mGL_PRINT_INTEGER_PARAM(GL_STENCIL_PASS_DEPTH_FAIL);
  mGL_PRINT_INTEGER_PARAM(GL_STENCIL_PASS_DEPTH_PASS);
  mGL_PRINT_INTEGER_PARAM(GL_STENCIL_REF);
  mGL_PRINT_BOOL_PARAM(GL_STENCIL_TEST);
  mGL_PRINT_INTEGER_PARAM(GL_STENCIL_VALUE_MASK);
  mGL_PRINT_INTEGER_PARAM(GL_STENCIL_WRITEMASK);
  mGL_PRINT_INTEGER_PARAM(GL_STEREO);
  mGL_PRINT_INTEGER_PARAM(GL_SUBPIXEL_BITS);
  mGL_PRINT_BOOL_PARAM(GL_TEXTURE_1D);
  mGL_PRINT_INTEGER_PARAM(GL_TEXTURE_BINDING_1D);
  mGL_PRINT_BOOL_PARAM(GL_TEXTURE_2D);
  mGL_PRINT_INTEGER_PARAM(GL_TEXTURE_BINDING_2D);
  mGL_PRINT_INTEGER_PARAM(GL_TEXTURE_BINDING_2D_MULTISAMPLE);
  mGL_PRINT_BOOL_PARAM(GL_TEXTURE_3D);
  mGL_PRINT_INTEGER_PARAM(GL_TEXTURE_BINDING_3D);
  mGL_PRINT_INTEGER_PARAM(GL_TEXTURE_BINDING_BUFFER);
  mGL_PRINT_INTEGER_PARAM(GL_TEXTURE_BINDING_CUBE_MAP);
  mGL_PRINT_INTEGER_PARAM(GL_TEXTURE_COMPRESSION_HINT);
  mGL_PRINT_BOOL_PARAM(GL_TEXTURE_COORD_ARRAY);
  mGL_PRINT_INTEGER_PARAM(GL_TEXTURE_COORD_ARRAY_BUFFER_BINDING);
  mGL_PRINT_INTEGER_PARAM(GL_TEXTURE_COORD_ARRAY_SIZE);
  mGL_PRINT_INTEGER_PARAM(GL_TEXTURE_COORD_ARRAY_STRIDE);
  mGL_PRINT_INTEGER_PARAM(GL_TEXTURE_COORD_ARRAY_TYPE);
  mGL_PRINT_BOOL_PARAM(GL_TEXTURE_CUBE_MAP);
  mGL_PRINT_INTEGER_PARAM(GL_UNPACK_ALIGNMENT);
  mGL_PRINT_INTEGER_PARAM(GL_UNPACK_IMAGE_HEIGHT);
  mGL_PRINT_BOOL_PARAM(GL_UNPACK_LSB_FIRST);
  mGL_PRINT_INTEGER_PARAM(GL_UNPACK_ROW_LENGTH);
  mGL_PRINT_INTEGER_PARAM(GL_UNPACK_SKIP_IMAGES);
  mGL_PRINT_INTEGER_PARAM(GL_UNPACK_SKIP_PIXELS);
  mGL_PRINT_INTEGER_PARAM(GL_UNPACK_SKIP_ROWS);
  mGL_PRINT_BOOL_PARAM(GL_UNPACK_SWAP_BYTES);
  mGL_PRINT_BOOL_PARAM(GL_VERTEX_ARRAY);
  mGL_PRINT_INTEGER_PARAM(GL_VERTEX_ARRAY_BUFFER_BINDING);
  mGL_PRINT_INTEGER_PARAM(GL_VERTEX_ARRAY_SIZE);
  mGL_PRINT_INTEGER_PARAM(GL_VERTEX_ARRAY_STRIDE);
  mGL_PRINT_INTEGER_PARAM(GL_VERTEX_ARRAY_TYPE);
  mGL_PRINT_INTEGER_PARAM(GL_VERTEX_PROGRAM_POINT_SIZE);
  mGL_PRINT_INTEGER_PARAM(GL_VERTEX_PROGRAM_TWO_SIDE);
  mGL_PRINT_INTEGER_VEC4_PARAM(GL_VIEWPORT);
  mGL_PRINT_INTEGER_PARAM(GL_ZOOM_X);
  mGL_PRINT_INTEGER_PARAM(GL_ZOOM_Y);
  
  mGL_PRINT_FRAMEBUFFER_ATTACHMENT_PARAM(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE);
  mGL_PRINT_FRAMEBUFFER_ATTACHMENT_PARAM(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME);
  mGL_PRINT_FRAMEBUFFER_ATTACHMENT_PARAM(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER_ATTACHMENT_RED_SIZE);
  mGL_PRINT_FRAMEBUFFER_ATTACHMENT_PARAM(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER_ATTACHMENT_GREEN_SIZE);
  mGL_PRINT_FRAMEBUFFER_ATTACHMENT_PARAM(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER_ATTACHMENT_BLUE_SIZE);
  mGL_PRINT_FRAMEBUFFER_ATTACHMENT_PARAM(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE);

  // Clear glError.
  while (glGetError() != GL_NO_ERROR)
    ;

  if (!onlyUpdateValues)
    mPRINT_DEBUG("================================================\nEND OF GL STATE\n");

#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mRenderParams_SetOnErrorDebugCallback, const std::function<mResult(const GLenum source, const GLenum type, const GLuint id, const GLenum severity, const GLsizei lenght, const char *msg)> &callback)
{
  mFUNCTION_SETUP();

  mERROR_IF(callback == nullptr, mR_InvalidParameter);

  static std::function<mResult(const GLenum source, const GLenum type, const GLuint id, const GLenum severity, const GLsizei lenght, const char *msg)> _Callback;
  
  struct _internal
  {
    static void GLAPIENTRY _ErrorMessageCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar *message, const void *)
    {
      _Callback(source, type, id, severity, length, message);
    }
  };

  // During init, enable debug output
  glEnable(GL_DEBUG_OUTPUT);
  glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
  _Callback = callback;

  glDebugMessageCallback(_internal::_ErrorMessageCallback, nullptr);
  glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);

  mRETURN_SUCCESS();
}
