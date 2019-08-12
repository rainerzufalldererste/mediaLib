#ifndef mRenderParams_h__
#define mRenderParams_h__

#include "mediaLib.h"
#include "mQueue.h"

#define mRENDERER_OPENGL

#if defined(mRENDERER_OPENGL)
#include "GL/glew.h"
#define DECLSPEC
#include "SDL.h"
#include "SDL_opengl.h"
#undef DECLSPEC
#endif

extern mVec2s mRenderParams_CurrentRenderResolution;
extern mVec2f mRenderParams_CurrentRenderResolutionF;

typedef size_t mRenderContextId;
extern mRenderContextId mRenderParams_CurrentRenderContext;

struct mRenderContext
{
#if defined(mRENDERER_OPENGL)
  SDL_GLContext glContext;
#endif

  size_t referenceCount;
};

extern mRenderContext *mRenderParams_pRenderContexts;
extern size_t mRenderParams_RenderContextCount;
extern size_t mRenderParams_InitializedRenderContextCount;

#if defined(mRENDERER_OPENGL)
extern GLenum mRenderParams_GLError;

#define mOPENGL_ERROR_CHECK() \
  do { \
    if (mRenderParams_InitializedRenderContextCount > 0) { \
      mRenderParams_GLError = glGetError(); \
      if (mRenderParams_GLError != GL_NO_ERROR) { \
        mPRINT_ERROR("Rendering Error in '%s': GLError Code %" PRIi32 " (%s) (File '" __FILE__ "'; Line %" PRIi32 ")\n", mRESULT_PRINT_FUNCTION_TITLE, mRenderParams_GLError, gluErrorString(mRenderParams_GLError), __LINE__); \
        mRenderParams_PrintRenderState(false); \
        mERROR_IF(mRenderParams_GLError != GL_NO_ERROR, mR_RenderingError); \
      } \
    } \
  } \
  while (0)

#define mGL_ERROR_CHECK() mOPENGL_ERROR_CHECK()

#if defined(_DEBUG)
#define mGL_DEBUG_ERROR_CHECK() mOPENGL_ERROR_CHECK()
#else
#define mGL_DEBUG_ERROR_CHECK()
#endif

#else
#define mGL_ERROR_CHECK()
#define mGL_DEBUG_ERROR_CHECK()
#endif

struct mHardwareWindow;

mFUNCTION(mRenderParams_InitializeToDefault);

mFUNCTION(mRenderParams_CreateRenderContext, OUT mRenderContextId *pRenderContextId, mPtr<mHardwareWindow> &window);
mFUNCTION(mRenderParams_ActivateRenderContext, mPtr<mHardwareWindow> &window, const mRenderContextId renderContextId);
mFUNCTION(mRenderParams_DestroyRenderContext, IN_OUT mRenderContextId *pRenderContextId);

#if defined(mRENDERER_OPENGL)
mFUNCTION(mRenderParams_GetRenderContext, mPtr<mHardwareWindow> &window, OUT SDL_GLContext *pContext);
#endif

mFUNCTION(mRenderParams_SetCurrentRenderResolution, const mVec2s &resolution);

mFUNCTION(mRenderParams_SetVsync, const bool vsync);
mFUNCTION(mRenderParams_SetDoubleBuffering, const bool doubleBuffering);
mFUNCTION(mRenderParams_SetMultisampling, const size_t count);
mFUNCTION(mRenderParams_SupportsStereo3d, OUT bool *pSupportsStereo3d);
mFUNCTION(mRenderParams_SetStereo3d, const bool enabled);
mFUNCTION(mRenderParams_ShowCursor, const bool isVisible);

enum mRenderParams_StereoRenderBuffer
{
  mRP_SRB_LeftEye,
  mRP_SRB_RightEye,
};

mFUNCTION(mRenderParams_SetStereo3dBuffer, const mRenderParams_StereoRenderBuffer buffer);

mFUNCTION(mRenderParams_ClearTargetColour, const mVector &colour = mVector(0, 0, 0, 1));
mFUNCTION(mRenderParams_ClearDepth);
mFUNCTION(mRenderParams_ClearTargetDepthAndColour, const mVector &colour = mVector(0, 0, 0, 1));

mFUNCTION(mRenderParams_SetBlendingEnabled, const bool enabled = true);
mFUNCTION(mRenderParams_SetDepthTestEnabled, const bool enabled = true);
mFUNCTION(mRenderParams_SetDepthMaskEnabled, const bool enabled = true);
mFUNCTION(mRenderParams_SetScissorTestEnabled, const bool enabled = true);

mFUNCTION(mRenderParams_GetMaxDepthPrecisionBits, OUT size_t *pDepth);
mFUNCTION(mRenderParams_SetDepthPrecisionBits, const size_t bits);

enum mRenderParam_BlendFunc
{
  mRP_BF_NoAlpha,
  mRP_BF_Additive,
  mRP_BF_AlphaBlend,
  mRP_BF_Premultiplied,
  mRP_BF_Override,
  mRP_BF_AlphaMask
};

mFUNCTION(mRenderParams_SetBlendFunc, const mRenderParam_BlendFunc blendFunc);

enum mRenderParams_DepthFunc
{
  mRP_DF_Less,
  mRP_DF_Greater,
  mRP_DF_Equal,
  mRP_DF_LessOrEqual,
  mRP_DF_GreaterOrEqual,
  mRP_DF_NotEqual,
  mRP_DF_Always,
  mRP_DF_NoDepth,
};

mFUNCTION(mRenderParams_SetDepthFunc, const mRenderParams_DepthFunc depthFunc);

#if defined(mRENDERER_OPENGL) && defined(_WIN32)
mFUNCTION(mRenderParams_GetCurrentGLContext_HGLRC, HGLRC *pGLContext);
mFUNCTION(mRenderParams_GetCurrentGLContext_HDC, HDC *pGLDrawable);
mFUNCTION(mRenderParams_GetCurrentGLContext_HWND, HWND *pGLWindow);
#endif

enum mRenderParams_UploadState
{
  mRP_US_NotInitialized,
  mRP_US_NotUploaded,
  mRP_US_Uploading,
  mRP_US_Ready
};

enum mRenderParams_VertexRenderMode
{
  mRP_VRM_Points = GL_POINTS,
  mRP_VRM_LineList = GL_LINES,
  mRP_VRM_LineStrip = GL_LINE_STRIP,
  mRP_VRM_LineLoop = GL_LINE_LOOP,
  mRP_VRM_TriangleList = GL_TRIANGLES,
  mRP_VRM_TriangleStrip = GL_TRIANGLE_STRIP,
  mRP_VRM_TriangleFan = GL_TRIANGLE_FAN,
  mRP_VRM_QuadList = GL_QUADS,
  mRP_VRM_QuadStrip = GL_QUAD_STRIP,
  mRP_VRM_Polygon = GL_POLYGON,
};

enum mRenderParams_TextureWrapMode
{
  mRP_TWM_ClampToEdge = GL_CLAMP_TO_EDGE,
  mRP_TWM_ClampToBorder = GL_CLAMP_TO_BORDER,
  mRP_TWM_MirroredRepeat = GL_MIRRORED_REPEAT,
  mRP_TWM_Repeat = GL_REPEAT,
  mRP_TWM_MirroredClampToEdge = GL_MIRROR_CLAMP_TO_EDGE,
};

enum mRenderParams_TextureMagnificationFilteringMode
{
  mRP_TMagFM_NearestNeighbor = GL_NEAREST,
  mRP_TMagFM_BilinearInterpolation = GL_LINEAR,
};

enum mRenderParams_TextureMinificationFilteringMode
{
  mRP_TMinFM_NearestNeighbor = GL_NEAREST,
  mRP_TMinFM_BilinearInterpolation = GL_LINEAR,
};

struct mTexture2DParams
{
  mRenderParams_TextureWrapMode wrapModeX = mRP_TWM_ClampToEdge, wrapModeY = mRP_TWM_ClampToEdge;
  mRenderParams_TextureMagnificationFilteringMode magFilter = mRP_TMagFM_BilinearInterpolation;
  mRenderParams_TextureMinificationFilteringMode minFilter = mRP_TMinFM_BilinearInterpolation;
};

mFUNCTION(mTexture2DParams_ApplyToBoundTexture, const mTexture2DParams &params, const bool isMultisampleTexture = false);

struct mTexture3DParams
{
  mRenderParams_TextureWrapMode wrapModeX = mRP_TWM_ClampToEdge, wrapModeY = mRP_TWM_ClampToEdge, wrapModeZ = mRP_TWM_ClampToEdge;
  mRenderParams_TextureMagnificationFilteringMode magFilter = mRP_TMagFM_BilinearInterpolation;
  mRenderParams_TextureMinificationFilteringMode minFilter = mRP_TMinFM_BilinearInterpolation;
};

mFUNCTION(mTexture3DParams_ApplyToBoundTexture, const mTexture3DParams &params, const bool isMultisampleTexture = false);

mFUNCTION(mRenderParams_PrintRenderState, const bool onlyNewValues = false);

#endif // mRenderParams_h__
