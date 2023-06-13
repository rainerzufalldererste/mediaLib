#ifndef mRenderParams_h__
#define mRenderParams_h__

#include "mediaLib.h"
#include "mQueue.h"

#define mRENDERER_OPENGL

#if defined(mRENDERER_OPENGL)
#include "GL/glew.h"
#include "SDL.h"
#include "SDL_opengl.h"
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

#if defined(mRENDERER_OPENGL)
extern GLenum mRenderParams_GLError;

#define mOPENGL_ERROR_CHECK() \
  do { \
    mRenderParams_GLError = glGetError(); \
    if(mRenderParams_GLError != GL_NO_ERROR) { \
      mDebugOut("Rendering Error in '" __FUNCTION__ "': GLError Code %" PRIi32 " (%s) (File '" __FILE__ "'; Line %" PRIi32 ")\n", mRenderParams_GLError, gluErrorString(mRenderParams_GLError), __LINE__); \
      mERROR_IF(mRenderParams_GLError != GL_NO_ERROR, mR_RenderingError); \
    } \
  } \
  while (0);

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

enum mRenderParams_UploadState
{
  mRP_US_NotInitialized,
  mRP_US_NotUploaded,
  mRP_US_Uploading,
  mRP_US_Ready
};

enum mRenderParams_VertexRenderMode
{
  mRP_VRM_Points,
  mRP_VRM_LineList,
  mRP_VRM_LineStrip,
  mRP_VRM_LineLoop,
  mRP_VRM_TriangleList,
  mRP_VRM_TriangleStrip,
  mRP_VRM_TriangleFan,
  mRP_VRM_QuadList,
  mRP_VRM_QuadStrip,
  mRP_VRM_Polygon,
};

#endif // mRenderParams_h__
