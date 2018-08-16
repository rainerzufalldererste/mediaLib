// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef mRenderParams_h__
#define mRenderParams_h__

#include "default.h"
#include "mQueue.h"

#define mRENDERER_OPENGL

#if defined(mRENDERER_OPENGL)
#include "GL/glew.h"
#include "SDL.h"
#include "SDL_opengl.h"
#endif

extern mVec2s mRendererParams_CurrentRenderSize;

typedef size_t mRenderContextId;
extern mRenderContextId mRenderParams_CurrentRenderResolution;

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

#define mOPENGL_ERROR_CHECK() mERROR_IF((mRenderParams_GLError = glGetError()) != GL_NO_ERROR, mR_RenderingError)

#define mGL_ERROR_CHECK() mOPENGL_ERROR_CHECK()

#else
#define mGL_ERROR_CHECK()
#endif

struct mHardwareWindow;

mFUNCTION(mRenderParams_CreateRenderContext, OUT mRenderContextId *pRenderContextId, mPtr<mHardwareWindow> &window);
mFUNCTION(mRenderParams_ActivateRenderContext, mPtr<mHardwareWindow> &window, const mRenderContextId renderContextId);
mFUNCTION(mRenderParams_DestroyRenderContext, IN_OUT mRenderContextId *pRenderContextId);
mFUNCTION(mRenderParams_SetVsync, const bool vsync);
mFUNCTION(mRenderParams_SetDoubleBuffering, const bool doubleBuffering);
mFUNCTION(mRenderParams_SetMultisampling, const size_t count);

#endif // mRenderParams_h__
