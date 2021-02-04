#ifndef mPointRenderer_h__
#define mPointRenderer_h__

#include "mRenderParams.h"
#include "mShader.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "4GwyxlJ2GfxXn5+dtpll92+2Fz9TB/Q6t5BMZy3t1QpS0F/qJ7EZSCC5cRQjnaHei42BzWx+SBHb8IGJ"
#endif

struct mPointRenderer
{
  mShader shader;

#if defined(mRENDERER_OPENGL)
  GLuint vbo;
#endif
};

struct mColouredPoint
{
  mVec2f position;
  mVec4f colour;
  float_t size;

  mColouredPoint();
  mColouredPoint(const mVec2f position, const mVec4f colour, const float_t size);
};

enum mColouredPointMode
{
  mCPM_Square,
  mCPM_Circle,
};

mFUNCTION(mPointRenderer_Create, OUT mPtr<mPointRenderer> *pPointRenderer, IN mAllocator *pAllocator);
mFUNCTION(mPointRenderer_Destroy, IN_OUT mPtr<mPointRenderer> *pPointRenderer);

mFUNCTION(mPointRenderer_Begin, mPtr<mPointRenderer> &pointRenderer, const mColouredPointMode mode = mCPM_Circle);
mFUNCTION(mPointRenderer_End, mPtr<mPointRenderer> &pointRenderer);

mFUNCTION(mPointRenderer_DrawPoint, mPtr<mPointRenderer> &pointRenderer, const mColouredPoint &point);

#endif // mPointRenderer_h__
