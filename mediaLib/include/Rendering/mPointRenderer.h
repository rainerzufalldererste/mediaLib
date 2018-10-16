#ifndef mPointRenderer_h__
#define mPointRenderer_h__

#include "mRenderParams.h"
#include "mShader.h"

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
