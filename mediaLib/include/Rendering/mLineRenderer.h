#ifndef mLineRenderer_h__
#define mLineRenderer_h__

#include "mRenderParams.h"
#include "mRenderDataBuffer.h"
#include "mBinaryChunk.h"

extern const char mLineRenderer_PositionAttribute[10];
extern const char mLineRenderer_ColourAttribute[8];

struct mLineRenderer_Point
{
  mVec2f position;
  mVec4f colour;
  float_t thickness;

  mLineRenderer_Point()
  { }

  mLineRenderer_Point(const mVec2f position, const mVec4f colour, const float_t thickness) :
    position(position),
    colour(colour),
    thickness(thickness)
  { }
};

struct mLineRenderer
{
  float_t subdivisionFactor;
  mPtr<mShader> shader;
  mRenderDataBuffer<mRDB_FloatAttribute<2, mLineRenderer_PositionAttribute>, mRDB_FloatAttribute<4, mLineRenderer_ColourAttribute>> renderDataBuffer;
  mPtr<mBinaryChunk> renderData;
  bool started;
};

mFUNCTION(mLineRenderer_Create, OUT mPtr<mLineRenderer> *pLineRenderer, IN mAllocator *pAllocator, const float_t subdivisionFactor = mTWOPIf / 150.0f);
mFUNCTION(mLineRenderer_Destroy, IN_OUT mPtr<mLineRenderer> *pLineRenderer);

mFUNCTION(mLineRenderer_Begin, mPtr<mLineRenderer> &lineRenderer);
mFUNCTION(mLineRenderer_DrawStraightLine, mPtr<mLineRenderer> &lineRenderer, const mLineRenderer_Point &startPoint, const mLineRenderer_Point &endPoint);
mFUNCTION(mLineRenderer_DrawStraightArrow, mPtr<mLineRenderer> &lineRenderer, const mLineRenderer_Point &startPoint, const mLineRenderer_Point &endPoint, const float_t arrowSize);
mFUNCTION(mLineRenderer_End, mPtr<mLineRenderer> &lineRenderer);

#endif // mLineRenderer_h__
