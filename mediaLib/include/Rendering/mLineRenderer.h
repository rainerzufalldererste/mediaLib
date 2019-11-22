#ifndef mLineRenderer_h__
#define mLineRenderer_h__

#include "mBezierCurve.h"

struct mLineRenderer;

struct mLineRenderer_Attribute
{
  mVec4f colour;
  float_t thickness;

  mLineRenderer_Attribute()
  { }

  mLineRenderer_Attribute(const mVec4f colour, const float_t thickness) :
    colour(colour),
    thickness(thickness)
  { }
};

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

  mLineRenderer_Point(const mVec2f position, const mLineRenderer_Attribute attribute) :
    position(position),
    colour(attribute.colour),
    thickness(attribute.thickness)
  { }
};

mFUNCTION(mLineRenderer_Create, OUT mPtr<mLineRenderer> *pLineRenderer, IN mAllocator *pAllocator, const float_t subdivisionFactor = mTWOPIf / 150.0f);
mFUNCTION(mLineRenderer_Destroy, IN_OUT mPtr<mLineRenderer> *pLineRenderer);

mFUNCTION(mLineRenderer_Begin, mPtr<mLineRenderer> &lineRenderer);
mFUNCTION(mLineRenderer_DrawStraightLine, mPtr<mLineRenderer> &lineRenderer, const mLineRenderer_Point &startPoint, const mLineRenderer_Point &endPoint);
mFUNCTION(mLineRenderer_DrawStraightArrow, mPtr<mLineRenderer> &lineRenderer, const mLineRenderer_Point &startPoint, const mLineRenderer_Point &endPoint, const float_t arrowSize);
mFUNCTION(mLineRenderer_DrawCubicBezierLine, mPtr<mLineRenderer> &lineRenderer, const mCubicBezierCurve<float_t> &curve, const mLineRenderer_Attribute &start, const mLineRenderer_Attribute &end);
mFUNCTION(mLineRenderer_DrawCubicBezierArrow, mPtr<mLineRenderer> &lineRenderer, const mCubicBezierCurve<float_t> &curve, const mLineRenderer_Attribute &start, const mLineRenderer_Attribute &end, const float_t arrowSize);
mFUNCTION(mLineRenderer_DrawCubicBezierLineSegment, mPtr<mLineRenderer> &lineRenderer, const mCubicBezierCurve<float_t> &curve, const mLineRenderer_Attribute &start, const mLineRenderer_Attribute &end, const float_t startT, const float_t endT);
mFUNCTION(mLineRenderer_End, mPtr<mLineRenderer> &lineRenderer);

#endif // mLineRenderer_h__
