#ifndef mLineRenderer_h__
#define mLineRenderer_h__

#include "mBezierCurve.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "212C9ISIqsJaWMoNqMFGBf5mURpIsDVXxLOHImS3gU4WiNIQXG6grQCVesh3q7Z4q/uhGu9OMttTi7aT"
#endif

struct mLineRenderer;

struct mLineRenderer_Attribute
{
  mVec4f colour;
  float_t thickness;

  inline mLineRenderer_Attribute() = default;

  inline mLineRenderer_Attribute(const mVec4f colour, const float_t thickness) :
    colour(colour),
    thickness(thickness)
  { }
};

struct mLineRenderer_Point
{
  mVec2f position;
  mVec4f colour;
  float_t thickness;

  inline mLineRenderer_Point() = default;

  inline mLineRenderer_Point(const mVec2f position, const mVec4f colour, const float_t thickness) :
    position(position),
    colour(colour),
    thickness(thickness)
  { }

  inline mLineRenderer_Point(const mVec2f position, const mLineRenderer_Attribute attribute) :
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

// `start`, `end` represent the curve appearance at t = 0 and t = 1.
// If the specified segment (`startT`, `endT`) is different, the given values will be interpolated accordingly.
mFUNCTION(mLineRenderer_DrawCubicBezierLineSegment, mPtr<mLineRenderer> &lineRenderer, const mCubicBezierCurve<float_t> &curve, const mLineRenderer_Attribute &start, const mLineRenderer_Attribute &end, const float_t startT, const float_t endT);

mFUNCTION(mLineRenderer_End, mPtr<mLineRenderer> &lineRenderer);

#endif // mLineRenderer_h__
