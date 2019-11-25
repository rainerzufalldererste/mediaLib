#ifndef mBezierCurve_h__
#define mBezierCurve_h__

#include "mediaLib.h"
#include "mQueue.h"

template <typename T>
struct mCubicBezierCurve
{
  mVec2t<T> startPoint, endPoint, controlPoint1, controlPoint2;

  mCubicBezierCurve(const mVec2t<T> &startPoint, const mVec2t<T> &endPoint, const mVec2t<T> &controlPoint1, const mVec2t<T> &controlPoint2) :
    startPoint(startPoint),
    endPoint(endPoint),
    controlPoint1(controlPoint1),
    controlPoint2(controlPoint2)
  { }
};

template <typename T, typename U>
inline mVec2t<T> mInterpolate(const mCubicBezierCurve<T> &curve, const U position)
{
  const U p2 = position * position;
  const U p3 = p2 * position;
  const U i = U(1) - position;
  const U i2 = i * i;
  const U i3 = i2 * i;

  return i3 * curve.startPoint + (U(3) * i2 * position) * curve.controlPoint1 + (U(3) * i * p2) * curve.controlPoint2 + p3 * curve.endPoint;
}

template <typename T>
inline mCubicBezierCurve<T> mCubicBezierCurve_GetFromPoints(const mVec2t<T> &pointA, const mVec2t<T> &pointB, const mVec2t<T> &pointC, const mVec2t<T> &pointD)
{
  // Differences.
  const mVec2t<T> bma = pointB - pointA;
  const mVec2t<T> cmb = pointC - pointB;
  const mVec2t<T> dmc = pointD - pointC;

  const T c1 = mSqrt(bma.x * bma.x + bma.y * bma.y);
  const T c2 = mSqrt(cmb.x * cmb.x + cmb.y * cmb.y);
  const T c3 = mSqrt(dmc.x * dmc.x + dmc.y * dmc.y);
  
  // Guess best value for `t` at `pointB` and `pointC`.
  const T t1 = c1 / (c1 + c2 + c3);
  const T t2 = (c1 + c2) / (c1 + c2 + c3);

  // Inverse.
  const T i1 = T(1) - t1;
  const T i2 = T(1) - t2;

  // Squared.
  const T t1_2 = t1 * t1;
  const T t2_2 = t2 * t2;
  const T i1_2 = i1 * i1;
  const T i2_2 = i2 * i2;

  // Cubed.
  const T t1_3 = t1_2 * t1;
  const T t2_3 = t2_2 * t2;
  const T i1_3 = i1_2 * i1;
  const T i2_3 = i2_2 * i2;

  // Factors p, q.
  const T p1 = t1 * i1_2 * (T)3;
  const T p2 = t2 * i2_2 * (T)3;
  const T q1 = i1 * t1_2 * (T)3;
  const T q2 = i2 * t2_2 * (T)3;

  // Solve for Control Points 1 and 2.
  const mVec2f c = pointB - (pointA * i1_3) - (pointD * t1_3);
  const mVec2f f = pointC - (pointA * i2_3) - (pointD * t2_3);

  const float_t div2 = (q1 - p1 * q2 / p2);

  const mVec2f cp2 = (c - mVec2f(p1 / p2) * f) / div2;
  const mVec2f cp1 = (c - (q1 * cp2)) / p1;

  return mCubicBezierCurve<T>(pointA, pointD, cp1, cp2);
}

#endif // mBezierCurve_h__
