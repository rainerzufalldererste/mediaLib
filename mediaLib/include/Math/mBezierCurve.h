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

// Solve linear equation: a * x + b * y = c and d * x + e * y = f for x and y.
// Returns mVec2t<T>(x, y).
template <typename T>
inline mVec2t<T> mSolveXY(const T a, const T b, const T c, const T d, const T e, const T f)
{
  const T y = (c - a / d * f) / (b - a * e / d);
  const T x = (c - (b * y)) / a;

  return mVec2t<T>(x, y);
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

  // Solve for Control Point X and Y coordinates.
  // TODO: Remove duplicate calculations.
  const mVec2t<T> x12 = mSolveXY(p1, q1, pointB.x - (pointA.x * i1_3) - (pointD.x * t1_3), p2, q2, pointC.x - (pointA.x * i2_3) - (pointD.x * t2_3));
  const mVec2t<T> y12 = mSolveXY(p1, q1, pointB.y - (pointA.y * i1_3) - (pointD.y * t1_3), p2, q2, pointC.y - (pointA.y * i2_3) - (pointD.y * t2_3));

  return mCubicBezierCurve<T>(pointA, pointD, mVec2t<T>(x12.x, y12.x), mVec2t<T>(x12.y, y12.y));
}

#endif // mBezierCurve_h__