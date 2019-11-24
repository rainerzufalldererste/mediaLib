#ifndef mSolve_h__
#define mSolve_h__

#include "mediaLib.h"

// Solve linear equation: a * x + b * y = c and d * x + e * y = f for x and y.
// Returns mVec2t<T>(x, y).
template <typename T>
inline mVec2t<T> mSolveXY(const T a, const T b, const T c, const T d, const T e, const T f)
{
  const T y = (c - a / d * f) / (b - a * e / d);
  const T x = (c - (b * y)) / a;

  return mVec2t<T>(x, y);
}

#endif // mSolve_h__
