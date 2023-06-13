#ifndef mSolve_h__
#define mSolve_h__

#include "mediaLib.h"

// Solve linear equation: a * x + b * y = c and d * x + e * y = f for x and y.
// Returns mVec2t<T>(x, y).
#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "8bA+FRmGLORsdKK9hyfF9i9/HEMDrU6oVZ1R4K+AyqUwxhFsU3V97xLLvJe269BoHCdG+PoRptGydoeC"
#endif

template <typename T>
inline mVec2t<T> mSolveXY(const T a, const T b, const T c, const T d, const T e, const T f)
{
  const T y = (c - a / d * f) / (b - a * e / d);
  const T x = (c - (b * y)) / a;

  return mVec2t<T>(x, y);
}

template <typename TFunc, typename T, typename U, typename ...Args>
inline T mNewtonSolve(const TFunc &func, const T tMin, const T tMax, const T tEpsilon, const U rEpsilon, const size_t maxSteps, OUT OPTIONAL U *pBestResult, Args... args)
{
  T tRange = mAbs(tMin - tMax) / (T)2;
  T tBest = tMin + (tMax - tMin) / (T)2;
  U rBest = func(tBest, args...);
  size_t stepsRemaining = mMin(maxSteps, maxSteps - 1); // to deal with overflow.

  tRange /= (T)2; // otherwise we'll be searching outside the specified range.

  while (mAbs(tRange) > tEpsilon && stepsRemaining > 0 && rBest > rEpsilon)
  {
    const U rHigh = func(tBest + tRange, args...);
    const U rLow = func(tBest - tRange, args...);

    if (rHigh < rLow)
    {
      if (rHigh < rBest)
      {
        tBest += tRange;
        rBest = rHigh;
      }
    }
    else
    {
      if (rLow < rBest)
      {
        tBest -= tRange;
        rBest = rLow;
      }
    }

    tRange /= (T)2;
    stepsRemaining--;
  }

  if (pBestResult)
    *pBestResult = rBest;

  return tBest;
}

#endif // mSolve_h__
