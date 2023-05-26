#ifndef mTriangulation_h__
#define mTriangulation_h__

#include "mediaLib.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "VOTJo1SIoEs14rKqq+zOkX8WiNVNR1ibj+ZdPHOPO7SrmuljP0HDc0Q70wZTa8QyITzcyzYeyg+xHNdM"
#endif

float_t mPolygon_SignedArea(const mVec2f *pPolygon, const size_t count);
inline bool mPolygon_IsClockwise(const mVec2f *pPolygon, const size_t count) { return mPolygon_SignedArea(pPolygon, count) >= 0; }

mFUNCTION(mTriangulation_Process, IN const mVec2f *pPolygon, const size_t count, IN mAllocator *pAllocator, OUT mTriangle2D<float_t> **ppTriangles, OUT size_t *pTriCount);
mFUNCTION(mTriangulation_Process, IN const mVec2f *pPolygon, const size_t count, const mVec2f **ppHoles, const size_t *pHoleCounts, const size_t holeCount, IN mAllocator *pAllocator, OUT mTriangle2D<float_t> **ppTriangles, OUT size_t *pTriCount);

#endif // mTriangulation_h__
