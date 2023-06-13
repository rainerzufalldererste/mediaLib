#include "mTriangulation.h"

#pragma warning(push)
#pragma warning(disable: 4505)
#pragma warning(disable: 4706)
typedef int64_t ssize_t;

#define MPE_POLY2TRI_IMPLEMENTATION
#include "fast-poly2tri/include/MPE_fastpoly2tri.h"
#pragma warning(pop)

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "wrPvpf4oI0rPW7pI2RYXK1uZQt6pxWYa7cIikL9iI87XNZPIiWXRRDS9lASCPuOu1dAHrKgS6mFlXu+r"
#endif

float_t mPolygon_SignedArea(const mVec2f *pPolygon, const size_t count)
{
  float_t area = 0.0f;

  for (size_t p = count - 1, q = 0; q < count; p = q++)
    area += pPolygon[p].x * pPolygon[q].y - pPolygon[q].x * pPolygon[p].y;

  return area * 0.5f;
}

mFUNCTION(mTriangulation_Process, IN const mVec2f *pPolygon, const size_t count, IN mAllocator *pAllocator, OUT mTriangle2D<float_t> **ppTriangles, OUT size_t *pTriCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(pPolygon == nullptr || ppTriangles == nullptr || pTriCount == nullptr, mR_ArgumentNull);
  mERROR_IF(count <= 2, mR_ArgumentOutOfBounds);
  mERROR_IF(count > UINT32_MAX, mR_ArgumentOutOfBounds);

  const size_t requiredSize = MPE_PolyMemoryRequired((uint32_t)count);
  uint8_t *pContext = nullptr;
  mDEFER_CALL_2(mAllocator_FreePtr, &mDefaultTempAllocator, &pContext);
  mERROR_CHECK(mAllocator_AllocateZero(&mDefaultTempAllocator, &pContext, requiredSize));
  
  MPEPolyContext context;
  mERROR_IF(!MPE_PolyInitContext(&context, pContext, (uint32_t)count), mR_InternalError);
  
  for (size_t i = 0; i < count; i++)
  {
    MPEPolyPoint *pPoint = MPE_PolyPushPoint(&context);
    pPoint->X = pPolygon[i].x;
    pPoint->Y = pPolygon[i].y;
  }

  MPE_PolyAddEdge(&context);
  MPE_PolyTriangulate(&context);

  mDEFER_ON_ERROR(
    *pTriCount = 0;
    mAllocator_FreePtr(pAllocator, ppTriangles);
  );

  *pTriCount = context.TriangleCount;
  mERROR_CHECK(mAllocator_Allocate(pAllocator, ppTriangles, context.TriangleCount));

  for (uint32_t i = 0; i < context.TriangleCount; ++i)
  {
    MPEPolyTriangle *pTriangle = context.Triangles[i];

    for (size_t j = 0; j < 3; j++)
    {
      (*ppTriangles)[i].position[j].x = pTriangle->Points[j]->X;
      (*ppTriangles)[i].position[j].y = pTriangle->Points[j]->Y;
    }
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mTriangulation_Process, IN const mVec2f *pPolygon, const size_t count, const mVec2f **ppHoles, const size_t *pHoleCounts, const size_t holeCount, IN mAllocator *pAllocator, OUT mTriangle2D<float_t> **ppTriangles, OUT size_t *pTriCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(pPolygon == nullptr || ppTriangles == nullptr || pTriCount == nullptr || ppHoles == nullptr || pHoleCounts == nullptr, mR_ArgumentNull);
  mERROR_IF(count <= 2, mR_ArgumentOutOfBounds);
  mERROR_IF(count > UINT32_MAX, mR_ArgumentOutOfBounds);

  size_t totalCount = count;

  for (size_t i = 0; i < holeCount; i++)
    totalCount += pHoleCounts[i];

  const size_t requiredSize = MPE_PolyMemoryRequired((uint32_t)totalCount);
  uint8_t *pContext = nullptr;
  mDEFER_CALL_2(mAllocator_FreePtr, &mDefaultTempAllocator, &pContext);
  mERROR_CHECK(mAllocator_AllocateZero(&mDefaultTempAllocator, &pContext, requiredSize));
  
  MPEPolyContext context;
  mERROR_IF(!MPE_PolyInitContext(&context, pContext, (uint32_t)totalCount), mR_InternalError);
  
  for (size_t i = 0; i < count; i++)
  {
    MPEPolyPoint *pPoint = MPE_PolyPushPoint(&context);
    pPoint->X = pPolygon[i].x;
    pPoint->Y = pPolygon[i].y;
  }

  MPE_PolyAddEdge(&context);

  for (size_t i = 0; i < holeCount; i++)
  {
    MPEPolyPoint *pHole = MPE_PolyPushPointArray(&context, 4);

    for (size_t j = 0; j < pHoleCounts[i]; j++)
    {
      pHole[j].X = ppHoles[i][j].x;
      pHole[j].Y = ppHoles[i][j].y;
    }

    MPE_PolyAddHole(&context);
  }

  MPE_PolyTriangulate(&context);

  mDEFER_ON_ERROR(
    *pTriCount = 0;
    mAllocator_FreePtr(pAllocator, ppTriangles);
  );

  *pTriCount = context.TriangleCount;
  mERROR_CHECK(mAllocator_Allocate(pAllocator, ppTriangles, context.TriangleCount));

  for (uint32_t i = 0; i < context.TriangleCount; ++i)
  {
    MPEPolyTriangle *pTriangle = context.Triangles[i];

    for (size_t j = 0; j < 3; j++)
    {
      (*ppTriangles)[i].position[j].x = pTriangle->Points[j]->X;
      (*ppTriangles)[i].position[j].y = pTriangle->Points[j]->Y;
    }
  }

  mRETURN_SUCCESS();
}
