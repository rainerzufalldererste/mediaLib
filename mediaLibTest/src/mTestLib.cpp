#include "mTestLib.h"

void _CallCouninitialize()
{
  CoUninitialize();
}

mFUNCTION(mTestLib_RunAllTests, int * pArgc, char ** pArgv)
{
  CoInitialize(nullptr);
  atexit(_CallCouninitialize);

  ::testing::InitGoogleTest(pArgc, pArgv);

  return RUN_ALL_TESTS() == 0 ? mR_Success : mR_Failure;
}

//////////////////////////////////////////////////////////////////////////

size_t mTestDestructible_Count = 0;

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mTestAllocator_Alloc, OUT uint8_t **ppData, const size_t size, const size_t count, IN void *pUserData)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mAlloc(ppData, size * count));
  ++(*(volatile size_t *)pUserData);

  mRETURN_SUCCESS();
}

mFUNCTION(mTestAllocator_AllocZero, OUT uint8_t **ppData, const size_t size, const size_t count, IN void *pUserData)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mAllocZero(ppData, size * count));
  ++(*(volatile size_t *)pUserData);

  mRETURN_SUCCESS();
}

mFUNCTION(mTestAllocator_Realloc, OUT uint8_t **ppData, const size_t size, const size_t count, IN void *pUserData)
{
  mFUNCTION_SETUP();

  if (*ppData == nullptr)
    ++(*(volatile size_t *)pUserData);

  mERROR_CHECK(mRealloc(ppData, size * count));

  mRETURN_SUCCESS();
}

mFUNCTION(mTestAllocator_Free, OUT uint8_t *pData, IN void *pUserData)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mFree(pData));

  if(pData != nullptr)
    --(*(volatile size_t *)pUserData);

  mRETURN_SUCCESS();
}

mFUNCTION(mTestAllocator_Destroy, IN mAllocator *pAllocator, IN void *pUserData)
{
  mFUNCTION_SETUP();

  mERROR_IF(pUserData == nullptr, mR_ArgumentNull);

#ifdef mDEBUG_TESTS
  mASSERT(*(volatile size_t *)pUserData == 0, "Not all memory allocated by this allocator has been released.");

#ifdef mDEBUG_MEMORY_ALLOCATIONS
  mAllocatorDebugging_PrintRemainingMemoryAllocations(pAllocator);
#else
  mUnused(pAllocator);
#endif

#else
  mUnused(pAllocator);
#endif

  mERROR_CHECK(mFree(pUserData));

  mRETURN_SUCCESS();
}

mFUNCTION(mTestAllocator_Create, mAllocator *pTestAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTestAllocator == nullptr, mR_ArgumentNull);

  size_t *pUserData = nullptr;
  mERROR_CHECK(mAllocZero(&pUserData, 1));

  mERROR_CHECK(mAllocator_Create(pTestAllocator, &mTestAllocator_Alloc, &mTestAllocator_Realloc, &mTestAllocator_Free, nullptr, nullptr, &mTestAllocator_AllocZero, &mTestAllocator_Destroy, pUserData));

  mRETURN_SUCCESS();
}

mFUNCTION(mTestAllocator_GetCount, mAllocator * pAllocator, size_t *pCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAllocator == nullptr || pAllocator->pUserData == nullptr || pCount == nullptr, mR_ArgumentNull);

  *pCount = *(volatile size_t *)pAllocator->pUserData;

  mRETURN_SUCCESS();
}

mFUNCTION(mDummyDestructible_Create, mDummyDestructible *pDestructable, mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pDestructable == nullptr, mR_ArgumentNull);
  mERROR_IF(pAllocator == nullptr, mR_InvalidParameter); // should be test allocator.

  pDestructable->pAllocator = pAllocator;
  mERROR_CHECK(mAllocator_Allocate(pAllocator, &pDestructable->pData, 1));
  pDestructable->destructed = false;
  *pDestructable->pData = mTestDestructible_Count;
  pDestructable->index = mTestDestructible_Count++;

  mRETURN_SUCCESS();
}

mFUNCTION(mDestruct, mDummyDestructible *pDestructable)
{
  mFUNCTION_SETUP();

  mERROR_IF(pDestructable == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mAllocator_FreePtr(pDestructable->pAllocator, &pDestructable->pData));
  pDestructable->destructed = true;

  mRETURN_SUCCESS();
}
