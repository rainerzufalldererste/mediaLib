// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "mTestLib.h"

mFUNCTION(mTestLib_RunAllTests, int * pArgc, char ** pArgv)
{
  ::testing::InitGoogleTest(pArgc, pArgv);

  return RUN_ALL_TESTS() == 0 ? mR_Success : mR_Failure;
}

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

mFUNCTION(mTestAllocator_Destroy, IN void *pUserData)
{
  mFUNCTION_SETUP();

  mERROR_IF(pUserData == nullptr, mR_ArgumentNull);

#ifdef mDEBUG_TESTS
  mASSERT(*(volatile size_t *)pUserData == 0, "Not all memory allocated by this allocator has been released.");
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
