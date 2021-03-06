// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "mTestLib.h"
#include "mRefPool.h"

mTEST(mRefPool, TestCreate)
{
  mTEST_ALLOCATOR_SETUP();

  mTEST_ASSERT_EQUAL(mR_ArgumentNull, mRefPool_Create((mPtr<mRefPool<mDummyDestructible>> *)nullptr, pAllocator));

  mPtr<mRefPool<mDummyDestructible>> refPool;
  mDEFER_DESTRUCTION(&refPool, mRefPool_Destroy);
  mTEST_ASSERT_SUCCESS(mRefPool_Create(&refPool, pAllocator));

  mTEST_ASSERT_NOT_EQUAL(refPool, nullptr);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mRefPool, TestAdd)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mRefPool<mDummyDestructible>> refPool;
  mDEFER_DESTRUCTION(&refPool, mRefPool_Destroy);
  mTEST_ASSERT_SUCCESS(mRefPool_Create(&refPool, pAllocator));

  mDummyDestructible dummy;
  mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy, pAllocator));

  mPtr<mDummyDestructible> ptr;
  mTEST_ASSERT_SUCCESS(mRefPool_Add(refPool, &dummy, &ptr));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mRefPool, TestAddDestruct)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mRefPool<mDummyDestructible>> refPool;
  mDEFER_DESTRUCTION(&refPool, mRefPool_Destroy);
  mTEST_ASSERT_SUCCESS(mRefPool_Create(&refPool, pAllocator));

  mDummyDestructible dummy;
  mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy, pAllocator));

  mPtr<mDummyDestructible> ptr;
  mTEST_ASSERT_SUCCESS(mRefPool_Add(refPool, &dummy, &ptr));
  mTEST_ASSERT_SUCCESS(mSharedPointer_Destroy(&ptr));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mRefPool, TestAddRemove)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mRefPool<mDummyDestructible>> refPool;
  mDEFER_DESTRUCTION(&refPool, mRefPool_Destroy);
  mTEST_ASSERT_SUCCESS(mRefPool_Create(&refPool, pAllocator));

  mDummyDestructible dummy;
  mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy, pAllocator));

  mPtr<mDummyDestructible> ptr;
  mTEST_ASSERT_SUCCESS(mRefPool_Add(refPool, &dummy, &ptr));

  mDummyDestructible dummy2;
  mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy2, pAllocator));

  mPtr<mDummyDestructible> ptr2;
  mTEST_ASSERT_SUCCESS(mRefPool_Add(refPool, &dummy2, &ptr2));

  mTEST_ASSERT_SUCCESS(mSharedPointer_Destroy(&ptr));

  mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy, pAllocator));
  mTEST_ASSERT_SUCCESS(mRefPool_Add(refPool, &dummy, &ptr));

  mDummyDestructible dummy3;
  mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy3, pAllocator));

  mPtr<mDummyDestructible> ptr3;
  mTEST_ASSERT_SUCCESS(mRefPool_Add(refPool, &dummy3, &ptr3));

  mTEST_ASSERT_SUCCESS(mSharedPointer_Destroy(&ptr2));
  mTEST_ASSERT_SUCCESS(mSharedPointer_Destroy(&ptr3));
  mTEST_ASSERT_SUCCESS(mSharedPointer_Destroy(&ptr));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mRefPool, TestForeach)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mRefPool<mDummyDestructible>> refPool;
  mDEFER_DESTRUCTION(&refPool, mRefPool_Destroy);
  mTEST_ASSERT_SUCCESS(mRefPool_Create(&refPool, pAllocator));

  mDummyDestructible dummy;
  mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy, pAllocator));

  mPtr<mDummyDestructible> ptr;
  mTEST_ASSERT_SUCCESS(mRefPool_Add(refPool, &dummy, &ptr));

  mDummyDestructible dummy2;
  mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy2, pAllocator));

  mPtr<mDummyDestructible> ptr2;
  mTEST_ASSERT_SUCCESS(mRefPool_Add(refPool, &dummy2, &ptr2));

  size_t count = 0;
  mTEST_ASSERT_SUCCESS(mRefPool_ForEach(refPool,
    (std::function<mResult(mPtr<mDummyDestructible> &)>)[&](mPtr<mDummyDestructible> &d)
  {
    mFUNCTION_SETUP();

    mERROR_IF(d == nullptr, mR_ArgumentNull);
    ++count;

    mRETURN_SUCCESS();
  }));

  mTEST_ASSERT_EQUAL(2, count);

  mTEST_ASSERT_SUCCESS(mSharedPointer_Destroy(&ptr));

  count = 0;
  mTEST_ASSERT_SUCCESS(mRefPool_ForEach(refPool,
    (std::function<mResult(mPtr<mDummyDestructible> &)>)[&](mPtr<mDummyDestructible> &d)
  {
    mFUNCTION_SETUP();

    mERROR_IF(d == nullptr, mR_ArgumentNull);
    ++count;

    mRETURN_SUCCESS();
  }));

  mTEST_ASSERT_EQUAL(1, count);

  mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy, pAllocator));
  mTEST_ASSERT_SUCCESS(mRefPool_Add(refPool, &dummy, &ptr));

  mDummyDestructible dummy3;
  mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy3, pAllocator));

  mPtr<mDummyDestructible> ptr3;
  mTEST_ASSERT_SUCCESS(mRefPool_Add(refPool, &dummy3, &ptr3));

  count = 0;
  mTEST_ASSERT_SUCCESS(mRefPool_ForEach(refPool,
    (std::function<mResult(mPtr<mDummyDestructible> &)>)[&](mPtr<mDummyDestructible> &d)
  {
    mFUNCTION_SETUP();

    mERROR_IF(d == nullptr, mR_ArgumentNull);
    ++count;

    mRETURN_SUCCESS();
  }));

  mTEST_ASSERT_EQUAL(3, count);

  count = 0;
  mTEST_ASSERT_EQUAL(mR_InternalError, mRefPool_ForEach(refPool,
    (std::function<mResult(mPtr<mDummyDestructible> &)>)[&](mPtr<mDummyDestructible> &d)
  {
    mFUNCTION_SETUP();

    mERROR_IF(d == nullptr, mR_ArgumentNull);
    ++count;

    if (count == 2)
      mRETURN_RESULT(mR_InternalError);

    mRETURN_SUCCESS();
  }));

  mTEST_ASSERT_EQUAL(2, count);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mRefPool, TestCrush)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mRefPool<mDummyDestructible>> refPool;
  mDEFER_DESTRUCTION(&refPool, mRefPool_Destroy);
  mTEST_ASSERT_SUCCESS(mRefPool_Create(&refPool, pAllocator));

  mPtr<mChunkedArray<mPtr<mDummyDestructible>>> pointerStore;
  mDEFER_DESTRUCTION(&pointerStore, mChunkedArray_Destroy);
  mTEST_ASSERT_SUCCESS(mChunkedArray_Create(&pointerStore, pAllocator));

  const size_t maxCount = 1024;

  for (size_t i = 0; i < maxCount; ++i)
  {
    mDummyDestructible dummy;
    mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy, pAllocator));

    mPtr<mDummyDestructible> dummyPtr;
    mDEFER_DESTRUCTION(&dummyPtr, mSharedPointer_Destroy);
    mTEST_ASSERT_SUCCESS(mRefPool_Add(refPool, &dummy, &dummyPtr));

    mTEST_ASSERT_SUCCESS(mChunkedArray_PushBack(pointerStore, &dummyPtr));
  }

  size_t count = (size_t)-1;
  mTEST_ASSERT_SUCCESS(mRefPool_GetCount(refPool, &count));

  mTEST_ASSERT_EQUAL(count, maxCount);

  for (size_t i = 0; i < maxCount / 2; ++i)
  {
    mPtr<mDummyDestructible> dummyPtr;
    mDEFER_DESTRUCTION(&dummyPtr, mSharedPointer_Destroy);

    mTEST_ASSERT_SUCCESS(mChunkedArray_PopAt(pointerStore, i, &dummyPtr));
  }

  mTEST_ASSERT_SUCCESS(mRefPool_Crush(refPool));

  count = (size_t)-1;
  mTEST_ASSERT_SUCCESS(mRefPool_GetCount(refPool, &count));

  mTEST_ASSERT_EQUAL(count, maxCount / 2);

  std::function<mResult(mPtr<mDummyDestructible> &)> indexChecker =
    [count](mPtr<mDummyDestructible> &d)
  {
    mFUNCTION_SETUP();

    size_t index = (size_t)-1;
    mERROR_CHECK(mRefPool_GetPointerIndex(d, &index));
    mERROR_IF(index >= count, mR_InternalError);

    mRETURN_SUCCESS();
  };

  mTEST_ASSERT_SUCCESS(mRefPool_ForEach(refPool, indexChecker));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mRefPool, TestRemoveOwnReference)
{
  mTEST_ALLOCATOR_SETUP();

  {
    mPtr<mRefPool<mDummyDestructible>> refPool;
    mDEFER_DESTRUCTION(&refPool, mRefPool_Destroy);
    mTEST_ASSERT_SUCCESS(mRefPool_Create(&refPool, pAllocator, false));

    mTEST_ASSERT_EQUAL(mR_ResourceStateInvalid, mRefPool_RemoveOwnReference(refPool));
  }

  {
    mPtr<mRefPool<mDummyDestructible>> refPool;
    mDEFER_DESTRUCTION(&refPool, mRefPool_Destroy);
    mTEST_ASSERT_SUCCESS(mRefPool_Create(&refPool, pAllocator, true));

    const size_t maxCount = 1024;

    for (size_t i = 0; i < maxCount; i++)
    {
      mDummyDestructible dummy;
      mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy, pAllocator));

      mPtr<mDummyDestructible> ptr;
      mDEFER_DESTRUCTION(&ptr, mSharedPointer_Destroy);
      mTEST_ASSERT_SUCCESS(mRefPool_Add(refPool, &dummy, &ptr));
    }

    size_t count = (size_t)-1;
    mTEST_ASSERT_SUCCESS(mRefPool_GetCount(refPool, &count));
    mTEST_ASSERT_EQUAL(count, maxCount);

    mTEST_ASSERT_SUCCESS(mRefPool_RemoveOwnReference(refPool));
    mTEST_ASSERT_SUCCESS(mRefPool_Crush(refPool));

    mTEST_ASSERT_SUCCESS(mRefPool_GetCount(refPool, &count));
    mTEST_ASSERT_EQUAL(count, 0);

    mTEST_ASSERT_EQUAL(mR_ResourceStateInvalid, mRefPool_RemoveOwnReference(refPool));
  }

  {
    mPtr<mRefPool<mDummyDestructible>> refPool;
    mDEFER_DESTRUCTION(&refPool, mRefPool_Destroy);
    mTEST_ASSERT_SUCCESS(mRefPool_Create(&refPool, pAllocator, true));

    const size_t maxCount = 1024;

    for (size_t i = 0; i < maxCount; i++)
    {
      mDummyDestructible dummy;
      mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy, pAllocator));

      mPtr<mDummyDestructible> ptr;
      mDEFER_DESTRUCTION(&ptr, mSharedPointer_Destroy);
      mTEST_ASSERT_SUCCESS(mRefPool_Add(refPool, &dummy, &ptr));
    }
  }

  mTEST_ALLOCATOR_ZERO_CHECK();
}
