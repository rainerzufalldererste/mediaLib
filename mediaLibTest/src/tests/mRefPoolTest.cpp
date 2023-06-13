#include "mTestLib.h"
#include "mRefPool.h"

mTEST(mRefPool, TestCreate)
{
  mTEST_ALLOCATOR_SETUP();

  mTEST_ASSERT_EQUAL(mR_ArgumentNull, mRefPool_Create((mPtr<mRefPool<mDummyDestructible>> *)nullptr, pAllocator));

  mPtr<mRefPool<mDummyDestructible>> refPool;
  mDEFER_CALL(&refPool, mRefPool_Destroy);
  mTEST_ASSERT_SUCCESS(mRefPool_Create(&refPool, pAllocator));

  mTEST_ASSERT_NOT_EQUAL(refPool, nullptr);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mRefPool, TestAdd)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mRefPool<mDummyDestructible>> refPool;
  mDEFER_CALL(&refPool, mRefPool_Destroy);
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
  mDEFER_CALL(&refPool, mRefPool_Destroy);
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
  mDEFER_CALL(&refPool, mRefPool_Destroy);
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
  mDEFER_CALL(&refPool, mRefPool_Destroy);
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

  const std::function<mResult(mPtr<mDummyDestructible> &)> &funcReturnErrorCode = [&](mPtr<mDummyDestructible> &d)
  {
    mFUNCTION_SETUP();

    mERROR_IF(d == nullptr, mR_ArgumentNull);
    ++count;

    if (count == 2)
      mRETURN_RESULT(mR_InternalError);

    mRETURN_SUCCESS();
  };

  count = 0;
  mTEST_ASSERT_EQUAL(mR_InternalError, mRefPool_ForEach(refPool, funcReturnErrorCode));

  mTEST_ASSERT_EQUAL(2, count);

  const std::function<mResult(mPtr<mDummyDestructible> &)> &funcBreak = [&](mPtr<mDummyDestructible> &d)
  {
    mFUNCTION_SETUP();

    mERROR_IF(d == nullptr, mR_ArgumentNull);
    ++count;

    if (count == 2)
      mRETURN_RESULT(mR_Break);

    mRETURN_SUCCESS();
  };

  count = 0;
  mTEST_ASSERT_EQUAL(mR_Success, mRefPool_ForEach(refPool, funcBreak));

  mTEST_ASSERT_EQUAL(2, count);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mRefPool, TestRangeBasedLoop)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mRefPool<mDummyDestructible>> refPool;
  mDEFER_CALL(&refPool, mRefPool_Destroy);
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

  for (auto _item : refPool->Iterate())
  {
    mTEST_ASSERT_FALSE(_item.data == nullptr);
    ++count;
  }

  mTEST_ASSERT_EQUAL(2, count);

  mTEST_ASSERT_SUCCESS(mSharedPointer_Destroy(&ptr));

  count = 0;

  for (auto _item : refPool->Iterate())
  {
    mTEST_ASSERT_FALSE(_item.data == nullptr);
    ++count;
  }

  mTEST_ASSERT_EQUAL(1, count);

  mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy, pAllocator));
  mTEST_ASSERT_SUCCESS(mRefPool_Add(refPool, &dummy, &ptr));

  mDummyDestructible dummy3;
  mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy3, pAllocator));

  mPtr<mDummyDestructible> ptr3;
  mTEST_ASSERT_SUCCESS(mRefPool_Add(refPool, &dummy3, &ptr3));

  count = 0;

  for (auto _item : refPool->Iterate())
  {
    mTEST_ASSERT_FALSE(_item.data == nullptr);

    size_t ptrIndex = 0;
    mTEST_ASSERT_SUCCESS(mRefPool_GetPointerIndex(_item.data, &ptrIndex));

    mTEST_ASSERT_EQUAL(ptrIndex, _item.index);

    ++count;
  }

  mTEST_ASSERT_EQUAL(3, count);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mRefPool, TestConstRangeBasedLoop)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mRefPool<mDummyDestructible>> refPool;
  mDEFER_CALL(&refPool, mRefPool_Destroy);
  mTEST_ASSERT_SUCCESS(mRefPool_Create(&refPool, pAllocator));

  const mPtr<mRefPool<mDummyDestructible>> constRefPool = refPool;

  mDummyDestructible dummy;
  mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy, pAllocator));

  mPtr<mDummyDestructible> ptr;
  mTEST_ASSERT_SUCCESS(mRefPool_Add(refPool, &dummy, &ptr));

  mDummyDestructible dummy2;
  mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy2, pAllocator));

  mPtr<mDummyDestructible> ptr2;
  mTEST_ASSERT_SUCCESS(mRefPool_Add(refPool, &dummy2, &ptr2));

  size_t count = 0;

  for (const auto _item : constRefPool->Iterate())
  {
    mTEST_ASSERT_FALSE(_item.data == nullptr);
    ++count;
  }

  mTEST_ASSERT_EQUAL(2, count);

  mTEST_ASSERT_SUCCESS(mSharedPointer_Destroy(&ptr));

  count = 0;

  for (const auto _item : constRefPool->Iterate())
  {
    mTEST_ASSERT_FALSE(_item.data == nullptr);
    ++count;
  }

  mTEST_ASSERT_EQUAL(1, count);

  mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy, pAllocator));
  mTEST_ASSERT_SUCCESS(mRefPool_Add(refPool, &dummy, &ptr));

  mDummyDestructible dummy3;
  mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy3, pAllocator));

  mPtr<mDummyDestructible> ptr3;
  mTEST_ASSERT_SUCCESS(mRefPool_Add(refPool, &dummy3, &ptr3));

  count = 0;
  
  for (const auto _item : constRefPool->Iterate())
  {
    mTEST_ASSERT_FALSE(_item.data == nullptr);

    size_t ptrIndex = 0;
    mTEST_ASSERT_SUCCESS(mRefPool_GetPointerIndex(_item.data, &ptrIndex));

    mTEST_ASSERT_EQUAL(ptrIndex, _item.index);

    ++count;
  }

  mTEST_ASSERT_EQUAL(3, count);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mRefPool, TestCrush)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mRefPool<mDummyDestructible>> refPool;
  mDEFER_CALL(&refPool, mRefPool_Destroy);
  mTEST_ASSERT_SUCCESS(mRefPool_Create(&refPool, pAllocator));

  mPtr<mChunkedArray<mPtr<mDummyDestructible>>> pointerStore;
  mDEFER_CALL(&pointerStore, mChunkedArray_Destroy);
  mTEST_ASSERT_SUCCESS(mChunkedArray_Create(&pointerStore, pAllocator));

  const size_t maxCount = 1024;

  for (size_t i = 0; i < maxCount; ++i)
  {
    mDummyDestructible dummy;
    mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy, pAllocator));

    mPtr<mDummyDestructible> dummyPtr;
    mDEFER_CALL(&dummyPtr, mSharedPointer_Destroy);
    mTEST_ASSERT_SUCCESS(mRefPool_Add(refPool, &dummy, &dummyPtr));

    mTEST_ASSERT_SUCCESS(mChunkedArray_PushBack(pointerStore, &dummyPtr));
  }

  size_t count = (size_t)-1;
  mTEST_ASSERT_SUCCESS(mRefPool_GetCount(refPool, &count));

  mTEST_ASSERT_EQUAL(count, maxCount);

  for (size_t i = 0; i < maxCount / 2; ++i)
  {
    mPtr<mDummyDestructible> dummyPtr;
    mDEFER_CALL(&dummyPtr, mSharedPointer_Destroy);

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
    mDEFER_CALL(&refPool, mRefPool_Destroy);
    mTEST_ASSERT_SUCCESS(mRefPool_Create(&refPool, pAllocator, false));

    mTEST_ASSERT_EQUAL(mR_ResourceStateInvalid, mRefPool_RemoveOwnReference(refPool));
  }

  {
    mPtr<mRefPool<mDummyDestructible>> refPool;
    mDEFER_CALL(&refPool, mRefPool_Destroy);
    mTEST_ASSERT_SUCCESS(mRefPool_Create(&refPool, pAllocator, true));

    const size_t maxCount = 1024;

    for (size_t i = 0; i < maxCount; i++)
    {
      mDummyDestructible dummy;
      mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy, pAllocator));

      mPtr<mDummyDestructible> ptr;
      mDEFER_CALL(&ptr, mSharedPointer_Destroy);
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
    mDEFER_CALL(&refPool, mRefPool_Destroy);
    mTEST_ASSERT_SUCCESS(mRefPool_Create(&refPool, pAllocator, true));

    mPtr<mQueue<mPtr<mDummyDestructible>>> queue;
    mDEFER_CALL(&queue, mQueue_Destroy);
    mTEST_ASSERT_SUCCESS(mQueue_Create(&queue, pAllocator));

    const size_t maxCount = 1024;

    for (size_t i = 0; i < maxCount; i++)
    {
      mDummyDestructible dummy;
      mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy, pAllocator));

      mPtr<mDummyDestructible> ptr;
      mDEFER_CALL(&ptr, mSharedPointer_Destroy);
      mTEST_ASSERT_SUCCESS(mRefPool_Add(refPool, &dummy, &ptr));

      if ((i & 1) == 0)
        mTEST_ASSERT_SUCCESS(mQueue_PushBack(queue, &ptr));
    }

    mTEST_ASSERT_SUCCESS(mRefPool_RemoveOwnReference(refPool));

    size_t count = (size_t)-1;
    mTEST_ASSERT_SUCCESS(mRefPool_GetCount(refPool, &count));
    mTEST_ASSERT_EQUAL(count, maxCount / 2);

    size_t forEachCount = 0;

    mTEST_ASSERT_SUCCESS(mRefPool_ForEach(refPool, 
      (std::function<mResult(mPtr<mDummyDestructible> &)>) [&forEachCount](mPtr<mDummyDestructible> &) 
    { 
      ++forEachCount; 
      RETURN mR_Success; 
    }));

    mTEST_ASSERT_EQUAL(forEachCount, count);

    mTEST_ASSERT_EQUAL(mR_ResourceStateInvalid, mRefPool_RemoveOwnReference(refPool));
  }

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mRefPool, TestKeepIfTrue)
{
  mTEST_ALLOCATOR_SETUP();

  {
    mPtr<mRefPool<mDummyDestructible>> refPool;
    mDEFER_CALL(&refPool, mRefPool_Destroy);
    mTEST_ASSERT_SUCCESS(mRefPool_Create(&refPool, pAllocator, false));

    mTEST_ASSERT_EQUAL(mR_ResourceStateInvalid, mRefPool_KeepIfTrue(refPool, 
      (std::function<mResult (mPtr<mDummyDestructible> &, OUT bool *)>)[](mPtr<mDummyDestructible> &, OUT bool *pKeep) 
    {
      *pKeep = true;
      return mR_Success;
    }));
  }

  {
    mPtr<mRefPool<mDummyDestructible>> refPool;
    mDEFER_CALL(&refPool, mRefPool_Destroy);
    mTEST_ASSERT_SUCCESS(mRefPool_Create(&refPool, pAllocator, true));

    const size_t maxCount = 1024;

    for (size_t i = 0; i < maxCount; i++)
    {
      mDummyDestructible dummy;
      mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy, pAllocator));

      mPtr<mDummyDestructible> ptr;
      mDEFER_CALL(&ptr, mSharedPointer_Destroy);
      mTEST_ASSERT_SUCCESS(mRefPool_Add(refPool, &dummy, &ptr));
    }

    size_t count = (size_t)-1;
    mTEST_ASSERT_SUCCESS(mRefPool_GetCount(refPool, &count));

    mTEST_ASSERT_SUCCESS(mRefPool_KeepIfTrue(refPool,
      (std::function<mResult(mPtr<mDummyDestructible> &, OUT bool *)>)[](mPtr<mDummyDestructible> &, OUT bool *pKeep)
    {
      *pKeep = true;
      return mR_Success;
    }));

    mTEST_ASSERT_SUCCESS(mRefPool_GetCount(refPool, &count));
    mTEST_ASSERT_EQUAL(count, maxCount);

    mTEST_ASSERT_SUCCESS(mRefPool_KeepIfTrue(refPool,
      (std::function<mResult(mPtr<mDummyDestructible> &, OUT bool *)>)[&](mPtr<mDummyDestructible> &dummy, OUT bool *pKeep)
    {
      *pKeep = dummy->index < maxCount / 2;
      return mR_Success;
    }));

    mTEST_ASSERT_SUCCESS(mRefPool_GetCount(refPool, &count));
    mTEST_ASSERT_EQUAL(count, maxCount / 2);

    std::function<mResult(mPtr<mDummyDestructible> &)> indexChecker =
      [maxCount](mPtr<mDummyDestructible> &d)
    {
      mFUNCTION_SETUP();

      size_t index = (size_t)-1;
      mERROR_CHECK(mRefPool_GetPointerIndex(d, &index));
      mERROR_IF(index >= maxCount / 2, mR_InternalError);

      mRETURN_SUCCESS();
    };

    mTEST_ASSERT_SUCCESS(mRefPool_ForEach(refPool, indexChecker));

    mTEST_ASSERT_EQUAL(mR_Failure, mRefPool_KeepIfTrue(refPool,
      (std::function<mResult(mPtr<mDummyDestructible> &, OUT bool *)>)[&](mPtr<mDummyDestructible> &, OUT bool *pKeep)
    {
      *pKeep = false;
      return mR_Failure;
    }));

    mTEST_ASSERT_SUCCESS(mRefPool_GetCount(refPool, &count));
    mTEST_ASSERT_EQUAL(count, maxCount / 2);

    mTEST_ASSERT_SUCCESS(mRefPool_KeepIfTrue(refPool,
      (std::function<mResult(mPtr<mDummyDestructible> &, OUT bool *)>)[&](mPtr<mDummyDestructible> &, OUT bool *pKeep)
    {
      *pKeep = false;
      return mR_Success;
    }));

    mTEST_ASSERT_SUCCESS(mRefPool_GetCount(refPool, &count));
    mTEST_ASSERT_EQUAL(count, 0);
  }

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mRefPool, TestEquals)
{
  mTEST_ALLOCATOR_SETUP();

  struct _inner
  {
    static bool IsNotZero(const mPtr<size_t> &a)
    {
      return *a != 0;
    }

    static bool IsSame(const mPtr<size_t> &a, const mPtr<size_t> &b)
    {
      return *a == *b;
    }
  };

  {
    mPtr<mRefPool<size_t>> a;
    mPtr<mRefPool<size_t>> b;
    mTEST_ASSERT_TRUE((mRefPool_Equals<size_t, mFN_WRAPPER(_inner::IsSame)>(a, b)));

    mDEFER_CALL(&a, mRefPool_Destroy);
    mTEST_ASSERT_SUCCESS(mRefPool_Create(&a, pAllocator, true));
    mTEST_ASSERT_FALSE((mRefPool_Equals<size_t, mFN_WRAPPER(_inner::IsSame)>(a, b)));
    mTEST_ASSERT_FALSE((mRefPool_Equals<size_t, mFN_WRAPPER(_inner::IsSame)>(b, a)));

    mDEFER_CALL(&b, mRefPool_Destroy);
    mTEST_ASSERT_SUCCESS(mRefPool_Create(&b, pAllocator, true));

    mTEST_ASSERT_TRUE((mRefPool_Equals<size_t, mFN_WRAPPER(_inner::IsSame)>(a, b)));
  }

  {
    mPtr<mRefPool<size_t>> a;
    mPtr<mRefPool<size_t>> b;
    mTEST_ASSERT_TRUE((mRefPool_Equals<size_t, mFN_WRAPPER(_inner::IsSame)>(a, b)));
    mTEST_ASSERT_TRUE((mRefPool_Equals<size_t, mFN_WRAPPER(_inner::IsSame)>(b, a)));

    mDEFER_CALL(&a, mRefPool_Destroy);
    mTEST_ASSERT_SUCCESS(mRefPool_Create(&a, pAllocator, true));
    mTEST_ASSERT_FALSE((mRefPool_Equals<size_t, mFN_WRAPPER(_inner::IsSame)>(a, b)));
    mTEST_ASSERT_FALSE((mRefPool_Equals<size_t, mFN_WRAPPER(_inner::IsSame)>(b, a)));

    mDEFER_CALL(&b, mRefPool_Destroy);
    mTEST_ASSERT_SUCCESS(mRefPool_Create(&b, pAllocator, true));

    mTEST_ASSERT_TRUE((mRefPool_Equals<size_t, mFN_WRAPPER(_inner::IsSame)>(a, b)));
    mTEST_ASSERT_TRUE((mRefPool_Equals<size_t, mFN_WRAPPER(_inner::IsSame)>(b, a)));

    mPtr<size_t> index;

    size_t value = 0;
    mTEST_ASSERT_SUCCESS(mRefPool_Add(a, &value, &index));
    mTEST_ASSERT_FALSE((mRefPool_Equals<size_t, mFN_WRAPPER(_inner::IsSame)>(a, b)));
    mTEST_ASSERT_FALSE((mRefPool_Equals<size_t, mFN_WRAPPER(_inner::IsSame)>(b, a)));

    value = 1;
    mTEST_ASSERT_SUCCESS(mRefPool_Add(b, &value, &index));
    mTEST_ASSERT_FALSE((mRefPool_Equals<size_t, mFN_WRAPPER(_inner::IsSame)>(a, b)));
    mTEST_ASSERT_FALSE((mRefPool_Equals<size_t, mFN_WRAPPER(_inner::IsSame)>(b, a)));

    value = 1;
    mTEST_ASSERT_SUCCESS(mRefPool_Add(b, &value, &index));
    mTEST_ASSERT_FALSE((mRefPool_Equals<size_t, mFN_WRAPPER(_inner::IsSame)>(a, b)));
    mTEST_ASSERT_FALSE((mRefPool_Equals<size_t, mFN_WRAPPER(_inner::IsSame)>(b, a)));

    mTEST_ASSERT_SUCCESS(mRefPool_PeekAt(b, 0, &index));
    *index = 0;
    mTEST_ASSERT_FALSE((mRefPool_Equals<size_t, mFN_WRAPPER(_inner::IsSame)>(a, b)));
    mTEST_ASSERT_FALSE((mRefPool_Equals<size_t, mFN_WRAPPER(_inner::IsSame)>(b, a)));

    value = 1;
    mTEST_ASSERT_SUCCESS(mRefPool_Add(a, &value, &index));
    mTEST_ASSERT_TRUE((mRefPool_Equals<size_t, mFN_WRAPPER(_inner::IsSame)>(a, b)));
    mTEST_ASSERT_TRUE((mRefPool_Equals<size_t, mFN_WRAPPER(_inner::IsSame)>(b, a)));

    value = 2;
    mTEST_ASSERT_SUCCESS(mRefPool_Add(a, &value, &index));
    mTEST_ASSERT_FALSE((mRefPool_Equals<size_t, mFN_WRAPPER(_inner::IsSame)>(a, b)));
    mTEST_ASSERT_FALSE((mRefPool_Equals<size_t, mFN_WRAPPER(_inner::IsSame)>(b, a)));

    value = 2;
    mTEST_ASSERT_SUCCESS(mRefPool_Add(b, &value, &index));
    mTEST_ASSERT_TRUE((mRefPool_Equals<size_t, mFN_WRAPPER(_inner::IsSame)>(a, b)));
    mTEST_ASSERT_TRUE((mRefPool_Equals<size_t, mFN_WRAPPER(_inner::IsSame)>(b, a)));
  }

  constexpr size_t x = (size_t)-1;

  const auto &cmpFunc = [pAllocator, x](const size_t *pA, const size_t aCount, const size_t *pB, const size_t bCount)
  {
    mPtr<mRefPool<size_t>> a, b;
    mASSERT(mSUCCEEDED(mRefPool_Create(&a, pAllocator, true)), "");
    mASSERT(mSUCCEEDED(mRefPool_Create(&b, pAllocator, true)), "");

    mUniqueContainer<mQueue<mPtr<size_t>>> tempStorage;
    mASSERT(mSUCCEEDED(mQueue_Create(&tempStorage, pAllocator)), "");

    mPtr<size_t> index;

    for (size_t i = 0; i < aCount; i++)
    {
      size_t j = pA[i];
      mASSERT(mSUCCEEDED(mRefPool_Add(a, &j, &index)), "");
    }

    for (size_t i = 0; i < aCount; i++)
    {
      if (pA[i] != x)
      {
        mASSERT(mSUCCEEDED(mRefPool_PeekAt(a, i, &index)), "");
        mASSERT(mSUCCEEDED(mQueue_PushBack(tempStorage, std::move(index))), "");
      }
    }

    for (size_t i = 0; i < bCount; i++)
    {
      size_t j = pB[i];
      mASSERT(mSUCCEEDED(mRefPool_Add(b, &j, &index)), "");
    }

    for (size_t i = 0; i < bCount; i++)
    {
      if (pB[i] != x)
      {
        mASSERT(mSUCCEEDED(mRefPool_PeekAt(b, i, &index)), "");
        mASSERT(mSUCCEEDED(mQueue_PushBack(tempStorage, std::move(index))), "");
      }
    }

    index = nullptr;
    
    mASSERT(mSUCCEEDED(mRefPool_RemoveOwnReference(a)), "");
    mASSERT(mSUCCEEDED(mRefPool_RemoveOwnReference(b)), "");

    const bool ret = mRefPool_Equals<size_t, mFN_WRAPPER(_inner::IsSame), mFN_WRAPPER(_inner::IsNotZero)>(a, b);
    mASSERT(ret == (mRefPool_Equals<size_t, mFN_WRAPPER(_inner::IsSame), mFN_WRAPPER(_inner::IsNotZero)>(b, a)), "");

    return ret;
  };

  {
    const size_t data0[] = { 0, x, 0 };
    const size_t data1[] = { 0, x, 0 };
    mTEST_ASSERT_TRUE(cmpFunc(data0, mARRAYSIZE(data0), data1, mARRAYSIZE(data1)));
  }

  {
    const size_t data0[] = { x, x, 0, 1, 0, 0, 0, x, x, x };
    const size_t data1[] = { x, x, 0, 1, 0, 0, 0 };
    mTEST_ASSERT_TRUE(cmpFunc(data0, mARRAYSIZE(data0), data1, mARRAYSIZE(data1)));
  }

  {
    const size_t data0[] = { x, 1, 0 };
    const size_t data1[] = { x, 1, 0 };
    mTEST_ASSERT_TRUE(cmpFunc(data0, mARRAYSIZE(data0), data1, mARRAYSIZE(data1)));
  }

  {
    const size_t data0[] = { x, x, x, x, x };
    const size_t data1[] = { x, x, x, x, x, x, x, x, x };
    mTEST_ASSERT_TRUE(cmpFunc(data0, mARRAYSIZE(data0), data1, mARRAYSIZE(data1)));
  }

  {
    const size_t data0[] = { x, 1, 2, 0 };
    const size_t data1[] = { 0, 1, 2, 0 };
    mTEST_ASSERT_TRUE(cmpFunc(data0, mARRAYSIZE(data0), data1, mARRAYSIZE(data1)));
  }

  {
    const size_t data0[] = { x, 1, 2, 0 };
    const size_t data1[] = { 1, 2, 2, 0 };
    mTEST_ASSERT_FALSE(cmpFunc(data0, mARRAYSIZE(data0), data1, mARRAYSIZE(data1)));
  }

  {
    const size_t data0[] = { 0, 1, x, x };
    const size_t data1[] = { 0, 1, x, 0, x, x };
    mTEST_ASSERT_TRUE(cmpFunc(data0, mARRAYSIZE(data0), data1, mARRAYSIZE(data1)));
  }

  {
    const size_t data0[] = { 0, 1, x, x };
    const size_t data1[] = { 0, 1, x, 1, x, x };
    mTEST_ASSERT_FALSE(cmpFunc(data0, mARRAYSIZE(data0), data1, mARRAYSIZE(data1)));
  }

  {
    const size_t data0[] = { x, x, 0, 1, 0, 0, 0, x, x, x };
    const size_t data1[] = { x, x, 0, 0, 1, 0, 0 };
    mTEST_ASSERT_FALSE(cmpFunc(data0, mARRAYSIZE(data0), data1, mARRAYSIZE(data1)));
  }

  {
    const size_t data0[] = { 1, 1 };
    const size_t data1[] = { 0, 1 };
    mTEST_ASSERT_FALSE(cmpFunc(data0, mARRAYSIZE(data0), data1, mARRAYSIZE(data1)));
  }

  {
    const size_t data0[] = { x, x, 1, 1 };
    const size_t data1[] = { x, x, 0, 1 };
    mTEST_ASSERT_FALSE(cmpFunc(data0, mARRAYSIZE(data0), data1, mARRAYSIZE(data1)));
  }

  {
    const size_t data0[] = { 1, 1, x, 1 };
    const size_t data1[] = { 1, 0, x, 1 };
    mTEST_ASSERT_FALSE(cmpFunc(data0, mARRAYSIZE(data0), data1, mARRAYSIZE(data1)));
  }

  {
    const size_t data0[] = { 1, 1, x, 1, 0, 0, 0, 0, 0 };
    const size_t data1[] = { 1, 1, x, 1 };
    mTEST_ASSERT_TRUE(cmpFunc(data0, mARRAYSIZE(data0), data1, mARRAYSIZE(data1)));
  }

  {
    const size_t data0[] = { 1, 1, x, 1, 0, 0, 0, 1, 0 };
    const size_t data1[] = { 1, 1, x, 1 };
    mTEST_ASSERT_FALSE(cmpFunc(data0, mARRAYSIZE(data0), data1, mARRAYSIZE(data1)));
  }

  {
    const size_t data0[] = { 1, 1, x, 1, 0, 0, 0, 1 };
    const size_t data1[] = { 1, 1, x, 1, 0, 0, 0, 0, 0 };
    mTEST_ASSERT_FALSE(cmpFunc(data0, mARRAYSIZE(data0), data1, mARRAYSIZE(data1)));
  }

  {
    const size_t data0[] = { 1, 1, x, 1, 0, 0, 0, 1, x, 2 };
    const size_t data1[] = { 1, 1, x, 1, x, x, x, 1, 0, 3 };
    mTEST_ASSERT_FALSE(cmpFunc(data0, mARRAYSIZE(data0), data1, mARRAYSIZE(data1)));
  }

  mTEST_ALLOCATOR_ZERO_CHECK();
}
