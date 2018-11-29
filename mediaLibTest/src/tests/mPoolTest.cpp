#include "mTestLib.h"
#include "mPool.h"

mTEST(mPool, TestCreate)
{
  mTEST_ALLOCATOR_SETUP();

  mTEST_ASSERT_EQUAL(mR_ArgumentNull, mPool_Create((mPtr<mPool<mDummyDestructible>> *)nullptr, pAllocator));

  mPtr<mPool<mDummyDestructible>> pool;
  mDEFER_CALL(&pool, mPool_Destroy);
  mTEST_ASSERT_SUCCESS(mPool_Create(&pool, pAllocator));

  mTEST_ASSERT_NOT_EQUAL(pool, nullptr);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mPool, TestAdd)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mPool<mDummyDestructible>> pool;
  mDEFER_CALL(&pool, mPool_Destroy);
  mTEST_ASSERT_SUCCESS(mPool_Create(&pool, pAllocator));

  size_t count = (size_t)-1;
  mTEST_ASSERT_SUCCESS(mPool_GetCount(pool, &count));
  mTEST_ASSERT_EQUAL(count, 0);

  mDummyDestructible dummy0;
  mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy0, pAllocator));

  size_t index0;
  mTEST_ASSERT_SUCCESS(mPool_Add(pool, &dummy0, &index0));

  mTEST_ASSERT_SUCCESS(mPool_GetCount(pool, &count));
  mTEST_ASSERT_EQUAL(count, 1);

  mDummyDestructible dummy1;
  mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy1, pAllocator));

  size_t index1;
  mTEST_ASSERT_SUCCESS(mPool_Add(pool, &dummy1, &index1));

  mTEST_ASSERT_SUCCESS(mPool_GetCount(pool, &count));
  mTEST_ASSERT_EQUAL(count, 2);

  mTEST_ASSERT_NOT_EQUAL(index0, index1);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mPool, TestAddRemove)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mPool<mDummyDestructible>> pool;
  mDEFER_CALL(&pool, mPool_Destroy);
  mTEST_ASSERT_SUCCESS(mPool_Create(&pool, pAllocator));

  mDummyDestructible dummy0;
  mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy0, pAllocator));

  size_t index0;
  mTEST_ASSERT_SUCCESS(mPool_Add(pool, &dummy0, &index0));

  mDummyDestructible dummy1;
  mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy1, pAllocator));

  size_t index1;
  mTEST_ASSERT_SUCCESS(mPool_Add(pool, &dummy1, &index1));

  mTEST_ASSERT_SUCCESS(mPool_RemoveAt(pool, index0, &dummy0));
  mTEST_ASSERT_FALSE(dummy0.destructed);

  size_t count = (size_t)-1;
  mTEST_ASSERT_SUCCESS(mPool_GetCount(pool, &count));
  mTEST_ASSERT_EQUAL(count, 1);

  // dummy0 has not been destructed yet.
  mTEST_ASSERT_SUCCESS(mPool_Add(pool, &dummy0, &index0));

  mTEST_ASSERT_SUCCESS(mPool_GetCount(pool, &count));
  mTEST_ASSERT_EQUAL(count, 2);

  mDummyDestructible dummy2;
  mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy2, pAllocator));

  size_t index2;
  mTEST_ASSERT_SUCCESS(mPool_Add(pool, &dummy2, &index2));

  mTEST_ASSERT_SUCCESS(mPool_GetCount(pool, &count));
  mTEST_ASSERT_EQUAL(count, 3);

  mTEST_ASSERT_SUCCESS(mPool_RemoveAt(pool, index0, &dummy0));
  mTEST_ASSERT_SUCCESS(mDestruct(&dummy0));

  mTEST_ASSERT_SUCCESS(mPool_RemoveAt(pool, index2, &dummy2));
  mTEST_ASSERT_SUCCESS(mDestruct(&dummy2));

  mTEST_ASSERT_SUCCESS(mPool_RemoveAt(pool, index1, &dummy1));
  mTEST_ASSERT_SUCCESS(mDestruct(&dummy1));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mPool, TestIterate)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mPool<size_t>> pool;
  mDEFER_CALL(&pool, mPool_Destroy);
  mTEST_ASSERT_SUCCESS(mPool_Create(&pool, pAllocator));

  const size_t testSize = 100;

  for (size_t i = 0; i < testSize; i++)
  {
    size_t index = 0;
    mTEST_ASSERT_SUCCESS(mPool_Add(pool, &i, &index));
    mTEST_ASSERT_EQUAL(i, index);
  }

  for (size_t i = 0; i < testSize; i += 2)
  {
    size_t element;
    mTEST_ASSERT_SUCCESS(mPool_RemoveAt(pool, i, &element));
    mTEST_ASSERT_EQUAL(element, i);
  }

  size_t count = 0;

  for (auto &&_index : pool->Iterate())
  {
    mTEST_ASSERT_EQUAL(_index.index, (count << 1) + 1);
    mTEST_ASSERT_EQUAL(*_index.pData, (count << 1) + 1);
    mTEST_ASSERT_EQUAL(*_index, (count << 1) + 1);
    ++count;
  }

  mTEST_ASSERT_EQUAL(count, testSize / 2);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mPool, TestForEach)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mPool<size_t>> pool;
  mDEFER_CALL(&pool, mPool_Destroy);
  mTEST_ASSERT_SUCCESS(mPool_Create(&pool, pAllocator));

  const size_t testSize = 100;

  for (size_t i = 0; i < testSize; i++)
  {
    size_t index = 0;
    mTEST_ASSERT_SUCCESS(mPool_Add(pool, &i, &index));
    mTEST_ASSERT_EQUAL(i, index);
  }

  for (size_t i = 0; i < testSize; i += 2)
  {
    size_t element;
    mTEST_ASSERT_SUCCESS(mPool_RemoveAt(pool, i, &element));
    mTEST_ASSERT_EQUAL(element, i);
  }

  size_t count = 0;

  const std::function<mResult (size_t *, const size_t)> &iterate = [&](size_t *pData, const size_t index)
  {
    mFUNCTION_SETUP();

    mERROR_IF(index != (count << 1) + 1, mR_Failure);
    mERROR_IF(*pData != (count << 1) + 1, mR_Failure);
    ++count;

    mRETURN_SUCCESS();
  };

  mTEST_ASSERT_SUCCESS(mPool_ForEach(pool, iterate));

  mTEST_ASSERT_EQUAL(count, testSize / 2);

  mTEST_ALLOCATOR_ZERO_CHECK();
}
