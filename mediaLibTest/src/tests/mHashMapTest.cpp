#include "mTestLib.h"
#include "mHashMap.h"

mTEST(mHashMap, TestCreate)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mHashMap<size_t, mDummyDestructible>> hashMap;
  mDEFER_CALL(&hashMap, mHashMap_Destroy);
  mTEST_ASSERT_EQUAL(mR_ArgumentNull, mHashMap_Create((mPtr<mHashMap<size_t, size_t>> *)nullptr, pAllocator, 1024));
  mTEST_ASSERT_SUCCESS(mHashMap_Create(&hashMap, pAllocator, 1024));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mHashMap, TestCleanup)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mHashMap<size_t, mDummyDestructible>> hashMap;
  mDEFER_CALL(&hashMap, mHashMap_Destroy);
  mTEST_ASSERT_SUCCESS(mHashMap_Create(&hashMap, pAllocator, 1024));

  for (size_t i = 0; i < 1024 * 8; i++)
  {
    mDummyDestructible dummy;
    mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy, pAllocator));
    mTEST_ASSERT_SUCCESS(mHashMap_Add(hashMap, i, &dummy));
  }

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mHashMap, TestGetValues)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mHashMap<size_t, mDummyDestructible>> hashMap;
  mDEFER_CALL(&hashMap, mHashMap_Destroy);
  mTEST_ASSERT_SUCCESS(mHashMap_Create(&hashMap, pAllocator, 128));

  const size_t maxCount = 1024;

  for (size_t i = 0; i < maxCount; i++)
  {
    mDummyDestructible dummy;
    mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy, pAllocator));
    mTEST_ASSERT_SUCCESS(mHashMap_Add(hashMap, i, &dummy));

    bool contains = false;
    mDummyDestructible *pDummy = nullptr;
    mTEST_ASSERT_SUCCESS(mHashMap_ContainsGetPointer(hashMap, i, &contains, &pDummy));
    mTEST_ASSERT_TRUE(contains);
    mTEST_ASSERT_EQUAL(pDummy->index, i);
  }

  for (size_t i = 0; i < maxCount; i++)
  {
    bool contains = false;
    mDummyDestructible *pDummy = nullptr;
    mTEST_ASSERT_SUCCESS(mHashMap_ContainsGetPointer(hashMap, i, &contains, &pDummy));
    mTEST_ASSERT_TRUE(contains);
    mTEST_ASSERT_EQUAL(pDummy->index, i);

    mTEST_ASSERT_SUCCESS(mHashMap_ContainsGetPointer(hashMap, i + maxCount, &contains, &pDummy));
    mTEST_ASSERT_FALSE(contains);
  }

  for (size_t i = 0; i < maxCount; i++)
  {
    mDummyDestructible dummy;
    mTEST_ASSERT_SUCCESS(mHashMap_Remove(hashMap, i, &dummy));
    mTEST_ASSERT_SUCCESS(mDestruct(&dummy));

    for (size_t j = 0; j < maxCount; j++)
    {
      bool contains = false;
      mDummyDestructible *pDummy = nullptr;
      mTEST_ASSERT_SUCCESS(mHashMap_ContainsGetPointer(hashMap, j, &contains, &pDummy));

      if (j > i)
      {
        mTEST_ASSERT_TRUE(contains);
        mTEST_ASSERT_EQUAL(pDummy->index, j);
      }
      else
      {
        mTEST_ASSERT_FALSE(contains);
      }
    }
  }

  mTEST_ALLOCATOR_ZERO_CHECK();
}
