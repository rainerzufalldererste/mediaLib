#include "mTestLib.h"
#include "mChunkedArray.h"
#include "mQueue.h"

mTEST(mChunkedArray, TestCreate)
{
  mTEST_ALLOCATOR_SETUP();

  mTEST_ASSERT_EQUAL(mR_ArgumentNull, mChunkedArray_Create((mPtr<mChunkedArray<size_t>> *)nullptr, pAllocator));
  mPtr<mChunkedArray<mDummyDestructible>> chunkedArray;
  mDEFER_CALL(&chunkedArray, mChunkedArray_Destroy);
  mTEST_ASSERT_EQUAL(mR_ArgumentOutOfBounds, mChunkedArray_Create(&chunkedArray, pAllocator, 0));
  mTEST_ASSERT_SUCCESS(mChunkedArray_Create(&chunkedArray, pAllocator));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mChunkedArray, TestPushBack)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mChunkedArray<size_t>> chunkedArray;
  mDEFER_CALL(&chunkedArray, mChunkedArray_Destroy);
  mTEST_ASSERT_SUCCESS(mChunkedArray_Create(&chunkedArray, pAllocator));

  mTEST_ASSERT_EQUAL(mR_ArgumentNull, mChunkedArray_PushBack(chunkedArray, (size_t *)nullptr));

  const size_t maxCount = 1024;
  size_t count = (size_t)-1;

  for (size_t i = 0; i < maxCount; ++i)
  {
    mTEST_ASSERT_SUCCESS(mChunkedArray_GetCount(chunkedArray, &count));
    mTEST_ASSERT_EQUAL(i, count);
    mTEST_ASSERT_SUCCESS(mChunkedArray_PushBack(chunkedArray, &i));
  }

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mChunkedArray, TestPushBackPointerAt)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mChunkedArray<mDummyDestructible>> chunkedArray;
  mDEFER_CALL(&chunkedArray, mChunkedArray_Destroy);
  mTEST_ASSERT_SUCCESS(mChunkedArray_Create(&chunkedArray, pAllocator));

  mTEST_ASSERT_EQUAL(mR_ArgumentNull, mChunkedArray_PushBack(chunkedArray, (mDummyDestructible *)nullptr));

  const size_t maxCount = 1024;
  size_t count = (size_t)-1;

  for (size_t i = 0; i < maxCount; ++i)
  {
    mTEST_ASSERT_SUCCESS(mChunkedArray_GetCount(chunkedArray, &count));
    mTEST_ASSERT_EQUAL(i, count);
    mDummyDestructible dummy;
    mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy, pAllocator));
    *dummy.pData = i;
    mTEST_ASSERT_SUCCESS(mChunkedArray_PushBack(chunkedArray, &dummy));
  }

  for (size_t i = 0; i < maxCount; ++i)
  {
    mDummyDestructible *pDummy = nullptr;
    mTEST_ASSERT_SUCCESS(mChunkedArray_PointerAt(chunkedArray, i, &pDummy));
    mTEST_ASSERT_EQUAL(i, *pDummy->pData);
  }

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mChunkedArray, TestPopAt)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mChunkedArray<mDummyDestructible>> chunkedArray;
  mDEFER_CALL(&chunkedArray, mChunkedArray_Destroy);
  mTEST_ASSERT_SUCCESS(mChunkedArray_Create(&chunkedArray, pAllocator));

  const size_t maxCount = 1024;

  for (size_t i = 0; i < maxCount; ++i)
  {
    mDummyDestructible dummy;
    mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy, pAllocator));
    mTEST_ASSERT_SUCCESS(mChunkedArray_PushBack(chunkedArray, &dummy));
  }

  for (size_t i = 0; i < maxCount / 2; ++i) // remove all even entries.
  {
    mDummyDestructible dummy;
    mTEST_ASSERT_SUCCESS(mChunkedArray_PopAt(chunkedArray, i, &dummy));
    mTEST_ASSERT_EQUAL(dummy.index, i * 2);
    mTEST_ASSERT_SUCCESS(mDestruct(&dummy));
    
    for (size_t j = 0; j <= i; ++j)
    {
      mDummyDestructible *pDummy = nullptr;
      mTEST_ASSERT_SUCCESS(mChunkedArray_PointerAt(chunkedArray, j, &pDummy));
      mTEST_ASSERT_EQUAL(pDummy->index, j * 2 + 1);
    }
  }

  for (int64_t i = maxCount / 2 - 1; i >= 0; i -= 2)
  {
    mDummyDestructible dummy;
    mTEST_ASSERT_SUCCESS(mChunkedArray_PopAt(chunkedArray, i, &dummy));
    mTEST_ASSERT_EQUAL(dummy.index & 0b11, 0b11);
    mTEST_ASSERT_SUCCESS(mDestruct(&dummy));
  }

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mChunkedArray, TestFillEmptyBlockCount)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mChunkedArray<mDummyDestructible>> chunkedArray;
  mDEFER_CALL(&chunkedArray, mChunkedArray_Destroy);
  mTEST_ASSERT_SUCCESS(mChunkedArray_Create(&chunkedArray, pAllocator));

  const size_t dummyTestIndex = 0xFFAFAFAF;
  size_t index = 0;
  mDummyDestructible dummy;
  mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy, pAllocator));

  dummy.index = dummyTestIndex;

  mTEST_ASSERT_SUCCESS(mChunkedArray_Push(chunkedArray, &dummy, &index));

  dummy.index = 0;

  mTEST_ASSERT_SUCCESS(mChunkedArray_PopAt(chunkedArray, index, &dummy));

  mTEST_ASSERT_EQUAL(dummy.index, dummyTestIndex);

  mDummyDestructible *pDummy = nullptr;
  mTEST_ASSERT_EQUAL(mR_IndexOutOfBounds, mChunkedArray_PointerAt(chunkedArray, index, &pDummy));
  mTEST_ASSERT_EQUAL(pDummy, nullptr);

  mTEST_ASSERT_SUCCESS(mChunkedArray_Push(chunkedArray, &dummy, &index));
  mTEST_ASSERT_SUCCESS(mChunkedArray_PointerAt(chunkedArray, index, &pDummy));
  mTEST_ASSERT_NOT_EQUAL(pDummy, nullptr);
  mTEST_ASSERT_EQUAL(pDummy->index, dummyTestIndex);

  mTEST_ASSERT_SUCCESS(mChunkedArray_PopAt(chunkedArray, index, &dummy));
  pDummy = nullptr;

  mTEST_ASSERT_EQUAL(mR_IndexOutOfBounds, mChunkedArray_PointerAt(chunkedArray, index, &pDummy));
  mTEST_ASSERT_EQUAL(pDummy, nullptr);

  mTEST_ASSERT_SUCCESS(mChunkedArray_PushBack(chunkedArray, &dummy));
  index = chunkedArray->itemCount - 1;
  mTEST_ASSERT_SUCCESS(mChunkedArray_PointerAt(chunkedArray, index, &pDummy));
  mTEST_ASSERT_NOT_EQUAL(pDummy, nullptr);
  mTEST_ASSERT_EQUAL(pDummy->index, dummyTestIndex);

  mTEST_ASSERT_SUCCESS(mChunkedArray_PopAt(chunkedArray, index, &dummy));
  pDummy = nullptr;

  mTEST_ASSERT_EQUAL(mR_IndexOutOfBounds, mChunkedArray_PointerAt(chunkedArray, index, &pDummy));
  mTEST_ASSERT_EQUAL(pDummy, nullptr);

  mTEST_ASSERT_SUCCESS(mChunkedArray_PushBack(chunkedArray, std::move(dummy)));
  index = chunkedArray->itemCount - 1;
  mTEST_ASSERT_SUCCESS(mChunkedArray_PointerAt(chunkedArray, index, &pDummy));
  mTEST_ASSERT_NOT_EQUAL(pDummy, nullptr);
  mTEST_ASSERT_EQUAL(pDummy->index, dummyTestIndex);
  mTEST_ASSERT_EQUAL(chunkedArray->blockSize, chunkedArray->itemCapacity);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mChunkedArray, TestGenericUsage)
{
  mTEST_ALLOCATOR_SETUP();

  {
    mPtr<mChunkedArray<size_t>> chunkedArray;
    mDEFER_CALL(&chunkedArray, mChunkedArray_Destroy);
    mTEST_ASSERT_SUCCESS(mChunkedArray_Create(&chunkedArray, pAllocator));

    constexpr size_t size = 128;

    for (size_t i = 0; i < size; i++)
      mTEST_ASSERT_SUCCESS(mChunkedArray_PushBack(chunkedArray, &i));

    size_t data;
    size_t count = size;

    mTEST_ASSERT_SUCCESS(mChunkedArray_GetCount(chunkedArray, &data));
    mTEST_ASSERT_EQUAL(data, size);

    for (size_t i = 0; i < count; i += 3)
    {
      count--;
      mTEST_ASSERT_SUCCESS(mChunkedArray_PopAt(chunkedArray, i, &data));

      mTEST_ASSERT_SUCCESS(mChunkedArray_GetCount(chunkedArray, &data));
      mTEST_ASSERT_EQUAL(data, count);
    }

    mTEST_ASSERT_SUCCESS(mChunkedArray_GetCount(chunkedArray, &data));
    mTEST_ASSERT_EQUAL(data, count);

    for (size_t i = count; i < size; i++)
    {
      size_t index;
      mTEST_ASSERT_SUCCESS(mChunkedArray_Push(chunkedArray, &i, &index));
      mTEST_ASSERT_TRUE(index < size);
    }
    
    mTEST_ASSERT_SUCCESS(mChunkedArray_GetCount(chunkedArray, &data));
    mTEST_ASSERT_EQUAL(data, size);
    
    count = size;
    
    for (size_t i = 0; i < count; i += 3)
    {
      count--;
      mTEST_ASSERT_SUCCESS(mChunkedArray_PopAt(chunkedArray, i, &data));
    }
    
    mTEST_ASSERT_SUCCESS(mChunkedArray_GetCount(chunkedArray, &data));
    mTEST_ASSERT_EQUAL(data, count);
    
    for (size_t i = count; i < size; i++)
      mTEST_ASSERT_SUCCESS(mChunkedArray_PushBack(chunkedArray, &i));
    
    mTEST_ASSERT_SUCCESS(mChunkedArray_GetCount(chunkedArray, &data));
    mTEST_ASSERT_EQUAL(data, size);
  }

  {
    mPtr<mChunkedArray<mString>> chunkedArray;
    mDEFER_CALL(&chunkedArray, mChunkedArray_Destroy);
    mTEST_ASSERT_SUCCESS(mChunkedArray_Create(&chunkedArray, pAllocator));

    constexpr size_t size = 128;

    for (size_t i = 0; i < size; i++)
    {
      mString data;
      mTEST_ASSERT_SUCCESS(mString_Format(&data, pAllocator, i));
      mTEST_ASSERT_SUCCESS(mChunkedArray_PushBack(chunkedArray, &data));
    }

    mString data;
    size_t count = size;

    for (size_t i = 0; i < count; i += 3)
    {
      count--;
      mTEST_ASSERT_SUCCESS(mChunkedArray_PopAt(chunkedArray, i, &data));
    }

    size_t expectedCount = 0;
    mTEST_ASSERT_SUCCESS(mChunkedArray_GetCount(chunkedArray, &expectedCount));
    mTEST_ASSERT_EQUAL(expectedCount, count);

    for (size_t i = count; i < size; i++)
    {
      mTEST_ASSERT_SUCCESS(mString_Format(&data, pAllocator, i));

      size_t index;
      mTEST_ASSERT_SUCCESS(mChunkedArray_Push(chunkedArray, &data, &index));
      mTEST_ASSERT_TRUE(index < size);

      mString *pData = nullptr;
      mTEST_ASSERT_SUCCESS(mChunkedArray_PointerAt(chunkedArray, index, &pData));

      mTEST_ASSERT_EQUAL(*pData, data);
    }

    mTEST_ASSERT_SUCCESS(mChunkedArray_GetCount(chunkedArray, &expectedCount));
    mTEST_ASSERT_EQUAL(expectedCount, size);

    count = size;

    for (size_t i = 0; i < count; i += 3)
    {
      count--;
      mTEST_ASSERT_SUCCESS(mChunkedArray_PopAt(chunkedArray, i, &data));
    }

    mTEST_ASSERT_SUCCESS(mChunkedArray_GetCount(chunkedArray, &expectedCount));
    mTEST_ASSERT_EQUAL(expectedCount, count);

    for (size_t i = count; i < size; i++)
    {
      mTEST_ASSERT_SUCCESS(mString_Format(&data, pAllocator, i));

      mTEST_ASSERT_SUCCESS(mChunkedArray_PushBack(chunkedArray, &data));
      mTEST_ASSERT_SUCCESS(mChunkedArray_GetCount(chunkedArray, &expectedCount));

      mString *pData = nullptr;
      mTEST_ASSERT_SUCCESS(mChunkedArray_PointerAt(chunkedArray, expectedCount - 1, &pData));

      mTEST_ASSERT_EQUAL(*pData, data);
    }

    mTEST_ASSERT_SUCCESS(mChunkedArray_GetCount(chunkedArray, &expectedCount));
    mTEST_ASSERT_EQUAL(expectedCount, size);
  }

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mChunkedArray, TestClear)
{
  mTEST_ALLOCATOR_SETUP();

  {
    mPtr<mChunkedArray<size_t>> chunkedArray;
    mDEFER_CALL(&chunkedArray, mChunkedArray_Destroy);
    mTEST_ASSERT_SUCCESS(mChunkedArray_Create(&chunkedArray, pAllocator));

    constexpr size_t size = 128;

    for (size_t i = 0; i < 4; i++)
    {
      for (size_t j = 0; j < size; j++)
        mTEST_ASSERT_SUCCESS(mChunkedArray_PushBack(chunkedArray, &j));

      size_t data;
      size_t count = size;

      mTEST_ASSERT_SUCCESS(mChunkedArray_GetCount(chunkedArray, &data));
      mTEST_ASSERT_EQUAL(data, size);

      for (size_t j = 0; j < count; j += 3)
      {
        count--;
        mTEST_ASSERT_SUCCESS(mChunkedArray_PopAt(chunkedArray, j, &data));

        mTEST_ASSERT_SUCCESS(mChunkedArray_GetCount(chunkedArray, &data));
        mTEST_ASSERT_EQUAL(data, count);
      }

      mTEST_ASSERT_SUCCESS(mChunkedArray_GetCount(chunkedArray, &data));
      mTEST_ASSERT_EQUAL(data, count);

      for (size_t j = count; j < size; j++)
      {
        size_t index;
        mTEST_ASSERT_SUCCESS(mChunkedArray_Push(chunkedArray, &j, &index));
        mTEST_ASSERT_TRUE(index < size);
      }

      mTEST_ASSERT_SUCCESS(mChunkedArray_GetCount(chunkedArray, &data));
      mTEST_ASSERT_EQUAL(data, size);

      count = size;

      for (size_t j = 0; j < count; j += 3)
      {
        count--;
        mTEST_ASSERT_SUCCESS(mChunkedArray_PopAt(chunkedArray, j, &data));
      }

      mTEST_ASSERT_SUCCESS(mChunkedArray_GetCount(chunkedArray, &data));
      mTEST_ASSERT_EQUAL(data, count);

      for (size_t j = count; j < size; j++)
        mTEST_ASSERT_SUCCESS(mChunkedArray_PushBack(chunkedArray, &j));

      mTEST_ASSERT_SUCCESS(mChunkedArray_GetCount(chunkedArray, &data));
      mTEST_ASSERT_EQUAL(data, size);

      mTEST_ASSERT_SUCCESS(mChunkedArray_Clear(chunkedArray));
      mTEST_ASSERT_SUCCESS(mChunkedArray_Clear(chunkedArray));
    }
  }
  
  {
    mPtr<mChunkedArray<mString>> chunkedArray;
    mDEFER_CALL(&chunkedArray, mChunkedArray_Destroy);
    mTEST_ASSERT_SUCCESS(mChunkedArray_Create(&chunkedArray, pAllocator));

    mTEST_ASSERT_SUCCESS(mChunkedArray_Clear(chunkedArray));

    constexpr size_t size = 128;

    for (size_t i = 0; i < 4; i++)
    {
      for (size_t j = 0; j < size; j++)
      {
        mString data;
        mTEST_ASSERT_SUCCESS(mString_Format(&data, pAllocator, j));
        mTEST_ASSERT_SUCCESS(mChunkedArray_PushBack(chunkedArray, &data));
      }

      mString data;
      size_t count = size;

      for (size_t j = 0; j < count; j += 3)
      {
        count--;
        mTEST_ASSERT_SUCCESS(mChunkedArray_PopAt(chunkedArray, j, &data));
      }

      size_t expectedCount = 0;
      mTEST_ASSERT_SUCCESS(mChunkedArray_GetCount(chunkedArray, &expectedCount));
      mTEST_ASSERT_EQUAL(expectedCount, count);

      for (size_t j = count; j < size; j++)
      {
        mTEST_ASSERT_SUCCESS(mString_Format(&data, pAllocator, j));

        size_t index;
        mTEST_ASSERT_SUCCESS(mChunkedArray_Push(chunkedArray, &data, &index));
        mTEST_ASSERT_TRUE(index < size);

        mString *pData = nullptr;
        mTEST_ASSERT_SUCCESS(mChunkedArray_PointerAt(chunkedArray, index, &pData));

        mTEST_ASSERT_EQUAL(*pData, data);
      }

      mTEST_ASSERT_SUCCESS(mChunkedArray_GetCount(chunkedArray, &expectedCount));
      mTEST_ASSERT_EQUAL(expectedCount, size);

      count = size;

      for (size_t j = 0; j < count; j += 3)
      {
        count--;
        mTEST_ASSERT_SUCCESS(mChunkedArray_PopAt(chunkedArray, j, &data));
      }

      mTEST_ASSERT_SUCCESS(mChunkedArray_GetCount(chunkedArray, &expectedCount));
      mTEST_ASSERT_EQUAL(expectedCount, count);

      for (size_t j = count; j < size; j++)
      {
        mTEST_ASSERT_SUCCESS(mString_Format(&data, pAllocator, j));

        mTEST_ASSERT_SUCCESS(mChunkedArray_PushBack(chunkedArray, &data));
        mTEST_ASSERT_SUCCESS(mChunkedArray_GetCount(chunkedArray, &expectedCount));

        mString *pData = nullptr;
        mTEST_ASSERT_SUCCESS(mChunkedArray_PointerAt(chunkedArray, expectedCount - 1, &pData));

        mTEST_ASSERT_EQUAL(*pData, data);
      }

      mTEST_ASSERT_SUCCESS(mChunkedArray_GetCount(chunkedArray, &expectedCount));
      mTEST_ASSERT_EQUAL(expectedCount, size);

      mTEST_ASSERT_SUCCESS(mChunkedArray_Clear(chunkedArray));
      mTEST_ASSERT_SUCCESS(mChunkedArray_Clear(chunkedArray));
    }
  }

  mTEST_ALLOCATOR_ZERO_CHECK();
}
