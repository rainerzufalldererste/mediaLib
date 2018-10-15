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
  mTEST_ASSERT_EQUAL(0, chunkedArray->blockCount);

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
