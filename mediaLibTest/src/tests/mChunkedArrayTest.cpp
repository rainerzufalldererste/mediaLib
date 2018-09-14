// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "mTestLib.h"
#include "mChunkedArray.h"
#include "mQueue.h"

mTEST(mChunkedArray, TestCreate)
{
  mTEST_ALLOCATOR_SETUP();

  mTEST_ASSERT_EQUAL(mR_ArgumentNull, mChunkedArray_Create((mPtr<mChunkedArray<size_t>> *)nullptr, pAllocator));
  mPtr<mChunkedArray<mDummyDestructible>> chunkedArray;
  mDEFER_DESTRUCTION(&chunkedArray, mChunkedArray_Destroy);
  mTEST_ASSERT_EQUAL(mR_ArgumentOutOfBounds, mChunkedArray_Create(&chunkedArray, pAllocator, 0));
  mTEST_ASSERT_SUCCESS(mChunkedArray_Create(&chunkedArray, pAllocator));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mChunkedArray, TestPushBack)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mChunkedArray<size_t>> chunkedArray;
  mDEFER_DESTRUCTION(&chunkedArray, mChunkedArray_Destroy);
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
  mDEFER_DESTRUCTION(&chunkedArray, mChunkedArray_Destroy);
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
    //*dummy.pData = i;
    mTEST_ASSERT_SUCCESS(mChunkedArray_PushBack(chunkedArray, &dummy));
  }

  for (size_t i = 0; i < maxCount; ++i)
  {
    mDummyDestructible *pDummy = nullptr;
    mTEST_ASSERT_SUCCESS(mChunkedArray_PointerAt(chunkedArray, i, &pDummy));
    //mTEST_ASSERT_EQUAL(i, *pDummy->pData);
  }

  mTEST_ALLOCATOR_ZERO_CHECK();
}
