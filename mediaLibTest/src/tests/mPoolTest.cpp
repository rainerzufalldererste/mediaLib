// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "mTestLib.h"
#include "mPool.h"

mTEST(mPool, TestCreate)
{
  mTEST_ALLOCATOR_SETUP();

  mTEST_ASSERT_EQUAL(mR_ArgumentNull, mPool_Create((mPtr<mPool<mDummyDestructible>> *)nullptr, pAllocator));

  mPtr<mPool<mDummyDestructible>> pool;
  mDEFER_DESTRUCTION(&pool, mPool_Destroy);
  mTEST_ASSERT_SUCCESS(mPool_Create(&pool, pAllocator));

  mTEST_ASSERT_NOT_EQUAL(pool, nullptr);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mPool, TestAdd)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mPool<mDummyDestructible>> pool;
  mDEFER_DESTRUCTION(&pool, mPool_Destroy);
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
  mDEFER_DESTRUCTION(&pool, mPool_Destroy);
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

  mTEST_ALLOCATOR_ZERO_CHECK();
}
