// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "mTestLib.h"
#include "mHashMap.h"

mTEST(mHashMap, TestCreate)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mHashMap<size_t, mDummyDestructible>> hashMap;
  mDEFER_DESTRUCTION(&hashMap, mHashMap_Destroy);
  mTEST_ASSERT_EQUAL(mR_ArgumentNull, mHashMap_Create((mPtr<mHashMap<size_t, size_t>> *)nullptr, pAllocator, 1024));
  mTEST_ASSERT_SUCCESS(mHashMap_Create(&hashMap, pAllocator, 1024));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mHashMap, TestCleanup)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mHashMap<size_t, mDummyDestructible>> hashMap;
  mDEFER_DESTRUCTION(&hashMap, mHashMap_Destroy);
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
  mDEFER_DESTRUCTION(&hashMap, mHashMap_Destroy);
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
