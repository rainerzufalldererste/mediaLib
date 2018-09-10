// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "mTestLib.h"
#include "mQueue.h"
#include "mHash.h"

mTEST(mQueue, TestCreate)
{
  mAllocator testAllocator;
  mTEST_ASSERT_SUCCESS(mTestAllocator_Create(&testAllocator));
  mDEFER_DESTRUCTION(&testAllocator, mAllocator_Destroy);
  
  {
    mTEST_ASSERT_EQUAL(mR_ArgumentNull, mQueue_Create((mPtr<mQueue<size_t>> *)nullptr, &testAllocator));

    mPtr<mQueue<size_t>> numberQueue = nullptr;
    mDEFER_DESTRUCTION(&numberQueue, mQueue_Destroy);
    mTEST_ASSERT_SUCCESS(mQueue_Create(&numberQueue, &testAllocator));

    mTEST_ASSERT_NOT_EQUAL(numberQueue, nullptr);

    // Destroys the old one.
    mTEST_ASSERT_SUCCESS(mQueue_Create(&numberQueue, &testAllocator));
  }

  size_t allocationCount = (size_t)-1;
  mTEST_ASSERT_SUCCESS(mTestAllocator_GetCount(&testAllocator, &allocationCount));
  mTEST_ASSERT_EQUAL(allocationCount, 0);
}

mTEST(mQueue, TestPushBackPopFront)
{
  mAllocator testAllocator;
  mTEST_ASSERT_SUCCESS(mTestAllocator_Create(&testAllocator));
  mDEFER_DESTRUCTION(&testAllocator, mAllocator_Destroy);

  {
    size_t count = (size_t)-1;
    size_t value;
    mPtr<mQueue<size_t>> numberQueue = nullptr;
    mDEFER_DESTRUCTION(&numberQueue, mQueue_Destroy);

    mTEST_ASSERT_EQUAL(mR_ArgumentNull, mQueue_PopFront(numberQueue, &value));
    mTEST_ASSERT_EQUAL(mR_ArgumentNull, mQueue_PushBack(numberQueue, 1LLU));
    mTEST_ASSERT_SUCCESS(mQueue_Create(&numberQueue, &testAllocator));
    mTEST_ASSERT_EQUAL(mR_ArgumentNull, mQueue_PushBack(numberQueue, (size_t *)nullptr));
    mTEST_ASSERT_EQUAL(mR_ArgumentNull, mQueue_PopFront(numberQueue, (size_t *)nullptr));

    for (size_t i = 0; i < 20; i++)
    {
      mTEST_ASSERT_SUCCESS(mQueue_GetCount(numberQueue, &count));
      mTEST_ASSERT_EQUAL(i, count);
      mTEST_ASSERT_SUCCESS(mQueue_PushBack(numberQueue, i));
    }

    for (size_t i = 0; i < 20; i++)
    {
      mTEST_ASSERT_SUCCESS(mQueue_GetCount(numberQueue, &count));
      mTEST_ASSERT_EQUAL(20 - i, count);
      mTEST_ASSERT_SUCCESS(mQueue_PopFront(numberQueue, &value));
      mTEST_ASSERT_EQUAL(i, value);
    }

    for (size_t i = 20; i < 40; i++)
    {
      mTEST_ASSERT_SUCCESS(mQueue_GetCount(numberQueue, &count));
      mTEST_ASSERT_EQUAL(i - 20, count);
      mTEST_ASSERT_SUCCESS(mQueue_PushBack(numberQueue, &i));
    }

    for (size_t i = 20; i < 40; i++)
    {
      mTEST_ASSERT_SUCCESS(mQueue_GetCount(numberQueue, &count));
      mTEST_ASSERT_EQUAL(40 - i, count);
      mTEST_ASSERT_SUCCESS(mQueue_PopFront(numberQueue, &value));
      mTEST_ASSERT_EQUAL(i, value);
    }
  }

  size_t allocationCount = (size_t)-1;
  mTEST_ASSERT_SUCCESS(mTestAllocator_GetCount(&testAllocator, &allocationCount));
  mTEST_ASSERT_EQUAL(allocationCount, 0);
}

mTEST(mQueue, TestPushFrontPopBack)
{
  mAllocator testAllocator;
  mTEST_ASSERT_SUCCESS(mTestAllocator_Create(&testAllocator));
  mDEFER_DESTRUCTION(&testAllocator, mAllocator_Destroy);

  {
    size_t count = (size_t)-1;
    size_t value;
    mPtr<mQueue<size_t>> numberQueue = nullptr;
    mDEFER_DESTRUCTION(&numberQueue, mQueue_Destroy);

    mTEST_ASSERT_EQUAL(mR_ArgumentNull, mQueue_PushFront(numberQueue, 1LLU));
    mTEST_ASSERT_EQUAL(mR_ArgumentNull, mQueue_PopBack(numberQueue, &value));
    mTEST_ASSERT_SUCCESS(mQueue_Create(&numberQueue, &testAllocator));
    mTEST_ASSERT_EQUAL(mR_ArgumentNull, mQueue_PushFront(numberQueue, (size_t *)nullptr));
    mTEST_ASSERT_EQUAL(mR_ArgumentNull, mQueue_PopBack(numberQueue, (size_t *)nullptr));

    for (size_t i = 0; i < 20; i++)
    {
      mTEST_ASSERT_SUCCESS(mQueue_GetCount(numberQueue, &count));
      mTEST_ASSERT_EQUAL(i, count);
      mTEST_ASSERT_SUCCESS(mQueue_PushFront(numberQueue, i));
    }

    for (size_t i = 0; i < 20; i++)
    {
      mTEST_ASSERT_SUCCESS(mQueue_GetCount(numberQueue, &count));
      mTEST_ASSERT_EQUAL(20 - i, count);
      mTEST_ASSERT_SUCCESS(mQueue_PopBack(numberQueue, &value));
      mTEST_ASSERT_EQUAL(i, value);
    }

    for (size_t i = 20; i < 40; i++)
    {
      mTEST_ASSERT_SUCCESS(mQueue_GetCount(numberQueue, &count));
      mTEST_ASSERT_EQUAL(i - 20, count);
      mTEST_ASSERT_SUCCESS(mQueue_PushFront(numberQueue, &i));
    }

    for (size_t i = 20; i < 40; i++)
    {
      mTEST_ASSERT_SUCCESS(mQueue_GetCount(numberQueue, &count));
      mTEST_ASSERT_EQUAL(40 - i, count);
      mTEST_ASSERT_SUCCESS(mQueue_PopBack(numberQueue, &value));
      mTEST_ASSERT_EQUAL(i, value);
    }
  }

  size_t allocationCount = (size_t)-1;
  mTEST_ASSERT_SUCCESS(mTestAllocator_GetCount(&testAllocator, &allocationCount));
  mTEST_ASSERT_EQUAL(allocationCount, 0);
}

mTEST(mQueue, TestPushPop)
{
  mAllocator testAllocator;
  mTEST_ASSERT_SUCCESS(mTestAllocator_Create(&testAllocator));
  mDEFER_DESTRUCTION(&testAllocator, mAllocator_Destroy);

  {
    std::vector<size_t> comparisonVector;
    mPtr<mQueue<size_t>> numberQueue = nullptr;
    mDEFER_DESTRUCTION(&numberQueue, mQueue_Destroy);
    mTEST_ASSERT_SUCCESS(mQueue_Create(&numberQueue, &testAllocator));

    size_t count = (size_t)-1;
    size_t seed = 0;

    for (size_t i = 0; i < 100; i++)
    {
      seed += i + seed * 0xDEADF00D;

      size_t operation = seed % 5;
      size_t value;

      mTEST_ASSERT_SUCCESS(mQueue_GetCount(numberQueue, &count));
      mTEST_ASSERT_EQUAL(count, comparisonVector.size());

      switch (operation)
      {
      case 0:
        mTEST_ASSERT_SUCCESS(mQueue_PushBack(numberQueue, seed));
        comparisonVector.push_back(seed);
        break;

      case 1:
        if (count > 0)
        {
          mTEST_ASSERT_SUCCESS(mQueue_PopBack(numberQueue, &value));
          comparisonVector.pop_back();
        }
        else
        {
          mTEST_ASSERT_EQUAL(mR_IndexOutOfBounds, mQueue_PopBack(numberQueue, &value));
        }

        break;

      case 2:
        if (count > 0)
        {
          mTEST_ASSERT_SUCCESS(mQueue_PopFront(numberQueue, &value));
          comparisonVector.erase(comparisonVector.begin());
        }
        else
        {
          mTEST_ASSERT_EQUAL(mR_IndexOutOfBounds, mQueue_PopFront(numberQueue, &value));
        }

        break;

      case 3:
        mTEST_ASSERT_SUCCESS(mQueue_PushFront(numberQueue, seed));
        comparisonVector.insert(comparisonVector.begin(), seed);
        break;

      case 4:
        for (size_t j = 0; j < count; j++)
        {
          mTEST_ASSERT_SUCCESS(mQueue_PeekFront(numberQueue, &value));
          mTEST_ASSERT_EQUAL(comparisonVector[0], value);
          mTEST_ASSERT_SUCCESS(mQueue_PeekBack(numberQueue, &value));
          mTEST_ASSERT_EQUAL(comparisonVector[comparisonVector.size() - 1], value);
          mTEST_ASSERT_SUCCESS(mQueue_PopBack(numberQueue, &value));
          mTEST_ASSERT_EQUAL(comparisonVector[comparisonVector.size() - 1], value);
          comparisonVector.pop_back();
        }

        break;
      }
    }
  }

  size_t allocationCount = (size_t)-1;
  mTEST_ASSERT_SUCCESS(mTestAllocator_GetCount(&testAllocator, &allocationCount));
  mTEST_ASSERT_EQUAL(allocationCount, 0);
}
