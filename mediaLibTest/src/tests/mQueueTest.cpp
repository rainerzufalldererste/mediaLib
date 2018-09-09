// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "mTestLib.h"
#include "mQueue.h"

mTEST(mQueue, TestCreate)
{
  mTEST_ASSERT_EQUAL(mR_ArgumentNull, mQueue_Create((mPtr<mQueue<size_t>> *)nullptr, nullptr));

  mPtr<mQueue<size_t>> numberQueue = nullptr;
  mDEFER_DESTRUCTION(&numberQueue, mQueue_Destroy);
  mTEST_ASSERT_SUCCESS(mQueue_Create(&numberQueue, nullptr));

  mTEST_ASSERT_NOT_EQUAL(numberQueue, nullptr);
}

mTEST(mQueue, TestPushBackPopFront)
{
  mAllocator testAllocator;
  mTEST_ASSERT_SUCCESS(mTestAllocator_Create(&testAllocator));
  mDEFER_DESTRUCTION(&testAllocator, mAllocator_Destroy);

  {
    mPtr<mQueue<size_t>> numberQueue = nullptr;
    mDEFER_DESTRUCTION(&numberQueue, mQueue_Destroy);

    mTEST_ASSERT_EQUAL(mR_ArgumentNull, mQueue_PushBack(numberQueue, 1LLU));
    mTEST_ASSERT_SUCCESS(mQueue_Create(&numberQueue, &testAllocator));
    mTEST_ASSERT_EQUAL(mR_ArgumentNull, mQueue_PushBack(numberQueue, (size_t *)nullptr));

    size_t count = (size_t)-1;
    size_t value;

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
