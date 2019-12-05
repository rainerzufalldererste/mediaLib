#include "mTestLib.h"
#include "mQueue.h"

mTEST(mQueue, TestCreate)
{
  mTEST_ALLOCATOR_SETUP();

  mTEST_ASSERT_EQUAL(mR_ArgumentNull, mQueue_Create((mPtr<mQueue<size_t>> *)nullptr, pAllocator));

  mPtr<mQueue<size_t>> numberQueue = nullptr;
  mDEFER_CALL(&numberQueue, mQueue_Destroy);
  mTEST_ASSERT_SUCCESS(mQueue_Create(&numberQueue, pAllocator));

  mTEST_ASSERT_NOT_EQUAL(numberQueue, nullptr);

  // Destroys the old one.
  mTEST_ASSERT_SUCCESS(mQueue_Create(&numberQueue, pAllocator));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mQueue, TestPushBackPopFront)
{
  mTEST_ALLOCATOR_SETUP();

  size_t count = (size_t)-1;
  size_t value;
  mPtr<mQueue<size_t>> numberQueue = nullptr;
  mDEFER_CALL(&numberQueue, mQueue_Destroy);

  mTEST_ASSERT_EQUAL(mR_ArgumentNull, mQueue_PopFront(numberQueue, &value));
  mTEST_ASSERT_EQUAL(mR_ArgumentNull, mQueue_PushBack(numberQueue, 1LLU));
  mTEST_ASSERT_SUCCESS(mQueue_Create(&numberQueue, pAllocator));
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

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mQueue, TestPushFrontPopBack)
{
  mTEST_ALLOCATOR_SETUP();

  size_t count = (size_t)-1;
  size_t value;
  mPtr<mQueue<size_t>> numberQueue = nullptr;
  mDEFER_CALL(&numberQueue, mQueue_Destroy);

  mTEST_ASSERT_EQUAL(mR_ArgumentNull, mQueue_PushFront(numberQueue, 1LLU));
  mTEST_ASSERT_EQUAL(mR_ArgumentNull, mQueue_PopBack(numberQueue, &value));
  mTEST_ASSERT_SUCCESS(mQueue_Create(&numberQueue, pAllocator));
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

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mQueue, TestPushPop)
{
  mTEST_ALLOCATOR_SETUP();

  std::vector<size_t> comparisonVector;
  mPtr<mQueue<size_t>> numberQueue = nullptr;
  mDEFER_CALL(&numberQueue, mQueue_Destroy);
  mTEST_ASSERT_SUCCESS(mQueue_Create(&numberQueue, pAllocator));

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

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mQueue, TestClear)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mQueue<mDummyDestructible>> queue;
  mDEFER_CALL(&queue, mQueue_Destroy);
  mTEST_ASSERT_SUCCESS(mQueue_Create(&queue, pAllocator));

  for (size_t i = 0; i < 1024; i++)
  {
    mDummyDestructible dummy;
    mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy, pAllocator));
    mTEST_ASSERT_SUCCESS(mQueue_PushBack(queue, &dummy));
  }

  mTEST_ASSERT_SUCCESS(mQueue_Clear(queue));

  size_t count = (size_t)-1;
  mTEST_ASSERT_SUCCESS(mQueue_GetCount(queue, &count));
  mTEST_ASSERT_EQUAL(0, count);

  for (size_t i = 0; i < 1024; i++)
  {
    mDummyDestructible dummy;
    mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy, pAllocator));
    mTEST_ASSERT_SUCCESS(mQueue_PushBack(queue, &dummy));
  }

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mQueue, TestPopAt)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mQueue<mDummyDestructible>> queue;
  mDEFER_CALL(&queue, mQueue_Destroy);

  // Test for argument null error.
  {
    mDummyDestructible unused;
    mTEST_ASSERT_EQUAL(mR_ArgumentNull, mQueue_PopAt(queue, 0, &unused));
  }

  mTEST_ASSERT_SUCCESS(mQueue_Create(&queue, pAllocator));

  std::vector<size_t> comparisonVector;

  // Test for out of bounds error.
  {
    mDummyDestructible unused;
    mTEST_ASSERT_EQUAL(mR_IndexOutOfBounds, mQueue_PopAt(queue, 0, &unused));
  }

  const size_t maxCount = 128;
  mTEST_ASSERT_SUCCESS(mQueue_Reserve(queue, maxCount));

  int64_t number = 0;
  size_t count = 0;

  while (count++ < maxCount)
  {
    mDummyDestructible dummy;
    mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy, pAllocator));
    dummy.index = (size_t)(number + (int64_t)maxCount);

    if (number > 0)
    {
      mTEST_ASSERT_SUCCESS(mQueue_PushBack(queue, &dummy));
      comparisonVector.push_back(dummy.index);
    }
    else
    {
      mTEST_ASSERT_SUCCESS(mQueue_PushFront(queue, &dummy));
      comparisonVector.insert(comparisonVector.begin(), dummy.index);
    }

    number = -number;

    if (number >= 0)
      ++number;
  }

  // Test for argument null value (for the parameter).
  {
    mTEST_ASSERT_EQUAL(mR_ArgumentNull, mQueue_PopAt(queue, 0, (mDummyDestructible *)nullptr));
  }

  // Make sure test data is what we expect it to be.
  {
    size_t queueCount;
    mTEST_ASSERT_SUCCESS(mQueue_GetCount(queue, &queueCount));
    mTEST_ASSERT_EQUAL(queueCount, maxCount);

    for (size_t i = 0; i < maxCount - 1; i++)
    {
      mDummyDestructible *pDummy0;
      mTEST_ASSERT_SUCCESS(mQueue_PointerAt(queue, i, &pDummy0));

      mDummyDestructible *pDummy1;
      mTEST_ASSERT_SUCCESS(mQueue_PointerAt(queue, i + 1, &pDummy1));

      mTEST_ASSERT_TRUE(pDummy0->index < pDummy1->index);
      mTEST_ASSERT_EQUAL(pDummy0->index, pDummy1->index - 1);
      mTEST_ASSERT_EQUAL(pDummy0->index, comparisonVector[i]);
      mTEST_ASSERT_EQUAL(pDummy1->index, comparisonVector[i + 1]);
    }
  }

  // Start popping
  {
    // Pop around wrap.
    {
      size_t index = queue->size - queue->startIndex - 1;

      for (size_t i = 0; i < 5; i++)
      {
        mDummyDestructible dummy;
        mTEST_ASSERT_SUCCESS(mQueue_PopAt(queue, index + i, &dummy));

        const size_t comparisonNumber = comparisonVector[index + i];
        mTEST_ASSERT_EQUAL(comparisonNumber, dummy.index);
        mTEST_ASSERT_SUCCESS(mDestruct(&dummy));

        comparisonVector.erase(comparisonVector.begin() + (index + i));

        size_t queueCount;
        mTEST_ASSERT_SUCCESS(mQueue_GetCount(queue, &queueCount));
        mTEST_ASSERT_EQUAL(queueCount, comparisonVector.size());

        for (size_t j = 0; j < comparisonVector.size(); j++)
        {
          mDummyDestructible *pDummy;
          mTEST_ASSERT_SUCCESS(mQueue_PointerAt(queue, j, &pDummy));

          mTEST_ASSERT_EQUAL(pDummy->index, comparisonVector[j]);
        }
      }
    }

    {
      size_t remainingSize;
      mTEST_ASSERT_SUCCESS(mQueue_GetCount(queue, &remainingSize));

      while (remainingSize-- > 1)
      {
        mTEST_ASSERT_EQUAL(comparisonVector.size() - 1, remainingSize);

        mDummyDestructible dummy;
        mTEST_ASSERT_SUCCESS(mQueue_PopAt(queue, remainingSize - 1, &dummy)); // second last.

        const size_t comparisonNumber = comparisonVector[remainingSize - 1];
        mTEST_ASSERT_EQUAL(comparisonNumber, dummy.index);
        mTEST_ASSERT_SUCCESS(mDestruct(&dummy));

        comparisonVector.erase(comparisonVector.begin() + (remainingSize - 1));

        mTEST_ASSERT_SUCCESS(mQueue_GetCount(queue, &remainingSize));
        mTEST_ASSERT_EQUAL(remainingSize, comparisonVector.size());

        for (size_t j = 0; j < comparisonVector.size(); j++)
        {
          mDummyDestructible *pDummy;
          mTEST_ASSERT_SUCCESS(mQueue_PointerAt(queue, j, &pDummy));

          mTEST_ASSERT_EQUAL(pDummy->index, comparisonVector[j]);
        }
      }
    }
  }

  mTEST_ASSERT_SUCCESS(mQueue_Clear(queue));
  comparisonVector.clear();
  
  // Pop first & pop last
  {
    for (size_t i = 0; i < 3; i++)
    {
      mDummyDestructible dummy;
      mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy, pAllocator));
      mTEST_ASSERT_SUCCESS(mQueue_PushBack(queue, &dummy));
      comparisonVector.push_back(dummy.index);
    }

    // Pop first.
    {
      mDummyDestructible dummy;
      mTEST_ASSERT_SUCCESS(mQueue_PopAt(queue, 0, &dummy));

      const size_t comparisonNumber = comparisonVector[0];
      mTEST_ASSERT_EQUAL(comparisonNumber, dummy.index);
      mTEST_ASSERT_SUCCESS(mDestruct(&dummy));

      comparisonVector.erase(comparisonVector.begin());

      size_t queueCount;
      mTEST_ASSERT_SUCCESS(mQueue_GetCount(queue, &queueCount));
      mTEST_ASSERT_EQUAL(queueCount, comparisonVector.size());
    }

    // Pop last.
    {
      mDummyDestructible dummy;
      mTEST_ASSERT_SUCCESS(mQueue_PopAt(queue, comparisonVector.size() - 1, &dummy));

      const size_t comparisonNumber = comparisonVector[comparisonVector.size() - 1];
      mTEST_ASSERT_EQUAL(comparisonNumber, dummy.index);
      mTEST_ASSERT_SUCCESS(mDestruct(&dummy));

      comparisonVector.erase(comparisonVector.begin() + (comparisonVector.size() - 1));

      size_t queueCount;
      mTEST_ASSERT_SUCCESS(mQueue_GetCount(queue, &queueCount));
      mTEST_ASSERT_EQUAL(queueCount, comparisonVector.size());
    }

    mTEST_ASSERT_SUCCESS(mQueue_Clear(queue));
    comparisonVector.clear();
  }

  // Add data again.
  {
    for (size_t i = 0; i < maxCount; i++)
    {
      mDummyDestructible dummy;
      mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy, pAllocator));

      if ((i & 1) != 0 || (i & 8) != 0) // to get an odd distribution.
      {
        mTEST_ASSERT_SUCCESS(mQueue_PushBack(queue, &dummy));
        comparisonVector.push_back(dummy.index);
      }
      else
      {
        mTEST_ASSERT_SUCCESS(mQueue_PushFront(queue, &dummy));
        comparisonVector.insert(comparisonVector.begin(), dummy.index);
      }
    }
  }

  // Pop in a different way.
  {
    for (size_t i = 0; i < maxCount - 10; i++)
    {
      mDummyDestructible dummy;
      mTEST_ASSERT_SUCCESS(mQueue_PopAt(queue, 5, &dummy)); // second last.

      const size_t comparisonNumber = comparisonVector[5];
      mTEST_ASSERT_EQUAL(comparisonNumber, dummy.index);
      mTEST_ASSERT_SUCCESS(mDestruct(&dummy));

      comparisonVector.erase(comparisonVector.begin() + 5);

      size_t queueCount;
      mTEST_ASSERT_SUCCESS(mQueue_GetCount(queue, &queueCount));
      mTEST_ASSERT_EQUAL(queueCount, comparisonVector.size());

      for (size_t j = 0; j < comparisonVector.size(); j++)
      {
        mDummyDestructible *pDummy;
        mTEST_ASSERT_SUCCESS(mQueue_PointerAt(queue, j, &pDummy));

        mTEST_ASSERT_EQUAL(pDummy->index, comparisonVector[j]);
      }
    }
  }

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mQueue, TestContainCppClass)
{
  mTEST_ASSERT_EQUAL(0, mTestCppClass::GlobalCount());
  mTEST_ALLOCATOR_SETUP();

  {

    mPtr<mQueue<mTestCppClass>> queue = nullptr;
    mDEFER_CALL(&queue, mQueue_Destroy);

    {
      mTEST_ASSERT_SUCCESS(mQueue_Create(&queue, pAllocator));

      mTEST_ASSERT_SUCCESS(mQueue_PushFront(queue, mTestCppClass(1)));
      mTEST_ASSERT_SUCCESS(mQueue_PushFront(queue, mTestCppClass(2)));
      mTEST_ASSERT_SUCCESS(mQueue_PushFront(queue, mTestCppClass(3)));

      mTestCppClass t0;
      mTestCppClass t1;
      mTestCppClass t2;

      mTEST_ASSERT_SUCCESS(mQueue_PopFront(queue, &t0));
      mTEST_ASSERT_SUCCESS(mQueue_PopFront(queue, &t1));
      mTEST_ASSERT_SUCCESS(mQueue_PopFront(queue, &t2));
    }

    {
      mTEST_ASSERT_SUCCESS(mQueue_Create(&queue, pAllocator));

      for (size_t i = 0; i < 100; i++)
        mTEST_ASSERT_SUCCESS(mQueue_PushFront(queue, mTestCppClass(i)));
    }
  }

  mTEST_ASSERT_EQUAL(0, mTestCppClass::GlobalCount());
  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mQueue, TestContainLambda)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mQueue<std::function<void(void)>>> queue = nullptr;
  mDEFER_CALL(&queue, mQueue_Destroy);

  {
    mTEST_ASSERT_SUCCESS(mQueue_Create(&queue, pAllocator));

    size_t n0 = 0;
    size_t n1 = 0;
    size_t n2 = 0;

    mTEST_ASSERT_SUCCESS(mQueue_PushFront(queue, (std::function<void(void)>)[&]() { n0 = 1; }));
    mTEST_ASSERT_SUCCESS(mQueue_PushFront(queue, (std::function<void(void)>)[&]() { n1 = 1; }));
    mTEST_ASSERT_SUCCESS(mQueue_PushFront(queue, (std::function<void(void)>)[&]() { n2 = 1; }));

    std::function<void(void)> functionA;
    std::function<void(void)> functionB;
    std::function<void(void)> functionC;

    mTEST_ASSERT_SUCCESS(mQueue_PopFront(queue, &functionA));
    mTEST_ASSERT_SUCCESS(mQueue_PopFront(queue, &functionB));
    mTEST_ASSERT_SUCCESS(mQueue_PopFront(queue, &functionC));

    functionA();
    functionB();
    functionC();

    mTEST_ASSERT_EQUAL(n0, 1);
    mTEST_ASSERT_EQUAL(n1, 1);
    mTEST_ASSERT_EQUAL(n2, 1);
  }

  {
    mTEST_ASSERT_SUCCESS(mQueue_Create(&queue, pAllocator));

    for (size_t i = 0; i < 100; i++)
      mTEST_ASSERT_SUCCESS(mQueue_PushFront(queue, (std::function<void(void)>)[=]() {printf("%" PRIu64 "\n", i);}));
  }

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mQueue, TestIterateForward)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mQueue<int64_t>> numberQueue = nullptr;
  mDEFER_CALL(&numberQueue, mQueue_Destroy);
  mTEST_ASSERT_SUCCESS(mQueue_Create(&numberQueue, pAllocator));

  mTEST_ASSERT_SUCCESS(mQueue_PushBack(numberQueue, 0LL));

  for (int64_t i = 1; i < 100; i++)
  {
    mTEST_ASSERT_SUCCESS(mQueue_PushBack(numberQueue, i));
    mTEST_ASSERT_SUCCESS(mQueue_PushFront(numberQueue, -i));
  }

  int64_t lastCount = -100;
  size_t count = 0;

  for (int64_t i : *numberQueue)
  {
    mTEST_ASSERT_EQUAL(i, lastCount + 1);
    lastCount = i;
    count++;
  }

  mTEST_ASSERT_EQUAL(lastCount, 99);
  mTEST_ASSERT_EQUAL(count, 99 * 2 + 1);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mQueue, TestIterateBackwardsManual)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mQueue<int64_t>> numberQueue = nullptr;
  mDEFER_CALL(&numberQueue, mQueue_Destroy);
  mTEST_ASSERT_SUCCESS(mQueue_Create(&numberQueue, pAllocator));

  mTEST_ASSERT_SUCCESS(mQueue_PushBack(numberQueue, 0LL));

  for (int64_t i = 1; i < 100; i++)
  {
    mTEST_ASSERT_SUCCESS(mQueue_PushBack(numberQueue, i));
    mTEST_ASSERT_SUCCESS(mQueue_PushFront(numberQueue, -i));
  }

  int64_t lastCount = 100;
  size_t count = 0;

  for (auto it = numberQueue->end(); it != numberQueue->begin(); ++it)
  {
    mTEST_ASSERT_EQUAL(*it, lastCount - 1);
    lastCount = *it;
    count++;
  }

  mTEST_ASSERT_EQUAL(lastCount, -99);
  mTEST_ASSERT_EQUAL(count, 99 * 2 + 1);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mQueue, TestIterateBackwards)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mQueue<int64_t>> numberQueue = nullptr;
  mDEFER_CALL(&numberQueue, mQueue_Destroy);
  mTEST_ASSERT_SUCCESS(mQueue_Create(&numberQueue, pAllocator));

  mTEST_ASSERT_SUCCESS(mQueue_PushBack(numberQueue, 0LL));

  for (int64_t i = 1; i < 100; i++)
  {
    mTEST_ASSERT_SUCCESS(mQueue_PushBack(numberQueue, i));
    mTEST_ASSERT_SUCCESS(mQueue_PushFront(numberQueue, -i));
  }

  int64_t lastCount = 100;
  size_t count = 0;

  for (int64_t i : numberQueue->IterateReverse())
  {
    mTEST_ASSERT_EQUAL(i, lastCount - 1);
    lastCount = i;
    count++;
  }

  mTEST_ASSERT_EQUAL(lastCount, -99);
  mTEST_ASSERT_EQUAL(count, 99 * 2 + 1);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mQueue, TestIterateEmptyForward)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mQueue<int64_t>> numberQueue = nullptr;
  mDEFER_CALL(&numberQueue, mQueue_Destroy);
  mTEST_ASSERT_SUCCESS(mQueue_Create(&numberQueue, pAllocator));

  size_t count = 0;

  for (int64_t i : *numberQueue)
  {
    mUnused(i);
    count += 1;
  }

  mTEST_ASSERT_EQUAL(count, 0);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mQueue, TestIterateEmptyBackwards)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mQueue<int64_t>> numberQueue = nullptr;
  mDEFER_CALL(&numberQueue, mQueue_Destroy);
  mTEST_ASSERT_SUCCESS(mQueue_Create(&numberQueue, pAllocator));

  size_t count = 0;

  for (int64_t i : numberQueue->IterateReverse())
  {
    mUnused(i);
    count++;
  }

  mTEST_ASSERT_EQUAL(count, 0);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mQueue, TestConstIterateForward)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mQueue<int64_t>> numberQueue = nullptr;
  mDEFER_CALL(&numberQueue, mQueue_Destroy);
  mTEST_ASSERT_SUCCESS(mQueue_Create(&numberQueue, pAllocator));

  mTEST_ASSERT_SUCCESS(mQueue_PushBack(numberQueue, 0LL));

  for (int64_t i = 1; i < 100; i++)
  {
    mTEST_ASSERT_SUCCESS(mQueue_PushBack(numberQueue, i));
    mTEST_ASSERT_SUCCESS(mQueue_PushFront(numberQueue, -i));
  }

  int64_t lastCount = -100;
  size_t count = 0;

  const mPtr<mQueue<int64_t>> &constNumberQueue = numberQueue;

  for (const int64_t i : constNumberQueue->Iterate())
  {
    mTEST_ASSERT_EQUAL(i, lastCount + 1);
    lastCount = i;
    count++;
  }

  mTEST_ASSERT_EQUAL(lastCount, 99);
  mTEST_ASSERT_EQUAL(count, 99 * 2 + 1);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mQueue, TestConstIterateBackwardsManual)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mQueue<int64_t>> numberQueue = nullptr;
  mDEFER_CALL(&numberQueue, mQueue_Destroy);
  mTEST_ASSERT_SUCCESS(mQueue_Create(&numberQueue, pAllocator));

  mTEST_ASSERT_SUCCESS(mQueue_PushBack(numberQueue, 0LL));

  for (int64_t i = 1; i < 100; i++)
  {
    mTEST_ASSERT_SUCCESS(mQueue_PushBack(numberQueue, i));
    mTEST_ASSERT_SUCCESS(mQueue_PushFront(numberQueue, -i));
  }

  int64_t lastCount = 100;
  size_t count = 0;

  const mPtr<mQueue<int64_t>> &constNumberQueue = numberQueue;

  for (auto it = constNumberQueue->end(); it != constNumberQueue->begin(); ++it)
  {
    mTEST_ASSERT_EQUAL(*it, lastCount - 1);
    lastCount = *it;
    count++;
  }

  mTEST_ASSERT_EQUAL(lastCount, -99);
  mTEST_ASSERT_EQUAL(count, 99 * 2 + 1);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mQueue, TestConstIterateBackwards)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mQueue<int64_t>> numberQueue = nullptr;
  mDEFER_CALL(&numberQueue, mQueue_Destroy);
  mTEST_ASSERT_SUCCESS(mQueue_Create(&numberQueue, pAllocator));

  mTEST_ASSERT_SUCCESS(mQueue_PushBack(numberQueue, 0LL));

  for (int64_t i = 1; i < 100; i++)
  {
    mTEST_ASSERT_SUCCESS(mQueue_PushBack(numberQueue, i));
    mTEST_ASSERT_SUCCESS(mQueue_PushFront(numberQueue, -i));
  }

  int64_t lastCount = 100;
  size_t count = 0;

  const mPtr<mQueue<int64_t>> &constNumberQueue = numberQueue;

  for (const int64_t i : constNumberQueue->IterateReverse())
  {
    mTEST_ASSERT_EQUAL(i, lastCount - 1);
    lastCount = i;
    count++;
  }

  mTEST_ASSERT_EQUAL(lastCount, -99);
  mTEST_ASSERT_EQUAL(count, 99 * 2 + 1);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mQueue, TestConstIterateEmptyForward)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mQueue<int64_t>> numberQueue = nullptr;
  mDEFER_CALL(&numberQueue, mQueue_Destroy);
  mTEST_ASSERT_SUCCESS(mQueue_Create(&numberQueue, pAllocator));

  size_t count = 0;

  const mPtr<mQueue<int64_t>> &constNumberQueue = numberQueue;

  for (const int64_t i : *constNumberQueue)
  {
    mUnused(i);
    count += 1;
  }

  mTEST_ASSERT_EQUAL(count, 0);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mQueue, TestConstIterateEmptyBackwards)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mQueue<int64_t>> numberQueue = nullptr;
  mDEFER_CALL(&numberQueue, mQueue_Destroy);
  mTEST_ASSERT_SUCCESS(mQueue_Create(&numberQueue, pAllocator));

  size_t count = 0;

  const mPtr<mQueue<int64_t>> &constNumberQueue = numberQueue;

  for (int64_t i : constNumberQueue->IterateReverse())
  {
    mUnused(i);
    count++;
  }

  mTEST_ASSERT_EQUAL(count, 0);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mQueue, TestMin)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mQueue<int64_t>> numberQueue = nullptr;
  mDEFER_CALL(&numberQueue, mQueue_Destroy);
  mTEST_ASSERT_SUCCESS(mQueue_Create(&numberQueue, pAllocator));

  for (int64_t i = 0; i <= 100; i++)
    if (i % 2 == 1)
      mTEST_ASSERT_SUCCESS(mQueue_PushBack(numberQueue, i));
    else
      mTEST_ASSERT_SUCCESS(mQueue_PushFront(numberQueue, i));

  int64_t minValue = -1;
  mTEST_ASSERT_SUCCESS(mQueue_Min(numberQueue, (std::function<mComparisonResult(const int64_t &, const int64_t &)>)[](const int64_t &a, const int64_t &b) { return a > b ? mCR_Greater : (a < b ? mCR_Less : mCR_Equal); }, &minValue));
  mTEST_ASSERT_EQUAL(minValue, 0);

  minValue = 0;
  mTEST_ASSERT_SUCCESS(mQueue_Min(numberQueue, &minValue));
  mTEST_ASSERT_EQUAL(minValue, 0);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mQueue, TestMax)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mQueue<int64_t>> numberQueue = nullptr;
  mDEFER_CALL(&numberQueue, mQueue_Destroy);
  mTEST_ASSERT_SUCCESS(mQueue_Create(&numberQueue, pAllocator));

  for (int64_t i = 0; i <= 100; i++)
    if (i % 2 == 1)
      mTEST_ASSERT_SUCCESS(mQueue_PushBack(numberQueue, i));
    else
      mTEST_ASSERT_SUCCESS(mQueue_PushFront(numberQueue, i));

  int64_t maxValue = -1;
  mTEST_ASSERT_SUCCESS(mQueue_Max(numberQueue, (std::function<mComparisonResult(const int64_t &, const int64_t &)>)[](const int64_t &a, const int64_t &b) { return a > b ? mCR_Greater : (a < b ? mCR_Less : mCR_Equal); }, &maxValue));
  mTEST_ASSERT_EQUAL(maxValue, 100);

  maxValue = 0;
  mTEST_ASSERT_SUCCESS(mQueue_Max(numberQueue, &maxValue));
  mTEST_ASSERT_EQUAL(maxValue, 100);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mQueue, TestContains)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mQueue<int64_t>> numberQueue = nullptr;
  mDEFER_CALL(&numberQueue, mQueue_Destroy);
  mTEST_ASSERT_SUCCESS(mQueue_Create(&numberQueue, pAllocator));

  for (int64_t i = 0; i <= 100; i++)
    mTEST_ASSERT_SUCCESS(mQueue_PushBack(numberQueue, i));

  bool contained = false;
  mTEST_ASSERT_SUCCESS(mQueue_Contains(numberQueue, (int64_t)0, (std::function<bool (const int64_t &, const int64_t &)>)[](const int64_t &a, const int64_t &b) { return a == b; }, &contained));
  mTEST_ASSERT_EQUAL(contained, true);
  
  mTEST_ASSERT_SUCCESS(mQueue_Contains(numberQueue, (int64_t)-1, (std::function<bool(const int64_t &, const int64_t &)>)[](const int64_t &a, const int64_t &b) { return a == b; }, &contained));
  mTEST_ASSERT_EQUAL(contained, false);

  mTEST_ASSERT_SUCCESS(mQueue_Contains(numberQueue, (int64_t)100, (std::function<bool(const int64_t &, const int64_t &)>)[](const int64_t &a, const int64_t &b) { return a == b; }, &contained));
  mTEST_ASSERT_EQUAL(contained, true);

  mTEST_ASSERT_SUCCESS(mQueue_Contains(numberQueue, (int64_t)101, (std::function<bool(const int64_t &, const int64_t &)>)[](const int64_t &a, const int64_t &b) { return a == b; }, &contained));
  mTEST_ASSERT_EQUAL(contained, false);

  mTEST_ASSERT_SUCCESS(mQueue_Contains(numberQueue, (int64_t)50, (std::function<bool(const int64_t &, const int64_t &)>)[](const int64_t &a, const int64_t &b) { return a == b; }, &contained));
  mTEST_ASSERT_EQUAL(contained, true);

  mTEST_ASSERT_SUCCESS(mQueue_Contains(numberQueue, (int64_t)-100, (std::function<bool(const int64_t &, const int64_t &)>)[](const int64_t &a, const int64_t &b) { return a == b; }, &contained));
  mTEST_ASSERT_EQUAL(contained, false);

  mTEST_ASSERT_SUCCESS(mQueue_Contains(numberQueue, (int64_t)0, &contained));
  mTEST_ASSERT_EQUAL(contained, true);

  mTEST_ASSERT_SUCCESS(mQueue_Contains(numberQueue, (int64_t)-1, &contained));
  mTEST_ASSERT_EQUAL(contained, false);

  mTEST_ASSERT_SUCCESS(mQueue_Contains(numberQueue, (int64_t)100, &contained));
  mTEST_ASSERT_EQUAL(contained, true);

  mTEST_ASSERT_SUCCESS(mQueue_Contains(numberQueue, (int64_t)101, &contained));
  mTEST_ASSERT_EQUAL(contained, false);

  mTEST_ASSERT_SUCCESS(mQueue_Contains(numberQueue, (int64_t)50, &contained));
  mTEST_ASSERT_EQUAL(contained, true);

  mTEST_ASSERT_SUCCESS(mQueue_Contains(numberQueue, (int64_t)-100, &contained));
  mTEST_ASSERT_EQUAL(contained, false);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mQueue, TestRemoveDuplicates)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mQueue<int64_t>> numberQueue = nullptr;
  mDEFER_CALL(&numberQueue, mQueue_Destroy);
  mTEST_ASSERT_SUCCESS(mQueue_Create(&numberQueue, pAllocator));

  constexpr size_t targetValue = 100;

  for (int64_t i = 0; i < targetValue; i++)
  {
    mTEST_ASSERT_SUCCESS(mQueue_PushBack(numberQueue, i));
    mTEST_ASSERT_SUCCESS(mQueue_PushFront(numberQueue, i));
  }

  {
    mPtr<mQueue<int64_t>> numberQueue0 = nullptr;

    mTEST_ASSERT_SUCCESS(mQueue_RemoveDuplicates(numberQueue, &numberQueue0, pAllocator, (std::function<bool(const int64_t &, const int64_t &)>)[](const int64_t &a, const int64_t &b) { return a == b; }));

    size_t count0 = 0;
    mTEST_ASSERT_SUCCESS(mQueue_GetCount(numberQueue0, &count0));

    mTEST_ASSERT_EQUAL(targetValue, count0);

    mPtr<mQueue<int64_t>> numberQueue1 = nullptr;

    mTEST_ASSERT_SUCCESS(mQueue_RemoveDuplicates(numberQueue0, &numberQueue1, pAllocator, (std::function<bool(const int64_t &, const int64_t &)>)[](const int64_t &a, const int64_t &b) { return a == b; }));

    size_t count1 = 0;
    mTEST_ASSERT_SUCCESS(mQueue_GetCount(numberQueue1, &count1));

    for (const auto &_value : numberQueue->Iterate())
    {
      bool contained = false;
      mTEST_ASSERT_SUCCESS(mQueue_Contains(numberQueue0, _value, &contained));

      mTEST_ASSERT_TRUE(contained);
    }

    for (size_t i = 0; i < count0; i++)
    {
      mTEST_ASSERT_EQUAL((*numberQueue0)[i], (*numberQueue1)[i]);
      mTEST_ASSERT_EQUAL((*numberQueue0)[i], (*numberQueue)[i]); // Implementation Dependent.
    }

    mTEST_ASSERT_EQUAL(count1, count0);
  }

  {
    mPtr<mQueue<int64_t>> numberQueue0 = nullptr;

    mTEST_ASSERT_SUCCESS(mQueue_RemoveDuplicates(numberQueue, &numberQueue0, pAllocator));

    size_t count0 = 0;
    mTEST_ASSERT_SUCCESS(mQueue_GetCount(numberQueue0, &count0));

    mTEST_ASSERT_EQUAL(targetValue, count0);

    mPtr<mQueue<int64_t>> numberQueue1 = nullptr;

    mTEST_ASSERT_SUCCESS(mQueue_RemoveDuplicates(numberQueue0, &numberQueue1, pAllocator));

    size_t count1 = 0;
    mTEST_ASSERT_SUCCESS(mQueue_GetCount(numberQueue1, &count1));

    for (const auto &_value : numberQueue->Iterate())
    {
      bool contained = false;
      mTEST_ASSERT_SUCCESS(mQueue_Contains(numberQueue0, _value, &contained));

      mTEST_ASSERT_TRUE(contained);
    }

    for (size_t i = 0; i < count0; i++)
    {
      mTEST_ASSERT_EQUAL((*numberQueue0)[i], (*numberQueue1)[i]);
      mTEST_ASSERT_EQUAL((*numberQueue0)[i], (*numberQueue)[i]); // Implementation Dependent.
    }

    mTEST_ASSERT_EQUAL(count1, count0);
  }

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mQueue, TestSelect)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mQueue<int64_t>> numberQueue = nullptr;
  mDEFER_CALL(&numberQueue, mQueue_Destroy);
  mTEST_ASSERT_SUCCESS(mQueue_Create(&numberQueue, pAllocator));

  constexpr size_t targetValue = 100;

  for (int64_t i = 0; i < targetValue; i++)
  {
    mTEST_ASSERT_SUCCESS(mQueue_PushBack(numberQueue, i));
    mTEST_ASSERT_SUCCESS(mQueue_PushFront(numberQueue, i));
  }

  {
    mPtr<mQueue<int64_t>> numberQueue0 = nullptr;

    mTEST_ASSERT_SUCCESS(mQueue_Select(numberQueue, &numberQueue0, pAllocator, (std::function<bool(const int64_t &)>)[](const int64_t &a) { return a == 50; }));

    size_t count0 = 0;
    mTEST_ASSERT_SUCCESS(mQueue_GetCount(numberQueue0, &count0));

    mTEST_ASSERT_EQUAL(2, count0);

    mPtr<mQueue<int64_t>> numberQueue1 = nullptr;

    mTEST_ASSERT_SUCCESS(mQueue_Select(numberQueue0, &numberQueue1, pAllocator, (std::function<bool(const int64_t &)>)[](const int64_t &a) { return a == 50; }));

    size_t count1 = 0;
    mTEST_ASSERT_SUCCESS(mQueue_GetCount(numberQueue1, &count1));

    for (size_t i = 0; i < 2; i++)
      mTEST_ASSERT_EQUAL((*numberQueue0)[i], 50);

    mTEST_ASSERT_EQUAL(count1, count0);

    bool any = false;
    mTEST_ASSERT_SUCCESS(mQueue_Any(numberQueue1, &any));

    mTEST_ASSERT_TRUE(any);
  }

  {
    mPtr<mQueue<int64_t>> numberQueue0 = nullptr;

    struct _internal
    {
      static bool is_fifty(int64_t a)
      {
        return a == 50;
      }
    };

    mTEST_ASSERT_SUCCESS(mQueue_Select(numberQueue, &numberQueue0, pAllocator, _internal::is_fifty));

    size_t count0 = 0;
    mTEST_ASSERT_SUCCESS(mQueue_GetCount(numberQueue0, &count0));

    mTEST_ASSERT_EQUAL(2, count0);

    mPtr<mQueue<int64_t>> numberQueue1 = nullptr;

    mTEST_ASSERT_SUCCESS(mQueue_Select(numberQueue0, &numberQueue1, pAllocator, _internal::is_fifty));

    size_t count1 = 0;
    mTEST_ASSERT_SUCCESS(mQueue_GetCount(numberQueue1, &count1));

    for (size_t i = 0; i < 2; i++)
      mTEST_ASSERT_EQUAL((*numberQueue0)[i], 50);

    mTEST_ASSERT_EQUAL(count1, count0);

    bool any = false;
    mTEST_ASSERT_SUCCESS(mQueue_Any(numberQueue1, &any));

    mTEST_ASSERT_TRUE(any);
  }

  {
    mPtr<mQueue<int64_t>> numberQueue0 = nullptr;

    struct _internal
    {
      static bool is_twohundred(int64_t a)
      {
        return a == 200;
      }
    };

    mTEST_ASSERT_SUCCESS(mQueue_Select(numberQueue, &numberQueue0, pAllocator, _internal::is_twohundred));

    size_t count0 = 0;
    mTEST_ASSERT_SUCCESS(mQueue_GetCount(numberQueue0, &count0));

    mTEST_ASSERT_EQUAL(0, count0);

    mPtr<mQueue<int64_t>> numberQueue1 = nullptr;

    mTEST_ASSERT_SUCCESS(mQueue_Select(numberQueue0, &numberQueue1, pAllocator, _internal::is_twohundred));

    size_t count1 = 0;
    mTEST_ASSERT_SUCCESS(mQueue_GetCount(numberQueue1, &count1));

    mTEST_ASSERT_EQUAL(count1, count0);

    bool any = true;
    mTEST_ASSERT_SUCCESS(mQueue_Any(numberQueue1, &any));

    mTEST_ASSERT_FALSE(any);
  }

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mQueue, TestCopyTo)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mQueue<int64_t>> numberQueue = nullptr;
  mDEFER_CALL(&numberQueue, mQueue_Destroy);
  mTEST_ASSERT_SUCCESS(mQueue_Create(&numberQueue, pAllocator));

  constexpr size_t targetValue = 100;

  for (int64_t i = 0; i < targetValue; i++)
  {
    mTEST_ASSERT_SUCCESS(mQueue_PushBack(numberQueue, i));
    mTEST_ASSERT_SUCCESS(mQueue_PushFront(numberQueue, i));
  }

  mPtr<mQueue<int64_t>> numberQueue0 = nullptr;
  mTEST_ASSERT_SUCCESS(mQueue_CopyTo(numberQueue, &numberQueue0, pAllocator));

  mTEST_ASSERT_EQUAL(numberQueue->count, numberQueue0->count);

  for (size_t i = 0; i < numberQueue->count; i++)
    mTEST_ASSERT_EQUAL((*numberQueue)[i], (*numberQueue0)[i]);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mQueue, TestOrderBy)
{
  mTEST_ALLOCATOR_SETUP();
  
  {
    mPtr<mQueue<int64_t>> numberQueue = nullptr;
    mDEFER_CALL(&numberQueue, mQueue_Destroy);
    mTEST_ASSERT_SUCCESS(mQueue_Create(&numberQueue, pAllocator));

    constexpr size_t targetValue = 1024;

    for (int64_t i = 0; i < targetValue; i++)
    {
      mTEST_ASSERT_SUCCESS(mQueue_PushBack(numberQueue, i));
      mTEST_ASSERT_SUCCESS(mQueue_PushFront(numberQueue, i));
    }

    mTEST_ASSERT_SUCCESS(mQueue_OrderBy(numberQueue));

    int64_t first;
    mTEST_ASSERT_SUCCESS(mQueue_PeekFront(numberQueue, &first));

    for (size_t i = 1; i < numberQueue->count; i++)
      mTEST_ASSERT_TRUE((*numberQueue)[i] >= first);
  }

  {
    mPtr<mQueue<int64_t>> numberQueue = nullptr;
    mDEFER_CALL(&numberQueue, mQueue_Destroy);
    mTEST_ASSERT_SUCCESS(mQueue_Create(&numberQueue, pAllocator));

    constexpr size_t targetValue = 1024;

    for (int64_t i = 0; i < targetValue; i++)
    {
      if (i % 4 < 2)
        mTEST_ASSERT_SUCCESS(mQueue_PushBack(numberQueue, i));
      
      if (i % 3 > 1)
        mTEST_ASSERT_SUCCESS(mQueue_PushFront(numberQueue, i));
    }

    mTEST_ASSERT_SUCCESS(mQueue_OrderBy(numberQueue, (std::function<mComparisonResult(const int64_t &, const int64_t &)>)[](const int64_t &a, const int64_t &b) { return a > b ? mCR_Greater : (a < b ? mCR_Less : mCR_Equal); }));

    int64_t first;
    mTEST_ASSERT_SUCCESS(mQueue_PeekFront(numberQueue, &first));

    for (size_t i = 1; i < numberQueue->count; i++)
      mTEST_ASSERT_TRUE((*numberQueue)[i] >= first);
  }

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mQueue, TestEquals)
{
  mTEST_ALLOCATOR_SETUP();

  {
    mPtr<mQueue<size_t>> a;
    mPtr<mQueue<size_t>> b;
    mTEST_ASSERT_TRUE(mQueue_Equals(a, b));
    mTEST_ASSERT_TRUE(mQueue_Equals(b, a));

    mDEFER_CALL(&a, mQueue_Destroy);
    mTEST_ASSERT_SUCCESS(mQueue_Create(&a, pAllocator));
    mTEST_ASSERT_FALSE(mQueue_Equals(a, b));
    mTEST_ASSERT_FALSE(mQueue_Equals(b, a));

    mDEFER_CALL(&b, mQueue_Destroy);
    mTEST_ASSERT_SUCCESS(mQueue_Create(&b, pAllocator));

    mTEST_ASSERT_TRUE(mQueue_Equals(a, b));
    mTEST_ASSERT_TRUE(mQueue_Equals(b, a));

    mTEST_ASSERT_SUCCESS(mQueue_PushBack(a, (size_t)0));
    mTEST_ASSERT_FALSE(mQueue_Equals(a, b));
    mTEST_ASSERT_FALSE(mQueue_Equals(b, a));

    mTEST_ASSERT_SUCCESS(mQueue_PushBack(b, (size_t)1));
    mTEST_ASSERT_FALSE(mQueue_Equals(a, b));
    mTEST_ASSERT_FALSE(mQueue_Equals(b, a));

    mTEST_ASSERT_SUCCESS(mQueue_PushFront(b, (size_t)0));
    mTEST_ASSERT_FALSE(mQueue_Equals(a, b));
    mTEST_ASSERT_FALSE(mQueue_Equals(b, a));

    mTEST_ASSERT_SUCCESS(mQueue_PushBack(a, (size_t)1));
    mTEST_ASSERT_TRUE(mQueue_Equals(a, b));
    mTEST_ASSERT_TRUE(mQueue_Equals(b, a));

    mTEST_ASSERT_SUCCESS(mQueue_PushBack(a, (size_t)2));
    mTEST_ASSERT_FALSE(mQueue_Equals(a, b));
    mTEST_ASSERT_FALSE(mQueue_Equals(b, a));

    mTEST_ASSERT_SUCCESS(mQueue_PushBack(b, (size_t)2));
    mTEST_ASSERT_TRUE(mQueue_Equals(a, b));
    mTEST_ASSERT_TRUE(mQueue_Equals(b, a));
  }

  struct _inner
  {
    static bool IsNotZero(const size_t a)
    {
      return a != 0;
    }
  };

  const auto &cmpFunc = [pAllocator](const size_t *pA, const size_t aCount, const size_t *pB, const size_t bCount)
  {
    mPtr<mQueue<size_t>> a, b;
    mASSERT(mSUCCEEDED(mQueue_Create(&a, pAllocator)), "");
    mASSERT(mSUCCEEDED(mQueue_Create(&b, pAllocator)), "");

    for (size_t i = 0; i < aCount; i++)
      mASSERT(mSUCCEEDED(mQueue_PushBack(a, pA[i])), "");

    for (size_t i = 0; i < bCount; i++)
      mASSERT(mSUCCEEDED(mQueue_PushBack(b, pB[i])), "");

    const bool ret = mQueue_Equals<size_t, mEquals<size_t>, mFN_WRAPPER(_inner::IsNotZero)>(a, b);
    mASSERT(ret == (mQueue_Equals<size_t, mEquals<size_t>, mFN_WRAPPER(_inner::IsNotZero)>(b, a)), "");

    return ret;
  };

  {
    const size_t data0[] = { 0, 0 };
    const size_t data1[] = { 0, 0, 0 };
    mTEST_ASSERT_TRUE(cmpFunc(data0, mARRAYSIZE(data0), data1, mARRAYSIZE(data1)));
  }

  {
    const size_t data0[] = { 1, 0, 0 };
    const size_t data1[] = { 0, 0, 0, 1, 0, 0, 0 };
    mTEST_ASSERT_TRUE(cmpFunc(data0, mARRAYSIZE(data0), data1, mARRAYSIZE(data1)));
  }

  {
    const size_t data0[] = { 1, 1, 1, 0 };
    const size_t data1[] = { 0, 1, 0, 1, 0, 0, 0, 1 };
    mTEST_ASSERT_TRUE(cmpFunc(data0, mARRAYSIZE(data0), data1, mARRAYSIZE(data1)));
  }

  {
    const size_t data0[] = { 1, 3, 5, 0 };
    const size_t data1[] = { 0, 1, 0, 3, 0, 0, 0, 5 };
    mTEST_ASSERT_TRUE(cmpFunc(data0, mARRAYSIZE(data0), data1, mARRAYSIZE(data1)));
  }

  {
    const size_t data0[] = { 1, 3, 5, 0 };
    const size_t data1[] = { 0, 1, 0, 5, 0, 0, 0, 3 };
    mTEST_ASSERT_FALSE(cmpFunc(data0, mARRAYSIZE(data0), data1, mARRAYSIZE(data1)));
  }

  {
    const size_t data0[] = { 1, 2, 3, 0 };
    const size_t data1[] = { 0, 1, 0, 5, 0, 0, 0, 3 };
    mTEST_ASSERT_FALSE(cmpFunc(data0, mARRAYSIZE(data0), data1, mARRAYSIZE(data1)));
  }

  {
    const size_t data0[] = { 1, 1, 1, 0 };
    const size_t data1[] = { 0, 1, 0, 1, 0, 0, 0, 0 };
    mTEST_ASSERT_FALSE(cmpFunc(data0, mARRAYSIZE(data0), data1, mARRAYSIZE(data1)));
  }

  {
    const size_t data0[] = { 1, 1, 1, 0 };
    const size_t data1[] = { 0, 1, 0, 1, 0, 0, 0, 0, 1 };
    mTEST_ASSERT_TRUE(cmpFunc(data0, mARRAYSIZE(data0), data1, mARRAYSIZE(data1)));
  }

  {
    const size_t data0[] = { 1, 1, 1, 0 };
    const size_t data1[] = { 0, 1, 0, 1, 0, 0, 0, 0, 2 };
    mTEST_ASSERT_FALSE(cmpFunc(data0, mARRAYSIZE(data0), data1, mARRAYSIZE(data1)));
  }

  {
    const size_t data0[] = { 0, 1, 0, 1, 0, 0, 0, 0 };
    const size_t data1[] = { 1, 1, 1, 0 };
    mTEST_ASSERT_FALSE(cmpFunc(data0, mARRAYSIZE(data0), data1, mARRAYSIZE(data1)));
  }

  {
    const size_t data0[] = { 0, 1, 0, 1, 0, 0, 0, 0 };
    const size_t data1[] = { 0, 1, 1, 0 };
    mTEST_ASSERT_TRUE(cmpFunc(data0, mARRAYSIZE(data0), data1, mARRAYSIZE(data1)));
  }

  {
    const size_t data0[] = { 0, 1, 0, 0, 0, 0, 0, 0, 1, 1 };
    const size_t data1[] = { 0, 1, 0, 1, 0, 0, 0, 0 };
    mTEST_ASSERT_FALSE(cmpFunc(data0, mARRAYSIZE(data0), data1, mARRAYSIZE(data1)));
  }

  mTEST_ALLOCATOR_ZERO_CHECK();
}
