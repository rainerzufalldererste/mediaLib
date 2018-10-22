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

  mTEST_ALLOCATOR_ZERO_CHECK();
  mTEST_ASSERT_EQUAL(0, mTestCppClass::GlobalCount());
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
