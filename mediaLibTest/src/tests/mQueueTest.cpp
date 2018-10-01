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
