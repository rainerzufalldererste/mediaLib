#include "mTestLib.h"
#include "mList.h"

mTEST(mList, TestCreate)
{
  mTEST_ALLOCATOR_SETUP();

  mTEST_ASSERT_EQUAL(mR_ArgumentNull, mList_Create((mList<size_t> *)nullptr, pAllocator));

  mList<size_t> numbers;
  mDEFER_CALL(&numbers, mList_Destroy);
  mTEST_ASSERT_SUCCESS(mList_Create(&numbers, pAllocator));

  mTEST_ASSERT_NOT_EQUAL(numbers.pAllocator, nullptr);

  // Destroys the old one.
  mTEST_ASSERT_SUCCESS(mList_Create(&numbers, pAllocator));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mList, TestInsertRemove)
{
  mTEST_ALLOCATOR_SETUP();

  std::vector<size_t> comparisonVector;
  mList<size_t> numberList;
  mDEFER_CALL(&numberList, mList_Destroy);
  mTEST_ASSERT_SUCCESS(mList_Create(&numberList, pAllocator));

  size_t count = (size_t)-1;
  size_t seed = 0;

  for (size_t i = 0; i < 100; i++)
  {
    seed += i + seed * 0xDEADF00D;

    size_t operation = seed % 4;
    size_t value;

    mTEST_ASSERT_SUCCESS(mList_GetCount(numberList, &count));
    mTEST_ASSERT_EQUAL(count, comparisonVector.size());

    switch (operation)
    {
    case 0:
      mTEST_ASSERT_SUCCESS(mList_PushBack(numberList, seed));
      comparisonVector.push_back(seed);
      break;

    case 1:
      if (count > 0)
      {
        mTEST_ASSERT_SUCCESS(mList_PopBack(numberList, &value));
        comparisonVector.pop_back();
      }
      else
      {
        mTEST_ASSERT_EQUAL(mR_IndexOutOfBounds, mList_PopBack(numberList, &value));
      }

      break;

    case 2:
      if (count > 0)
      {
        const size_t index = (seed + 0xDEADF00D) % count;

        mTEST_ASSERT_SUCCESS(mList_PopAt(numberList, index, &value));
        mTEST_ASSERT_EQUAL(value, comparisonVector[index]);

        comparisonVector.erase(comparisonVector.begin() + index);
      }

      break;

    case 3:
    {
      const size_t index = (seed + 0xDEADF00D) % count;

      mTEST_ASSERT_SUCCESS(mList_InsertAt(numberList, index, seed));
      comparisonVector.insert(comparisonVector.begin() + index, seed);

      break;
    }
    }
  }

  mTEST_ASSERT_EQUAL(numberList.count, comparisonVector.size());

  for (size_t i = 0; i < numberList.count; i++)
    mTEST_ASSERT_EQUAL(numberList[i], comparisonVector[i]);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mList, TestClear)
{
  mTEST_ALLOCATOR_SETUP();

  mList<mDummyDestructible> numbers;
  mDEFER_CALL(&numbers, mList_Destroy);
  mTEST_ASSERT_SUCCESS(mList_Create(&numbers, pAllocator));

  for (size_t i = 0; i < 1024; i++)
  {
    mDummyDestructible dummy;
    mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy, pAllocator));
    mTEST_ASSERT_SUCCESS(mList_PushBack(numbers, &dummy));
  }

  mTEST_ASSERT_SUCCESS(mList_Clear(numbers));

  size_t count = (size_t)-1;
  mTEST_ASSERT_SUCCESS(mList_GetCount(numbers, &count));
  mTEST_ASSERT_EQUAL(0, count);

  for (size_t i = 0; i < 1024; i++)
  {
    mDummyDestructible dummy;
    mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy, pAllocator));
    mTEST_ASSERT_SUCCESS(mList_PushBack(numbers, &dummy));
  }

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mList, TestIterate)
{
  mTEST_ALLOCATOR_SETUP();

  mList<size_t> numbers;
  mTEST_ASSERT_SUCCESS(mList_Create(&numbers, pAllocator));

  for (size_t i = 0; i < 1024; i++)
    mTEST_ASSERT_SUCCESS(mList_PushBack(numbers, i));

  size_t last = 0;

  for (size_t i : numbers)
    mTEST_ASSERT_EQUAL(i, last++);

  mTEST_ASSERT_EQUAL(1024, last);

  last = 0;

  for (size_t i : numbers.Iterate())
    mTEST_ASSERT_EQUAL(i, last++);

  mTEST_ASSERT_EQUAL(1024, last);

  last = 0;

  for (const size_t &i : numbers.Iterate())
    mTEST_ASSERT_EQUAL(i, last++);

  mTEST_ASSERT_EQUAL(1024, last);

  struct _internal
  {
    static bool test_all(const mList<size_t> &list)
    {
      size_t last = 0;

      for (const size_t &i : list.Iterate())
        if (i != last++)
          return false;

      if (1024 != last)
        return false;

      last = 0;

      for (const size_t &i : list)
        if (i != last++)
          return false;

      if (1024 != last)
        return false;

      return true;
    }
  };

  mTEST_ASSERT_TRUE(_internal::test_all(numbers));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mList, TestIterateReverse)
{
  mTEST_ALLOCATOR_SETUP();

  mList<size_t> numbers;
  mTEST_ASSERT_SUCCESS(mList_Create(&numbers, pAllocator));

  for (size_t i = 0; i < 1024; i++)
    mTEST_ASSERT_SUCCESS(mList_PushBack(numbers, i));

  size_t last = 1023;

  for (size_t i : numbers.IterateReverse())
    mTEST_ASSERT_EQUAL(i, last--);

  mTEST_ASSERT_EQUAL((size_t)-1, last);

  last = 1023;

  for (const size_t &i : numbers.IterateReverse())
    mTEST_ASSERT_EQUAL(i, last--);

  mTEST_ASSERT_EQUAL((size_t)-1, last);

  struct _internal
  {
    static bool test_all(const mList<size_t> &list)
    {
      size_t last = 1023;

      for (const size_t &i : list.IterateReverse())
        if (i != last--)
          return false;

      if ((size_t)-1 != last)
        return false;

      return true;
    }
  };

  mTEST_ASSERT_TRUE(_internal::test_all(numbers));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mList, TestResizeWith)
{
  mTEST_ALLOCATOR_SETUP();

  mList<size_t> numbers;
  mTEST_ASSERT_SUCCESS(mList_Create(&numbers, pAllocator));

  for (size_t i = 0; i < 1024; i++)
    mTEST_ASSERT_SUCCESS(mList_PushBack(numbers, i));

  size_t count = 0;
  mTEST_ASSERT_SUCCESS(mList_GetCount(numbers, &count));
  mTEST_ASSERT_EQUAL(count, 1024);

  mTEST_ASSERT_SUCCESS(mList_ResizeWith(numbers, 128, (size_t)0xFFFF));

  mTEST_ASSERT_SUCCESS(mList_GetCount(numbers, &count));
  mTEST_ASSERT_EQUAL(count, 1024);

  mTEST_ASSERT_SUCCESS(mList_ResizeWith(numbers, 1024, (size_t)0xF0F0));

  mTEST_ASSERT_SUCCESS(mList_GetCount(numbers, &count));
  mTEST_ASSERT_EQUAL(count, 1024);

  mTEST_ASSERT_SUCCESS(mList_ResizeWith(numbers, 1024 * 2, (size_t)0xF000));

  mTEST_ASSERT_SUCCESS(mList_GetCount(numbers, &count));
  mTEST_ASSERT_EQUAL(count, 1024 * 2);

  size_t last = 0;

  for (size_t i : numbers)
    if (last < 1024)
      mTEST_ASSERT_EQUAL(i, last++);
    else
      mTEST_ASSERT_EQUAL(i, 0xF000);

  mTEST_ALLOCATOR_ZERO_CHECK();
}
