#include "mTestLib.h"

#include <vector>

mTEST_STATIC_ASSERT(true);

mTEST(TestDestructible, Destruct)
{
  mTEST_ALLOCATOR_SETUP();

  mDummyDestructible dummy;
  mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy, pAllocator));
  mTEST_ASSERT_FALSE(dummy.destructed);

  mTEST_ASSERT_SUCCESS(mDestruct(&dummy));
  mTEST_ASSERT_TRUE(dummy.destructed);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(TestTemplatedDestructible, Destruct)
{
  mTEST_ALLOCATOR_SETUP();

  mTemplatedDestructible<size_t> dummy;
  mTEST_ASSERT_SUCCESS(mDummyDestructible_Create(&dummy, pAllocator));
  mTEST_ASSERT_FALSE(dummy.destructed);

  mTEST_ASSERT_SUCCESS(mDestruct(&dummy));
  mTEST_ASSERT_TRUE(dummy.destructed);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(TestCppClass, TestMoveCopyDestruct)
{
  mTEST_ASSERT_EQUAL(mTestCppClass::GlobalCount(), 0);

  {
    std::vector<mTestCppClass> testCppClass;

    for (size_t i = 0; i < 1024; i++)
      testCppClass.push_back(mTestCppClass(i));
  }

  mTEST_ASSERT_EQUAL(mTestCppClass::GlobalCount(), 0);

  {
    std::vector<mTestCppClass> testCppClass;

    for (size_t i = 0; i < 1024; i++)
    {
      mTestCppClass c = std::move(mTestCppClass(i));

      testCppClass.push_back(std::move(c));
    }
  }

  mTEST_ASSERT_EQUAL(mTestCppClass::GlobalCount(), 0);

  {
    std::vector<mTestCppClass> testCppClass;

    for (size_t i = 0; i < 1024; i++)
    {
      mTestCppClass c = mTestCppClass(i);

      testCppClass.push_back(c);
    }
  }

  mTEST_ASSERT_EQUAL(mTestCppClass::GlobalCount(), 0);

  {
    std::vector<mTestCppClass> testCppClass;

    for (size_t i = 0; i < 1024; i++)
    {
      mTestCppClass c = mTestCppClass(i);
      mTestCppClass a = mTestCppClass(i);
      a = c;

      testCppClass.push_back(std::move(c));
    }
  }

  mTEST_ASSERT_EQUAL(mTestCppClass::GlobalCount(), 0);

  {
    std::vector<mTestCppClass> testCppClass;

    for (size_t i = 0; i < 1024; i++)
    {
      mTestCppClass c = mTestCppClass(i);
      mTestCppClass a = mTestCppClass(i);
      a = std::move(c);

      testCppClass.push_back(c);
    }
  }

  mTEST_ASSERT_EQUAL(mTestCppClass::GlobalCount(), 0);

  mTEST_RETURN_SUCCESS();
}
