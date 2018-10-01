#include "mTestLib.h"

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
