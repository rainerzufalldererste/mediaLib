#include "mTestLib.h"

mTEST(TestAllocator, TestAlloc)
{
  mTEST_ALLOCATOR_SETUP();
  
  size_t *pData = nullptr;
  mDEFER_CALL_2(mAllocator_FreePtr, pAllocator, &pData);

  mTEST_ASSERT_SUCCESS(mAllocator_Allocate(pAllocator, &pData, 10));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(TestAllocator, TestAllocZero)
{
  mTEST_ALLOCATOR_SETUP();

  size_t *pData = nullptr;
  mDEFER_CALL_2(mAllocator_FreePtr, pAllocator, &pData);

  mTEST_ASSERT_SUCCESS(mAllocator_AllocateZero(pAllocator, &pData, 10));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(TestAllocator, TestRealloc)
{
  mTEST_ALLOCATOR_SETUP();

  size_t *pData = nullptr;
  mDEFER_CALL_2(mAllocator_FreePtr, pAllocator, &pData);

  mTEST_ASSERT_SUCCESS(mAllocator_Reallocate(pAllocator, &pData, 10));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(TestAllocator, TestFreePtr)
{
  mTEST_ALLOCATOR_SETUP();

  size_t *pData = nullptr;
  mDEFER_CALL_2(mAllocator_FreePtr, pAllocator, &pData);

  mTEST_ASSERT_SUCCESS(mAllocator_AllocateZero(pAllocator, &pData, 10));
  mTEST_ASSERT_SUCCESS(mAllocator_FreePtr(pAllocator, &pData));

  mTEST_ALLOCATOR_ZERO_CHECK();
}


