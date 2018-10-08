#include "mTestLib.h"

mTEST(TestAllocator, TestAlloc)
{
  mTEST_ALLOCATOR_SETUP();
  
  size_t *pData = nullptr;
  mDEFER(mAllocator_FreePtr(pAllocator, &pData));

  mTEST_ASSERT_SUCCESS(mAllocator_Allocate(pAllocator, &pData, 10));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(TestAllocator, TestAllocZero)
{
  mTEST_ALLOCATOR_SETUP();

  size_t *pData = nullptr;
  mDEFER(mAllocator_FreePtr(pAllocator, &pData));

  mTEST_ASSERT_SUCCESS(mAllocator_AllocateZero(pAllocator, &pData, 10));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(TestAllocator, TestRealloc)
{
  mTEST_ALLOCATOR_SETUP();

  size_t *pData = nullptr;
  mDEFER(mAllocator_FreePtr(pAllocator, &pData));
  
  mTEST_ASSERT_SUCCESS(mAllocator_AllocateZero(pAllocator, &pData, 10));
  mTEST_ASSERT_SUCCESS(mAllocator_Reallocate(pAllocator, &pData, 20));
  mTEST_ASSERT_SUCCESS(mAllocator_Reallocate(pAllocator, &pData, 40));
  mTEST_ASSERT_SUCCESS(mAllocator_Reallocate(pAllocator, &pData, 80));
  mTEST_ASSERT_SUCCESS(mAllocator_Reallocate(pAllocator, &pData, 160));
  mTEST_ASSERT_SUCCESS(mAllocator_Reallocate(pAllocator, &pData, 320));
  mTEST_ASSERT_SUCCESS(mAllocator_Reallocate(pAllocator, &pData, 640));
  mTEST_ASSERT_SUCCESS(mAllocator_Reallocate(pAllocator, &pData, 1280));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(TestAllocator, TestFreePtr)
{
  mTEST_ALLOCATOR_SETUP();

  size_t *pData = nullptr;
  mDEFER(mAllocator_FreePtr(pAllocator, &pData));

  mTEST_ASSERT_SUCCESS(mAllocator_AllocateZero(pAllocator, &pData, 10));
  mTEST_ASSERT_SUCCESS(mAllocator_FreePtr(pAllocator, &pData));

  mTEST_ALLOCATOR_ZERO_CHECK();
}


