#include "mTestLib.h"

mTEST(mAllocator, TestReallocFailure)
{
  mTEST_ALLOCATOR_SETUP();

  uint8_t *pData = nullptr;
  mDEFER_CALL_2(mAllocator_FreePtr, pAllocator, &pData);

  mTEST_ASSERT_SUCCESS(mAllocator_Allocate(pAllocator, &pData, 1));
  mTEST_ASSERT_EQUAL(mR_MemoryAllocationFailure, mAllocator_Reallocate(pAllocator, &pData, (size_t)0xFFFFFFFFFFFF));

  mTEST_ASSERT_EQUAL(pData, nullptr);

  mTEST_ALLOCATOR_ZERO_CHECK();
}
