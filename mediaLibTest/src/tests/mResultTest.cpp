#include "mTestLib.h"

mTEST(mResult, TestToString)
{
  mTEST_ALLOCATOR_SETUP();

  const char *resultNames[mResult_Count] =
  {
    "mR_Success",
    "mR_Break",
    "mR_InvalidParameter",
    "mR_ArgumentNull",
    "mR_InternalError",
    "mR_MemoryAllocationFailure",
    "mR_NotImplemented",
    "mR_NotInitialized",
    "mR_IndexOutOfBounds",
    "mR_ArgumentOutOfBounds",
    "mR_Timeout",
    "mR_OperationNotSupported",
    "mR_ResourceNotFound",
    "mR_ResourceInvalid",
    "mR_ResourceStateInvalid",
    "mR_ResourceIncompatible",
    "mR_EndOfStream",
    "mR_RenderingError",
    "mR_Failure",
    "mR_NotSupported",
    "mR_ResourceAlreadyExists",
    "mR_IOFailure",
  };

  const char invalidResult[] = "<Unknown mResult>";

  mString resultString;
  mTEST_ASSERT_SUCCESS(mString_Create(&resultString, (char *)nullptr, pAllocator));
  
  for (size_t i = 0; i < mResult_Count; i++)
  {
    mTEST_ASSERT_SUCCESS(mResult_ToString((mResult)(i), &resultString));
    mTEST_ASSERT_EQUAL(resultString, resultNames[i]);
  }

  mTEST_ASSERT_SUCCESS(mResult_ToString((mResult)(mResult_Count), &resultString));
  mTEST_ASSERT_EQUAL(resultString, invalidResult);

  mTEST_ALLOCATOR_ZERO_CHECK();
}
