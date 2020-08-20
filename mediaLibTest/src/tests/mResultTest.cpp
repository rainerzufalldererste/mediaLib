#include "mTestLib.h"

#ifdef _DEBUG

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
    "mR_InsufficientPrivileges",
    "mR_ResourceBusy"
  };

  const char invalidResult[] = "<Unknown mResult>";

  for (size_t i = 0; i < mResult_Count; i++)
    mTEST_ASSERT_EQUAL(0, strcmp(mResult_ToString((mResult)i), resultNames[i]));

  mTEST_ASSERT_EQUAL(0, strcmp(mResult_ToString((mResult)mResult_Count), invalidResult));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

#else

void UNUSED_FILE_MRESULT_TEST_CPP(void)
{
  return;
}

#endif
