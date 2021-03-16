#include "mTestLib.h"

void mIncrementPtrValue(size_t *pPtr)
{
  (*pPtr)++;
}

//////////////////////////////////////////////////////////////////////////

mTEST(mDefer, TestExecuteLambda)
{
  size_t i = 0;

  {
    mDEFER(i++);
    mTEST_ASSERT_EQUAL(0, i);
  }

  mTEST_ASSERT_EQUAL(1, i);

  mTEST_RETURN_SUCCESS();
}

mTEST(mDefer, TestExecuteAssigned)
{
  size_t i = 0;

  {
    mDefer defer;

    {
      defer = mDefer([&]() {i++;});
      mTEST_ASSERT_EQUAL(0, i);
    }

    mTEST_ASSERT_EQUAL(0, i);
  }

  mTEST_ASSERT_EQUAL(1, i);

  mTEST_RETURN_SUCCESS();
}

mTEST(mDefer, TestExecuteIncrementPtrValue)
{
  size_t value = 0;
  size_t *pPtr = &value;

  {
    mDEFER_CALL(pPtr, mIncrementPtrValue);
    mTEST_ASSERT_EQUAL(0, value);
  }

  mTEST_ASSERT_EQUAL(1, value);

  mTEST_RETURN_SUCCESS();
}

mTEST(mDefer, TestExecuteAssignedIncrementPtrValue)
{
  size_t value = 0;
  size_t *pPtr = &value;

  {
    mDEFER_CALL(pPtr, mIncrementPtrValue);
    
    mTEST_ASSERT_EQUAL(0, value);
  }

  mTEST_ASSERT_EQUAL(1, value);

  mTEST_RETURN_SUCCESS();
}

mTEST(mDefer, TestExecuteLambdaOnNoError)
{
  mResult mSTDRESULT = mR_Success;
  size_t i = 0;

  {
    mDEFER_ON_ERROR(i++);
    mTEST_ASSERT_EQUAL(0, i);
  }

  mTEST_ASSERT_EQUAL(0, i);

  mTEST_RETURN_SUCCESS();
}

mTEST(mDefer, TestExecuteLambdaOnError)
{
  mResult mSTDRESULT = mR_Failure;
  size_t i = 0;

  {
    mDEFER_ON_ERROR(i++);
    mTEST_ASSERT_EQUAL(0, i);
  }

  mTEST_ASSERT_EQUAL(1, i);

  mTEST_RETURN_SUCCESS();
}

mTEST(mDefer, TestExecuteLambdaOnSuccess)
{
  mResult mSTDRESULT = mR_Success;
  size_t i = 0;

  {
    mDEFER_ON_SUCCESS(i++);
    mTEST_ASSERT_EQUAL(0, i);
  }

  mTEST_ASSERT_EQUAL(1, i);

  mTEST_RETURN_SUCCESS();
}

mTEST(mDefer, TestExecuteLambdaOnNoSuccess)
{
  mResult mSTDRESULT = mR_Failure;
  size_t i = 0;

  {
    mDEFER_ON_SUCCESS(i++);
    mTEST_ASSERT_EQUAL(0, i);
  }

  mTEST_ASSERT_EQUAL(0, i);

  mTEST_RETURN_SUCCESS();
}

mTEST(mDefer, TestExecuteIncrementPtrValueOnNoError)
{
  mResult mSTDRESULT = mR_Success;
  size_t value = 0;
  size_t *pPtr = &value;

  {
    mDEFER_CALL_ON_ERROR(pPtr, mIncrementPtrValue);
    mTEST_ASSERT_EQUAL(0, value);
  }

  mTEST_ASSERT_EQUAL(0, value);

  mTEST_RETURN_SUCCESS();
}

mTEST(mDefer, TestExecuteIncrementPtrValueOnError)
{
  mResult mSTDRESULT = mR_Failure;
  size_t value = 0;
  size_t *pPtr = &value;

  {
    mDEFER_CALL_ON_ERROR(pPtr, mIncrementPtrValue);
    mTEST_ASSERT_EQUAL(0, value);
  }

  mTEST_ASSERT_EQUAL(1, value);

  mTEST_RETURN_SUCCESS();
}

mTEST(mDefer, TestExecuteIncrementPtrValueOnSuccess)
{
  mResult mSTDRESULT = mR_Success;
  size_t value = 0;
  size_t *pPtr = &value;

  {
    mDEFER_CALL_ON_SUCCESS(pPtr, mIncrementPtrValue);
    mTEST_ASSERT_EQUAL(0, value);
  }

  mTEST_ASSERT_EQUAL(1, value);

  mTEST_RETURN_SUCCESS();
}

mTEST(mDefer, TestExecuteIncrementPtrValueOnNoSuccess)
{
  mResult mSTDRESULT = mR_Failure;
  size_t value = 0;
  size_t *pPtr = &value;

  {
    mDEFER_CALL_ON_SUCCESS(pPtr, mIncrementPtrValue);
    mTEST_ASSERT_EQUAL(0, value);
  }

  mTEST_ASSERT_EQUAL(0, value);

  mTEST_RETURN_SUCCESS();
}
