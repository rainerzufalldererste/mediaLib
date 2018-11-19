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
}

mTEST(mDefer, TestExecuteAssigned)
{
  size_t i = 0;

  {
    mDefer<size_t> defer;

    {
      defer = mDefer_Create([&]() {i++;});
      mTEST_ASSERT_EQUAL(0, i);
    }

    mTEST_ASSERT_EQUAL(0, i);
  }

  mTEST_ASSERT_EQUAL(1, i);
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
}

mTEST(mDefer, TestExecuteAssignedIncrementPtrValue)
{
  size_t value = 0;
  size_t *pPtr = &value;

  {
    mDefer<size_t *> defer;

    {
      defer = mDefer_Create(mIncrementPtrValue, pPtr);
      mTEST_ASSERT_EQUAL(0, value);
    }

    mTEST_ASSERT_EQUAL(0, value);
  }

  mTEST_ASSERT_EQUAL(1, value);
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
}

mTEST(mDefer, TestExecuteAssignedOnNoError)
{
  mResult mSTDRESULT = mR_Success;
  size_t i = 0;

  {
    mDefer<size_t> defer;

    {
      defer = mDefer_Create([&]() {i++;}, &mSTDRESULT);
      mTEST_ASSERT_EQUAL(0, i);
    }

    mTEST_ASSERT_EQUAL(0, i);
  }

  mTEST_ASSERT_EQUAL(0, i);
}

mTEST(mDefer, TestExecuteAssignedOnError)
{
  mResult mSTDRESULT = mR_Failure;
  size_t i = 0;

  {
    mDefer<size_t> defer;

    {
      defer = mDefer_Create([&]() {i++;}, &mSTDRESULT);
      mTEST_ASSERT_EQUAL(0, i);
    }

    mTEST_ASSERT_EQUAL(0, i);
  }

  mTEST_ASSERT_EQUAL(1, i);
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
}

mTEST(mDefer, TestExecuteAssignedIncrementPtrValueOnNoError)
{
  mResult mSTDRESULT = mR_Success;
  size_t value = 0;
  size_t *pPtr = &value;

  {
    mDefer<size_t *> defer;

    {
      defer = mDefer_Create(mIncrementPtrValue, pPtr, &mSTDRESULT);
      mTEST_ASSERT_EQUAL(0, value);
    }

    mTEST_ASSERT_EQUAL(0, value);
  }

  mTEST_ASSERT_EQUAL(0, value);
}

mTEST(mDefer, TestExecuteAssignedIncrementPtrValueOnError)
{
  mResult mSTDRESULT = mR_Failure;
  size_t value = 0;
  size_t *pPtr = &value;

  {
    mDefer<size_t *> defer;

    {
      defer = mDefer_Create(mIncrementPtrValue, pPtr, &mSTDRESULT);
      mTEST_ASSERT_EQUAL(0, value);
    }

    mTEST_ASSERT_EQUAL(0, value);
  }

  mTEST_ASSERT_EQUAL(1, value);
}
