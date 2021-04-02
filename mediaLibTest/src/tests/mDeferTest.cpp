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

static size_t testValue0;

void IncrementTestValue0()
{
  testValue0++;
}

mTEST(mDefer, TestDeferCall0)
{
  testValue0 = 1234;

  {
    mDEFER_CALL_0(IncrementTestValue0);
    mTEST_ASSERT_EQUAL(1234, testValue0);
  }

  mTEST_ASSERT_EQUAL(1235, testValue0);

  mTEST_RETURN_SUCCESS();
}

template <typename T>
void IncrementDecrementValues(T *pValue0, T *pValue1)
{
  (*pValue0)++;
  (*pValue1)--;
}

mTEST(mDefer, TestDeferCall2)
{
  size_t value0 = 1234;
  size_t value1 = 4321;

  {
    mDEFER_CALL_2(IncrementDecrementValues, &value0, &value1);
    mTEST_ASSERT_EQUAL(1234, value0);
    mTEST_ASSERT_EQUAL(4321, value1);
  }

  mTEST_ASSERT_EQUAL(1235, value0);
  mTEST_ASSERT_EQUAL(4320, value1);

  mTEST_RETURN_SUCCESS();
}

template <typename T>
void IncrementDecrementSquareValues(T *pValue0, T *pValue1, T *pValue2)
{
  (*pValue0)++;
  (*pValue1)--;
  *pValue2 *= *pValue2;
}

mTEST(mDefer, TestDeferCall3)
{
  size_t value0 = 1234;
  size_t value1 = 4321;
  size_t value2 = 9876;

  {
    mDEFER_CALL_3(IncrementDecrementSquareValues, &value0, &value1, &value2);
    mTEST_ASSERT_EQUAL(1234, value0);
    mTEST_ASSERT_EQUAL(4321, value1);
    mTEST_ASSERT_EQUAL(9876, value2);
  }

  mTEST_ASSERT_EQUAL(1235, value0);
  mTEST_ASSERT_EQUAL(4320, value1);
  mTEST_ASSERT_EQUAL(97535376, value2);

  mTEST_RETURN_SUCCESS();
}
