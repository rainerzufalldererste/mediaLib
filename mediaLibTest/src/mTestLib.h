#ifndef mTestLib_h__
#define mTestLib_h__

#include "mediaLib.h"
#include "gtest/gtest.h"

#define mDEBUG_TESTS

void mTest_PrintTestFailure(const char *text);
#define mTEST_PRINT_FAILURE(...) mPrintToFunction(&mTest_PrintTestFailure, __VA_ARGS__)

#ifdef mDEBUG_TESTS

#include <vector>
#include <tuple>
#include <string>

typedef mResult TestFunc(void);
std::vector<std::tuple<std::string, std::string, TestFunc *>> & mTest_TestContainer();

struct mTest_TestInit {};
mTest_TestInit mTest_CreateTestInit(const char *component, const char *testCase, TestFunc *pTestFunc);

#define mTEST_TESTCASENAME(Component, TestCase) __mTest_TestCase_COMPONENT__##Component##_TEST_CASE__##TestCase##_Func
#define mTEST(Component, TestCase) \
  mResult mTEST_TESTCASENAME(Component, TestCase)(); \
  const mTest_TestInit &mTestTestInit_COMPONENT__##Component##_TEST_CASE__##TestCase##_INIT__ = mTest_CreateTestInit(#Component, #TestCase, &mTEST_TESTCASENAME(Component, TestCase)); \
  mResult mTEST_TESTCASENAME(Component, TestCase)()

#define mTEST_RETURN_SUCCESS() return mR_Success;
#define mTEST_RETURN_FAILURE() return mR_Failure;

#define mTEST_FAIL() \
  do \
  { mTEST_PRINT_FAILURE("Test Failed\n at " __FUNCTION__ "\n in File '" __FILE__ "' Line " mSTRINGIFY_VALUE(__LINE__) ".\n"); \
    __debugbreak(); \
      mTEST_RETURN_FAILURE(); \
  } while(0)

#define mTEST_ASSERT_EQUAL(a, b) \
  do \
  { auto a_ = (a); \
    auto b_ = (b); \
    \
    if (!(a_ == b_)) \
    { mTEST_PRINT_FAILURE("Test Failed\n on '" #a " == " #b "'\n at " __FUNCTION__ "\n in File '" __FILE__ "' Line " mSTRINGIFY_VALUE(__LINE__) ".\n"); \
      __debugbreak(); \
      mTEST_RETURN_FAILURE(); \
    } \
  } while(0)

#define mTEST_ASSERT_TRUE(a) mTEST_ASSERT_EQUAL(a, true)
#define mTEST_ASSERT_FALSE(a) mTEST_ASSERT_EQUAL(a, false)

#define mTEST_ASSERT_NOT_EQUAL(a, b) \
  do \
  { auto a_ = (a); \
    auto b_ = (b); \
    \
    if (!(a_ != b_)) \
    { mTEST_PRINT_FAILURE("Test Failed\n on '" #a " != " #b "'\n at " __FUNCTION__ "\n in File '" __FILE__ "' Line " mSTRINGIFY_VALUE(__LINE__) ".\n"); \
      __debugbreak(); \
      mTEST_RETURN_FAILURE(); \
    } \
  } while(0)

#define mTEST_ASSERT_SUCCESS(functionCall) \
  do \
  { const mResult __result = (functionCall); \
    \
    if (mFAILED(__result)) \
    { mTEST_PRINT_FAILURE("Test Failed on 'mFAILED(" #functionCall ")'\n with Result '", mResult_ToString(__result), "' [0x", mFUInt<mFHex>(__result), "]\n at " __FUNCTION__ "\n in File '" __FILE__ "' Line " mSTRINGIFY_VALUE(__LINE__) ".\n"); \
      mDEBUG_BREAK(); \
      mTEST_RETURN_FAILURE(); \
    } \
  } while(0)
  

#else
#define mTEST_FAIL() ASSERT_TRUE(false)
#define mTEST_ASSERT_EQUAL(a, b) ASSERT_EQ(a, b)
#define mTEST_ASSERT_TRUE(a) ASSERT_TRUE(a)
#define mTEST_ASSERT_FALSE(a) ASSERT_FALSE(a)
#define mTEST_ASSERT_NOT_EQUAL(a, b) ASSERT_TRUE((a) != (b))
#define mTEST_ASSERT_SUCCESS(functionCall) mTEST_ASSERT_EQUAL(mR_Success, functionCall)

#define mTEST(Component, TestCase) TEST(Component, TestCase) 

#define mTEST_RETURN_SUCCESS() return
#define mTEST_RETURN_FAILURE() mTEST_FAIL()
#endif

#define mTEST_STATIC_ASSERT(x) static_assert(x, "ASSERTION FAILED: " #x)

inline bool mTest_FloatInRange(const double_t expected, const double_t value, const double_t variance)
{
  return variance >= mAbs(value - expected);
}

inline bool mTest_FloatEquals(const double_t expected, const double_t value)
{
  return mTest_FloatInRange(expected, value, 4.0 * (double_t)mMax(mSmallest<float_t>(), mSmallest<float_t>((float_t)mAbs(expected)) + mSmallest<float_t>((float_t)mAbs(value))));
}

#define mTEST_FLOAT_IN_RANGE(expected, value, variance) mTest_FloatInRange((expected), (value), (variance))
#define mTEST_FLOAT_EQUALS(expected, value) mTest_FloatEquals((expected), (value))
#define mTEST_ASSERT_FLOAT_EQUALS(expected, value) mTEST_ASSERT_TRUE(mTEST_FLOAT_EQUALS(expected, value))

extern size_t mTestDestructible_Count;

#define mTEST_ALLOCATOR_SETUP() \
  mTestDestructible_Count = 0; \
  mAllocator __testAllocator; \
  mTEST_ASSERT_SUCCESS(mTestAllocator_Create(&__testAllocator)); \
  mDEFER_CALL(&__testAllocator, mAllocator_Destroy); \
  mAllocator *pAllocator = &__testAllocator; \
  mUnused(pAllocator); \
  { 

#ifdef mDEBUG_MEMORY_ALLOCATIONS
#define mTEST_ALLOCATOR_ZERO_CHECK() \
  } \
  do \
  { size_t allocationCount = (size_t)-1; \
    mTEST_ASSERT_SUCCESS(mTestAllocator_GetCount(&__testAllocator, &allocationCount)); \
    if (allocationCount != 0) \
      mAllocatorDebugging_PrintRemainingMemoryAllocations(&__testAllocator); \
    mTEST_ASSERT_EQUAL(allocationCount, 0); \
    mTEST_RETURN_SUCCESS(); \
  } while (0)
#else
#define mTEST_ALLOCATOR_ZERO_CHECK() \
  } \
  do \
  { size_t allocationCount = (size_t)-1; \
    mTEST_ASSERT_SUCCESS(mTestAllocator_GetCount(&__testAllocator, &allocationCount)); \
    mTEST_ASSERT_EQUAL(allocationCount, 0); \
    mTEST_RETURN_SUCCESS(); \
  } while (0)
#endif

void mTestLib_Initialize();
mFUNCTION(mTestLib_RunAllTests, int32_t *pArgc, const char **pArgv);

mFUNCTION(mTestAllocator_Create, mAllocator *pTestAllocator);
mFUNCTION(mTestAllocator_GetCount, mAllocator *pAllocator, size_t *pCount);

struct mDummyDestructible
{
  bool destructed;
  size_t *pData;
  mAllocator *pAllocator;
  size_t index;
};

mFUNCTION(mDummyDestructible_Create, mDummyDestructible *pDestructable, mAllocator *pAllocator);
mFUNCTION(mDestruct, mDummyDestructible *pDestructable);

template <typename T>
struct mTemplatedDestructible
{
  bool destructed;
  T *pData;
  mAllocator *pAllocator;
  size_t index;
};

template <typename T>
inline mFUNCTION(mDummyDestructible_Create, mTemplatedDestructible<T> *pDestructable, mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pDestructable == nullptr, mR_ArgumentNull);
  mERROR_IF(pAllocator == nullptr, mR_InvalidParameter); // should be test allocator.

  pDestructable->pAllocator = pAllocator;
  mERROR_CHECK(mAllocator_Allocate(pAllocator, &pDestructable->pData, 1));
  pDestructable->destructed = false;
  *pDestructable->pData = mTestDestructible_Count;
  pDestructable->index = mTestDestructible_Count++;

  mRETURN_SUCCESS();
}

template <typename T>
inline mFUNCTION(mDestruct, mTemplatedDestructible<T> *pDestructable)
{
  mFUNCTION_SETUP();

  mERROR_IF(pDestructable == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mAllocator_FreePtr(pDestructable->pAllocator, &pDestructable->pData));
  pDestructable->destructed = true;

  mRETURN_SUCCESS();
}

class mTestCppClass
{
public:
  inline static size_t & GlobalCount()
  {
    static size_t count = 0;
    return count;
  };

  struct Data
  {
    size_t data = 0;

    struct RefCountContainer
    {
      size_t copied = 0;
      size_t moved = 0;
      size_t copyConstructed = 0;
      size_t moveConstructed = 0;
      size_t referenceCount = 0;
    } *pRef = nullptr;

  } *pData = nullptr;

  inline mTestCppClass(const size_t data = (size_t)-1)
  {
    pData = new Data();
    pData->data = data;

    pData->pRef = new Data::RefCountContainer();
    ++pData->pRef->referenceCount;

    GlobalCount()++;
  }

  inline mTestCppClass(const mTestCppClass &copy)
  {
    if (copy.pData == nullptr)
      return;

    pData = new Data();
    *pData = *copy.pData;
    ++pData->pRef->referenceCount;
    ++pData->pRef->copyConstructed;

    GlobalCount()++;
  }

  inline mTestCppClass (mTestCppClass &&move) noexcept
  {
    if (move.pData == nullptr)
      return;

    pData = new Data();
    *pData = *move.pData;

    ++pData->pRef->moveConstructed;
    ++pData->pRef->referenceCount;

    move.~mTestCppClass();

    GlobalCount()++;
  }

  inline mTestCppClass & operator = (const mTestCppClass &copy)
  {
    if (copy.pData == nullptr)
    {
      this->~mTestCppClass();
      return *this;
    }

    this->~mTestCppClass();
    pData = new Data();

    *pData = *copy.pData;
    ++pData->pRef->referenceCount;
    ++pData->pRef->copied;

    GlobalCount()++;

    return *this;
  }

  inline mTestCppClass & operator = (mTestCppClass &&move) noexcept
  {
    if (move.pData == nullptr)
    {
      this->~mTestCppClass();
      return *this;
    }

    this->~mTestCppClass();
    pData = new Data();

    *pData = *move.pData;
    ++pData->pRef->moved;
    ++pData->pRef->referenceCount;

    move.~mTestCppClass();

    GlobalCount()++;

    return *this;
  }

  inline ~mTestCppClass()
  {
    if (pData == nullptr)
      return;

    --pData->pRef->referenceCount;

    if (pData->pRef->referenceCount == 0)
      delete pData->pRef;

    delete pData;

    pData = nullptr;

    GlobalCount()--;
  }
};

template <>
struct mIsTriviallyMemoryMovable<mTestCppClass>
{
  static constexpr bool value = true;
};

#endif // mTestLib_h__
