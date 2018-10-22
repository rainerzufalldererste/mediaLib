#ifndef mTestLib_h__
#define mTestLib_h__

#include "mediaLib.h"
#include "gtest/gtest.h"

//#define mDEBUG_TESTS

#if defined(mDEBUG_TESTS) && defined(GIT_BUILD)
static_assert(false, "mDEBUG_TESTS has to be turned off for Buildserver builds.");
#endif

#ifdef mDEBUG_TESTS
#define mTEST_FAIL() \
  do \
  { printf("Test Failed at " __FUNCTION__ " in File '" __FILE__ "' Line %" PRIi32 ".\n", __LINE__); \
    __debugbreak(); \
  } while(0)

#define mTEST_ASSERT_EQUAL(a, b) \
  do \
  { auto a_ = (a); \
    auto b_ = (b); \
    \
    if (!(a_ == b_)) \
    { printf("Test Failed on '" #a " == " #b "' at " __FUNCTION__ " in File '" __FILE__ "' Line %" PRIi32 ".\n", __LINE__); \
      __debugbreak(); \
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
    { printf("Test Failed on '" #a " != " #b "' at " __FUNCTION__ " in File '" __FILE__ "' Line %" PRIi32 ".\n", __LINE__); \
      __debugbreak(); \
    } \
  } while(0)

#define mTEST_ASSERT_SUCCESS(functionCall) \
  do \
  { mResult __result = (functionCall); \
    \
    if (mFAILED(__result)) \
    { mString resultString; \
      if (mFAILED(mResult_ToString(__result, &resultString))) \
        printf("Test Failed on 'mFAILED(%s)' with invalid result [0x%" PRIx64 "] at " __FUNCTION__ " in File '"  __FILE__ "' Line %" PRIi32 ".\n", #functionCall, (uint64_t)__result, __LINE__); \
      else \
        printf("Test Failed on 'mFAILED(%s)' with Result '%s' [0x%" PRIx64 "] at " __FUNCTION__ " in File '" __FILE__ "' Line %" PRIi32 ".\n", #functionCall, resultString.c_str(), (uint64_t)__result, __LINE__); \
      __debugbreak(); \
    } \
  } while(0)
  

#else
#define mTEST_FAIL() ASSERT_TRUE(false)
#define mTEST_ASSERT_EQUAL(a, b) ASSERT_EQ(a, b)
#define mTEST_ASSERT_TRUE(a) ASSERT_TRUE(a)
#define mTEST_ASSERT_FALSE(a) ASSERT_FALSE(a)
#define mTEST_ASSERT_NOT_EQUAL(a, b) ASSERT_TRUE((a) != (b))
#define mTEST_ASSERT_SUCCESS(functionCall) mTEST_ASSERT_EQUAL(mR_Success, functionCall)
#endif

#define mTEST(Component, TestCase) TEST(Component, TestCase) 

extern size_t mTestDestructible_Count;

#define mTEST_ALLOCATOR_SETUP() \
  mTestDestructible_Count = 0; \
  mAllocator __testAllocator; \
  mTEST_ASSERT_SUCCESS(mTestAllocator_Create(&__testAllocator)); \
  mDEFER_CALL(&__testAllocator, mAllocator_Destroy); \
  mAllocator *pAllocator = &__testAllocator; \
  mUnused(pAllocator); \
  { 

#define mTEST_ALLOCATOR_ZERO_CHECK() \
  } \
  do \
  { size_t allocationCount = (size_t)-1; \
    mTEST_ASSERT_SUCCESS(mTestAllocator_GetCount(&__testAllocator, &allocationCount)); \
    mTEST_ASSERT_EQUAL(allocationCount, 0); \
  } while (0)

mFUNCTION(mTestLib_RunAllTests, int *pArgc, char **pArgv);

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

  inline mTestCppClass(mTestCppClass &&move)
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

  inline mTestCppClass & operator = (mTestCppClass &&move)
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
