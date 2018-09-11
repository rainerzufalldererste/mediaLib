// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef mTestLib_h__
#define mTestLib_h__

#include "default.h"
#include "gtest/gtest.h"

//#define mDEBUG_TESTS

#ifdef mDEBUG_TESTS
#define mTEST_FAIL() \
  do \
  { printf("Test Failed at " __FUNCTION__ " in File '" __FILE__ "' Line %" PRIi32 ".", __LINE__); \
    __debugbreak(); \
  } while(0)

#define mTEST_ASSERT_EQUAL(a, b) \
  do \
  { auto a_ = (a); \
    auto b_ = (b); \
    \
    if (a_ != b_) \
    { printf("Test Failed on '" #a " == " #b "' at " __FUNCTION__ " in File '" __FILE__ "' Line %" PRIi32 ".", __LINE__); \
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
    if (a_ == b_) \
    { printf("Test Failed on '" #a " != " #b "' at " __FUNCTION__ " in File '" __FILE__ "' Line %" PRIi32 ".", __LINE__); \
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
        printf("Test Failed on 'mFAILED(" #functionCall ")' with invalid result [0x%" PRIx64 "] at " __FUNCTION__ " in File '" __FILE__ "' Line %" PRIi32 ".", (uint64_t)__result, __LINE__); \
      else \
        printf("Test Failed on 'mFAILED(" #functionCall ")' with Result '%s' [0x%" PRIx64 "] at " __FUNCTION__ " in File '" __FILE__ "' Line %" PRIi32 ".", resultString.c_str(), (uint64_t)__result, __LINE__); \
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

#define mTEST_ALLOCATOR_SETUP() \
  mAllocator __testAllocator; \
  mTEST_ASSERT_SUCCESS(mTestAllocator_Create(&__testAllocator)); \
  mDEFER_DESTRUCTION(&__testAllocator, mAllocator_Destroy); \
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
};

mFUNCTION(mDummyDestructible_Create, mDummyDestructible *pDestructable, mAllocator *pAllocator);
mFUNCTION(mDestruct, mDummyDestructible *pDestructable);

template <typename T>
struct mTemplatedDestructible
{
  bool destructed;
  T *pData;
  mAllocator *pAllocator;
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

#endif // mTestLib_h__
