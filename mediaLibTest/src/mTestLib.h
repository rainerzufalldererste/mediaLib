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

#define mDEBUG_TESTS

#ifdef mDEBUG_TESTS
#define mTEST_FAIL() do { printf("Test Failed at " __FUNCTION__ " in File '" __FILE__ "' Line %" PRIi32 ".", __LINE__); __debugbreak(); } while(0)
#define mTEST_ASSERT_EQUAL(a, b) do { auto a_ = (a); auto b_ = (b); if (a_ != b_) { printf("Test Failed on '" #a " == " #b "' at " __FUNCTION__ " in File '" __FILE__ "' Line %" PRIi32 ".", __LINE__); __debugbreak();} } while(0)
#define mTEST_ASSERT_TRUE(a) mTEST_ASSERT_EQ(a, true)
#define mTEST_ASSERT_FALSE(a) mTEST_ASSERT_EQ(a, false)
#define mTEST_ASSERT_NOT_EQUAL(a, b) do { auto a_ = (a); auto b_ = (b); if (a_ == b_) { printf("Test Failed on '" #a " != " #b "' at " __FUNCTION__ " in File '" __FILE__ "' Line %" PRIi32 ".", __LINE__); __debugbreak();} } while(0)
#else
#define mTEST_FAIL() ASSERT_TRUE(false)
#define mTEST_ASSERT_EQUAL(a, b) ASSERT_EQ(a, b)
#define mTEST_ASSERT_TRUE(a) ASSERT_TRUE(a)
#define mTEST_ASSERT_FALSE(a) ASSERT_FALSE(a)
#define mTEST_ASSERT_NOT_EQUAL(a, b) ASSERT_TRUE((a) != (b))
#endif

#define mTEST(Component, TestCase) TEST(Component, TestCase) 
#define mTEST_ASSERT_SUCCESS(functionCall) mTEST_ASSERT_EQUAL(mR_Success, functionCall)

mFUNCTION(mTestLib_RunAllTests, int *pArgc, char **pArgv);

mFUNCTION(mTestAllocator_Create, mAllocator *pTestAllocator);
mFUNCTION(mTestAllocator_GetCount, mAllocator *pAllocator, size_t *pCount);

#endif // mTestLib_h__
