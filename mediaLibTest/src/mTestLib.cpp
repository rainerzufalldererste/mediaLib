#include "mTestLib.h"

#include <chrono>

bool IsInitialized = false;

#ifdef mDEBUG_TESTS
std::vector<std::tuple<std::string, std::string, TestFunc *>> & mTest_TestContainer()
{
  static std::vector<std::tuple<std::string, std::string, TestFunc *>> instance;
  return instance;
}

mTest_TestInit mTest_CreateTestInit(const char *component, const char *testCase, TestFunc *pTestFunc)
{
  mTest_TestContainer().push_back(std::make_tuple(std::string(component), std::string(testCase), pTestFunc));

  return mTest_TestInit();
}
#endif

void mTest_PrintTestFailure(const char *text)
{
  mSetConsoleColour(mCC_BrightYellow, mCC_DarkRed);
  mPrintToOutputArray("\n[ERROR]");
  mSetConsoleColour(mCC_BrightRed, mCC_Black);
  mPrintToOutputArray(" ");
  mPrintToOutput(text);
  mResetConsoleColour();
}

void _CallCouninitialize()
{
  CoUninitialize();
}

void mTestLib_Initialize()
{
  if (!IsInitialized)
  {
    CoInitialize(nullptr);
    atexit(_CallCouninitialize);
  }

  IsInitialized = true;
}

mFUNCTION(mTestLib_RunAllTests, int *pArgc, char **pArgv)
{
  mTestLib_Initialize();

#ifdef mDEBUG_TESTS
  mUnused(pArgc, pArgv);

  std::vector<std::tuple<std::string, std::string, mResult>> failedTests;

  mSetConsoleColour(mCC_BrightBlue, mCC_Black);
  mPrintToOutputArray("[START OF TESTS]\n\n");
  mResetConsoleColour();

  const auto &beforeTests = std::chrono::high_resolution_clock::now();
  size_t testCount = 0;

  char buffer[1024];

  for (const auto &test : mTest_TestContainer())
  {
    ++testCount;

    mSetConsoleColour(mCC_BrightGreen, mCC_Black);
    mPrintToOutputArray("[RUNNING TEST]");
    mResetConsoleColour();

    snprintf(buffer, sizeof(buffer), "  %s : %s\n", std::get<0>(test).c_str(), std::get<1>(test).c_str());
    mPrintToOutputArray(buffer);

    snprintf(buffer, sizeof(buffer), "\r(Test %" PRIu64 " / %" PRIu64 ")", testCount, mTest_TestContainer().size());
    mPrintToOutputArray(buffer);

    const auto &start = std::chrono::high_resolution_clock::now();

    const mResult result = std::get<2>(test)();

    const size_t milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();

    if (mSUCCEEDED(result))
    {
      mSetConsoleColour(mCC_BrightGreen, mCC_Black);
      mPrintToOutputArray("\r[TEST PASSED ]");
    }
    else
    {
      mSetConsoleColour(mCC_BrightYellow, mCC_DarkRed);
      mPrintToOutputArray("\r[TEST FAILED ]");
      failedTests.push_back(make_tuple(std::get<0>(test), std::get<1>(test), result));
    }

    mResetConsoleColour();
    snprintf(buffer, sizeof(buffer), "  %s : %s (in %" PRIu64 " ms)\n\n", std::get<0>(test).c_str(), std::get<1>(test).c_str(), milliseconds);
    mPrintToOutputArray(buffer);
  }

  const size_t afterTestsMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - beforeTests).count();

  mSetConsoleColour(mCC_BrightBlue, mCC_Black);
  mPrintToOutputArray("[END OF TESTS]\n");
  mResetConsoleColour();

  if (failedTests.size() == 0)
  {
    mSetConsoleColour(mCC_BrightGreen, mCC_Black);
    snprintf(buffer, sizeof(buffer), "\nALL %" PRIu64 " TESTS SUCCEEDED. (in %" PRIu64 " ms)\n", mTest_TestContainer().size(), afterTestsMilliseconds);
    mPrintToOutputArray(buffer);
    mResetConsoleColour();

    return mR_Success;
  }
  else
  {
    mSetConsoleColour(mCC_BrightRed, mCC_Black);
    snprintf(buffer, sizeof(buffer), "\n%" PRIu64 " / %" PRIu64 " TESTS FAILED: (in %" PRIu64 " ms)\n\n", failedTests.size(), mTest_TestContainer().size(), afterTestsMilliseconds);
    mPrintToOutputArray(buffer);
    mResetConsoleColour();

    for (const auto &failedTest : failedTests)
    {
      mSetConsoleColour(mCC_DarkGray, mCC_Black);
      mPrintToOutputArray(" ## ");
      mResetConsoleColour();

      snprintf(buffer, sizeof(buffer), "%s : %s with %s (0x%" PRIx64 ")\n", std::get<0>(failedTest).c_str(), std::get<1>(failedTest).c_str(), mResult_ToString(std::get<2>(failedTest)), (uint64_t)std::get<2>(failedTest));
      mPrintToOutputArray(buffer);
    }

    return mR_Failure;
  }
#else
  ::testing::InitGoogleTest(pArgc, pArgv);

  return RUN_ALL_TESTS() == 0 ? mR_Success : mR_Failure;
#endif
}

//////////////////////////////////////////////////////////////////////////
                                
size_t mTestDestructible_Count = 0;
const size_t mTestAllocator_BeginOffset = 16;
const size_t mTestAllocator_EndOffset = 16;
const uint8_t mTestAllocator_TestFlag = 0xCE;
uint8_t mTestAllocator_TestFlagChunk[mTestAllocator_EndOffset];

struct mTestAllocatorUserData
{
  mAllocator *pSelf;
  volatile size_t allocationCount;
};

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mTestAllocator_Alloc, OUT uint8_t **ppData, const size_t size, const size_t count, IN void *pUserData)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mAlloc(ppData, size * count + mTestAllocator_BeginOffset + mTestAllocator_EndOffset));
  ++(reinterpret_cast<mTestAllocatorUserData *>(pUserData))->allocationCount;

  *(size_t *)*ppData = size * count;
  *ppData += mTestAllocator_BeginOffset;

  mERROR_CHECK(mMemset(*ppData + size * count, mTestAllocator_EndOffset, mTestAllocator_TestFlag));

  mRETURN_SUCCESS();
}

mFUNCTION(mTestAllocator_AllocZero, OUT uint8_t **ppData, const size_t size, const size_t count, IN void *pUserData)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mAllocZero(ppData, size * count + mTestAllocator_BeginOffset + mTestAllocator_EndOffset));
  ++(reinterpret_cast<mTestAllocatorUserData *>(pUserData))->allocationCount;

  *(size_t *)*ppData = size * count;
  *ppData += mTestAllocator_BeginOffset;

  mERROR_CHECK(mMemset(*ppData + size * count, mTestAllocator_EndOffset, mTestAllocator_TestFlag));

  mRETURN_SUCCESS();
}

mFUNCTION(mTestAllocator_Realloc, OUT uint8_t **ppData, const size_t size, const size_t count, IN void *pUserData)
{
  mFUNCTION_SETUP();

  if (*ppData == nullptr)
  {
    ++(reinterpret_cast<mTestAllocatorUserData *>(pUserData))->allocationCount;
  }
  else
  {
    const size_t oldSize = *(size_t *)(*ppData - mTestAllocator_BeginOffset);

#ifdef mDEBUG_MEMORY_ALLOCATIONS
    if (memcmp(mTestAllocator_TestFlagChunk, *ppData + oldSize, mTestAllocator_EndOffset) != 0)
      mAllocatorDebugging_PrintMemoryAllocationInfo((reinterpret_cast<mTestAllocatorUserData *>(pUserData))->pSelf, *ppData);
#endif

    mASSERT(memcmp(mTestAllocator_TestFlagChunk, *ppData + oldSize, mTestAllocator_EndOffset) == 0, "Memory override detected");

    *ppData -= mTestAllocator_BeginOffset;
  }

  mDEFER_ON_ERROR(--(reinterpret_cast<mTestAllocatorUserData *>(pUserData))->allocationCount);
  mERROR_CHECK(mRealloc(ppData, size * count + mTestAllocator_BeginOffset + mTestAllocator_EndOffset));

  *(size_t *)*ppData = size * count;
  *ppData += mTestAllocator_BeginOffset;

  mERROR_CHECK(mMemset(*ppData + size * count, mTestAllocator_EndOffset, mTestAllocator_TestFlag));

  mRETURN_SUCCESS();
}

mFUNCTION(mTestAllocator_Free, OUT uint8_t *pData, IN void *pUserData)
{
  mFUNCTION_SETUP();

  if (pData != nullptr)
  {
    const size_t oldSize = *(size_t *)(pData - mTestAllocator_BeginOffset);

#ifdef mDEBUG_MEMORY_ALLOCATIONS
    if (memcmp(mTestAllocator_TestFlagChunk, pData + oldSize, mTestAllocator_EndOffset) != 0)
      mAllocatorDebugging_PrintMemoryAllocationInfo((reinterpret_cast<mTestAllocatorUserData *>(pUserData))->pSelf, pData);
#endif

    mASSERT(memcmp(mTestAllocator_TestFlagChunk, pData + oldSize, mTestAllocator_EndOffset) == 0, "Memory override detected");
  }

  pData -= mTestAllocator_BeginOffset;
  mERROR_CHECK(mFree(pData));

  if(pData != nullptr)
    --(reinterpret_cast<mTestAllocatorUserData *>(pUserData))->allocationCount;

  mRETURN_SUCCESS();
}

mFUNCTION(mTestAllocator_Destroy, IN mAllocator *pAllocator, IN void *pUserData)
{
  mFUNCTION_SETUP();

  mERROR_IF(pUserData == nullptr, mR_ArgumentNull);

#ifdef mDEBUG_MEMORY_ALLOCATIONS
  if ((reinterpret_cast<mTestAllocatorUserData *>(pUserData))->allocationCount != 0)
  {
    mAllocatorDebugging_PrintRemainingMemoryAllocations(pAllocator);
    mFAIL(mFormat("Memory leak detected: not all memory allocated by this allocator has been released (", (reinterpret_cast<mTestAllocatorUserData *>(pUserData))->allocationCount, " Allocations)."));
  }
#else
  mUnused(pAllocator);
  mASSERT((reinterpret_cast<mTestAllocatorUserData *>(pUserData))->allocationCount == 0, mFormat("Memory leak detected: not all memory allocated by this allocator has been released (", (reinterpret_cast<mTestAllocatorUserData *>(pUserData))->allocationCount, " Allocations)."));
#endif

  mERROR_CHECK(mFree(pUserData));

  mRETURN_SUCCESS();
}

mFUNCTION(mTestAllocator_Create, mAllocator *pTestAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTestAllocator == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mMemset(mTestAllocator_TestFlagChunk, mTestAllocator_EndOffset, mTestAllocator_TestFlag));

  mTestAllocatorUserData *pUserData = nullptr;
  mERROR_CHECK(mAllocZero(&pUserData, 1));
  pUserData->pSelf = pTestAllocator;

  mERROR_CHECK(mAllocator_Create(pTestAllocator, &mTestAllocator_Alloc, &mTestAllocator_Realloc, &mTestAllocator_Free, &mTestAllocator_AllocZero, &mTestAllocator_Destroy, pUserData));

  mRETURN_SUCCESS();
}

mFUNCTION(mTestAllocator_GetCount, mAllocator *pAllocator, size_t *pCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAllocator == nullptr || pAllocator->pUserData == nullptr || pCount == nullptr, mR_ArgumentNull);

  *pCount = (reinterpret_cast<mTestAllocatorUserData *>(pAllocator->pUserData))->allocationCount;

  mRETURN_SUCCESS();
}

mFUNCTION(mDummyDestructible_Create, mDummyDestructible *pDestructable, mAllocator *pAllocator)
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

mFUNCTION(mDestruct, mDummyDestructible *pDestructable)
{
  mFUNCTION_SETUP();

  mERROR_IF(pDestructable == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mAllocator_FreePtr(pDestructable->pAllocator, &pDestructable->pData));
  pDestructable->destructed = true;

  mRETURN_SUCCESS();
}
