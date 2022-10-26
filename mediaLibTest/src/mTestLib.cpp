#include "mTestLib.h"

#include <chrono>

#include "mDebugSymbolInfo.h"

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

void _HandleSignal(OPTIONAL IN _EXCEPTION_POINTERS *pExceptionInfo)
{
  mUnused(pExceptionInfo);

  char stackTrace[1024 * 16];

  if (mSUCCEEDED(mDebugSymbolInfo_GetStackTrace(stackTrace, mARRAYSIZE(stackTrace))))
    printf("Stack Trace: \n%s\n", stackTrace);
  else
    puts("Failed to get stacktrace.");

  fflush(stdout);
}

BOOL WINAPI _SignalHandler(DWORD type)
{
  printf("Signal raised: 0x%" PRIX32 ".\n", type);

  _HandleSignal(nullptr);

  mFlushOutput();
  mResetOutputFile();

  return TRUE;
}

LONG WINAPI TopLevelExceptionHandler(IN _EXCEPTION_POINTERS *pExceptionInfo)
{
  printf("Exception raised: 0x%" PRIX32 " at 0x%" PRIX64 ".\n", pExceptionInfo->ExceptionRecord->ExceptionCode, (uint64_t)pExceptionInfo->ExceptionRecord->ExceptionAddress);

  _HandleSignal(pExceptionInfo);

  mFlushOutput();
  mResetOutputFile();

  return EXCEPTION_CONTINUE_SEARCH;
}

void SetupSignalHandler()
{
  SetUnhandledExceptionFilter(TopLevelExceptionHandler);

  if (0 == SetConsoleCtrlHandler(_SignalHandler, TRUE))
    mPRINT_ERROR("Failed to set ConsoleCrtHandler.");
}

void mTestLib_Initialize()
{
  if (!IsInitialized)
  {
    CoInitialize(nullptr);
    atexit(_CallCouninitialize);

    SetupSignalHandler();
  }

  IsInitialized = true;
}

static const char mTestLib_OnlyArgument[] = "--only";
static const char mTestLib_ExcludeArgument[] = "--exclude";

mFUNCTION(mTestLib_RunAllTests, int32_t *pArgc, const char **pArgv)
{
  mTestLib_Initialize();

  char buffer[1024];

#ifdef mDEBUG_TESTS
  mUnused(pArgc, pArgv);

  const char *onlyTestNamePattern = nullptr;
  const char *excludeTestNamePattern = nullptr;
  
  if (*pArgc > 1)
  {
    size_t remainingArgs = *pArgc - 1;
    size_t argIndex = 1;

    while (remainingArgs > 0)
    {
      if (strcmp(pArgv[argIndex], mTestLib_OnlyArgument) == 0 && remainingArgs >= 2)
      {
        onlyTestNamePattern = pArgv[argIndex + 1];
        argIndex += 2;
        remainingArgs -= 2;
      }
      else if (strcmp(pArgv[argIndex], mTestLib_ExcludeArgument) == 0 && remainingArgs >= 2)
      {
        excludeTestNamePattern = pArgv[argIndex + 1];
        argIndex += 2;
        remainingArgs -= 2;
      }
      else
      {
        snprintf(buffer, sizeof(buffer), "Invalid Parameter '%s'. Aborting.\n", pArgv[argIndex]);
        mPrintToOutputArray(buffer);
        return mR_InvalidParameter;
      }
    }
  }

  std::vector<std::tuple<std::string, std::string, mResult>> failedTests;

  mSetConsoleColour(mCC_BrightBlue, mCC_Black);
  mPrintToOutputArray("[START OF TESTS]\n\n");
  mResetConsoleColour();

  const auto &beforeTests = std::chrono::high_resolution_clock::now();
  size_t testCount = 0;

  for (const auto &test : mTest_TestContainer())
  {
    ++testCount;

    if (onlyTestNamePattern != nullptr)
    {
      if (nullptr == strstr(std::get<0>(test).c_str(), onlyTestNamePattern) && nullptr == strstr(std::get<1>(test).c_str(), onlyTestNamePattern))
      {
        mSetConsoleColour(mCC_BrightGreen, mCC_Black);
        mPrintToOutputArray("[SKIPPING    ]");
        mResetConsoleColour();
        snprintf(buffer, sizeof(buffer), "  %s : %s\n", std::get<0>(test).c_str(), std::get<1>(test).c_str());
        mPrintToOutputArray(buffer);

        continue;
      }
    }

    if (excludeTestNamePattern != nullptr)
    {
      if (nullptr != strstr(std::get<0>(test).c_str(), excludeTestNamePattern) || nullptr != strstr(std::get<1>(test).c_str(), excludeTestNamePattern))
      {
        mSetConsoleColour(mCC_BrightGreen, mCC_Black);
        mPrintToOutputArray("[SKIPPING    ]");
        mResetConsoleColour();
        snprintf(buffer, sizeof(buffer), "  %s : %s\n", std::get<0>(test).c_str(), std::get<1>(test).c_str());
        mPrintToOutputArray(buffer);

        continue;
      }
    }

    mSetConsoleColour(mCC_BrightGreen, mCC_Black);
    mPrintToOutputArray("[RUNNING TEST]");
    mResetConsoleColour();

    snprintf(buffer, sizeof(buffer), "  %s : %s\n", std::get<0>(test).c_str(), std::get<1>(test).c_str());
    mPrintToOutputArray(buffer);

    snprintf(buffer, sizeof(buffer), "\r(Test %" PRIu64 " / %" PRIu64 ")", testCount, mTest_TestContainer().size());
    mPrintToOutputArray(buffer);

    mResult result = mR_Failure;
    size_t milliseconds = 0;

    {
#ifdef GIT_BUILD
      mPrintCallbackFunc *pPrintCallbackTmp = mPrintCallback;
      mPrintCallback = nullptr;
      mDEFER(pPrintCallbackTmp = pPrintCallbackTmp);
#endif

#ifdef _DEBUG
      try
#endif
      {
        const auto &start = std::chrono::high_resolution_clock::now();

        result = std::get<2>(test)();

        milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
      }
#ifdef _DEBUG
      catch (const std::exception &e)
      {
        snprintf(buffer, sizeof(buffer), "Test Failed. Captured Exception '%s'\n", e.what());
        mPrintToOutputArray(buffer);
      }
      catch (...)
      {
        mPrintToOutputArray("Test Failed. Captured Unknown Exception.\n");
      }
#endif
    }

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
#ifdef mTEST_STORE_ALLOCATIONS
constexpr size_t mTestAllocator_BeginOffsetRaw = 16;
constexpr size_t mTestAllocator_BeginOffset = mTestAllocator_BeginOffsetRaw + sizeof(mDebugSymbolInfo_CallStack);
#else
constexpr size_t mTestAllocator_BeginOffset = 16;
#endif
constexpr size_t mTestAllocator_EndOffset = 16;
constexpr uint8_t mTestAllocator_TestFlag = 0xCE;
uint8_t mTestAllocator_TestFlagChunk[mTestAllocator_EndOffset];

#ifdef mTEST_STORE_ALLOCATIONS
struct Bucket
{
  size_t capacity;
  size_t *pValues;
};
#endif

struct mTestAllocatorUserData
{
  mAllocator *pSelf;
  volatile size_t allocationCount;
#ifdef mTEST_STORE_ALLOCATIONS
  Bucket buckets[1024];
#endif
};

#ifdef mTEST_STORE_ALLOCATIONS
inline size_t Hash(size_t value)
{
  const uint64_t m = 0xC6A4A7935BD1E995ULL;
  const uint64_t seed = 0x0B9EADD924CD36D0ULL;
  const int32_t r = 47;

  uint64_t h = seed ^ (8 * m);

  uint64_t k = value;

  k *= m;
  k ^= k >> r;
  k *= m;

  h ^= k;
  h *= m;

  h ^= h >> r;
  h *= m;
  h ^= h >> r;

  return h;
}

void AddToBucket(void *pUsrDat, uint8_t *pData)
{
  const size_t value = reinterpret_cast<size_t>(pData);
  mTestAllocatorUserData *pUserData = reinterpret_cast<mTestAllocatorUserData *>(pUsrDat);
  Bucket &bucket = pUserData->buckets[Hash(value) % mARRAYSIZE(pUserData->buckets)];
  
  if (bucket.capacity == 0)
  {
    const size_t newCapacity = 16;
    bucket.pValues = reinterpret_cast<size_t *>(malloc(sizeof(size_t) * newCapacity));
    bucket.capacity = newCapacity;
    
    if (bucket.pValues == nullptr)
    {
      puts("Failed to allocate buckets.");
      fflush(stdout);
      __debugbreak();
    }

    bucket.pValues[0] = value;
    return;
  }

  for (size_t i = 0; i < bucket.capacity; i++)
  {
    if (bucket.pValues[i] == 0)
    {
      bucket.pValues[i] = value;
      return;
    }
  }

  size_t oldCapacity = bucket.capacity;
  const size_t newCapacity = oldCapacity * 4;

  bucket.pValues = reinterpret_cast<size_t *>(realloc(bucket.pValues, sizeof(size_t) * newCapacity));
  bucket.capacity = newCapacity;

  if (bucket.pValues == nullptr)
  {
    puts("Failed to allocate more buckets.");
    fflush(stdout);
    __debugbreak();
  }

  bucket.pValues[oldCapacity] = value;
  memset(bucket.pValues + oldCapacity + 1, 0, (newCapacity - oldCapacity - 1) * sizeof(size_t));
}

void RemoveFromBucket(void *pUsrDat, const uint8_t *pData)
{
  if (pData == nullptr)
    return;

  const size_t value = reinterpret_cast<size_t>(pData);
  mTestAllocatorUserData *pUserData = reinterpret_cast<mTestAllocatorUserData *>(pUsrDat);
  Bucket &bucket = pUserData->buckets[Hash(value) % mARRAYSIZE(pUserData->buckets)];

  if (bucket.capacity == 0)
    return;

  for (size_t i = 0; i < bucket.capacity; i++)
  {
    if (bucket.pValues[i] == value)
    {
      bucket.pValues[i] = 0;
      return;
    }
  }
}
#endif

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mTestAllocator_Alloc, OUT uint8_t **ppData, const size_t size, const size_t count, IN void *pUserData)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mAlloc(ppData, size * count + mTestAllocator_BeginOffset + mTestAllocator_EndOffset));
  ++(reinterpret_cast<mTestAllocatorUserData *>(pUserData))->allocationCount;

  *reinterpret_cast<size_t *>(*ppData) = size * count;
#ifdef mTEST_STORE_ALLOCATIONS
  mERROR_CHECK(mDebugSymbolInfo_GetCallStack(reinterpret_cast<mDebugSymbolInfo_CallStack *>(*ppData + sizeof(size_t))));
#endif
  *ppData += mTestAllocator_BeginOffset;
#ifdef mTEST_STORE_ALLOCATIONS
  AddToBucket(pUserData, *ppData);
#endif

  mERROR_CHECK(mMemset(*ppData + size * count, mTestAllocator_EndOffset, mTestAllocator_TestFlag));

  mRETURN_SUCCESS();
}

mFUNCTION(mTestAllocator_AllocZero, OUT uint8_t **ppData, const size_t size, const size_t count, IN void *pUserData)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mAllocZero(ppData, size * count + mTestAllocator_BeginOffset + mTestAllocator_EndOffset));
  ++(reinterpret_cast<mTestAllocatorUserData *>(pUserData))->allocationCount;

  *reinterpret_cast<size_t *>(*ppData) = size * count;
#ifdef mTEST_STORE_ALLOCATIONS
  mERROR_CHECK(mDebugSymbolInfo_GetCallStack(reinterpret_cast<mDebugSymbolInfo_CallStack *>(*ppData + sizeof(size_t))));
#endif
  *ppData += mTestAllocator_BeginOffset;
#ifdef mTEST_STORE_ALLOCATIONS
  AddToBucket(pUserData, *ppData);
#endif

  mERROR_CHECK(mMemset(*ppData + size * count, mTestAllocator_EndOffset, mTestAllocator_TestFlag));

  mRETURN_SUCCESS();
}

mFUNCTION(mTestAllocator_Realloc, OUT uint8_t **ppData, const size_t size, const size_t count, IN void *pUserData)
{
  mFUNCTION_SETUP();

#ifdef mTEST_STORE_ALLOCATIONS
  uint8_t *pOldPtr = *ppData;
#endif

  if (*ppData == nullptr)
  {
    ++(reinterpret_cast<mTestAllocatorUserData *>(pUserData))->allocationCount;
  }
  else
  {
    const size_t oldSize = *reinterpret_cast<size_t *>(*ppData - mTestAllocator_BeginOffset);

#ifdef mDEBUG_MEMORY_ALLOCATIONS
    if (memcmp(mTestAllocator_TestFlagChunk, *ppData + oldSize, mTestAllocator_EndOffset) != 0)
      mAllocatorDebugging_PrintMemoryAllocationInfo((reinterpret_cast<mTestAllocatorUserData *>(pUserData))->pSelf, *ppData);
#endif

    mASSERT(memcmp(mTestAllocator_TestFlagChunk, *ppData + oldSize, mTestAllocator_EndOffset) == 0, "Memory override detected");

    *ppData -= mTestAllocator_BeginOffset;
  }

  mDEFER_ON_ERROR(--(reinterpret_cast<mTestAllocatorUserData *>(pUserData))->allocationCount);
  mERROR_CHECK(mRealloc(ppData, size * count + mTestAllocator_BeginOffset + mTestAllocator_EndOffset));

  *reinterpret_cast<size_t *>(*ppData) = size * count;
  *ppData += mTestAllocator_BeginOffset;
#ifdef mTEST_STORE_ALLOCATIONS
  if (pOldPtr != *ppData)
  {
    RemoveFromBucket(pUserData, pOldPtr);
    AddToBucket(pUserData, *ppData);
  }
#endif

  mERROR_CHECK(mMemset(*ppData + size * count, mTestAllocator_EndOffset, mTestAllocator_TestFlag));

  mRETURN_SUCCESS();
}

mFUNCTION(mTestAllocator_Free, OUT uint8_t *pData, IN void *pUserData)
{
  mFUNCTION_SETUP();

  if (pData != nullptr)
  {
    const size_t size = *reinterpret_cast<size_t *>(pData - mTestAllocator_BeginOffset);

#ifdef mDEBUG_MEMORY_ALLOCATIONS
    if (memcmp(mTestAllocator_TestFlagChunk, pData + oldSize, mTestAllocator_EndOffset) != 0)
      mAllocatorDebugging_PrintMemoryAllocationInfo((reinterpret_cast<mTestAllocatorUserData *>(pUserData))->pSelf, pData);
#endif

    mASSERT(memcmp(mTestAllocator_TestFlagChunk, pData + size, mTestAllocator_EndOffset) == 0, "Memory override detected");
  }

  pData -= mTestAllocator_BeginOffset;
  mERROR_CHECK(mFree(pData));

  if(pData != nullptr)
    --(reinterpret_cast<mTestAllocatorUserData *>(pUserData))->allocationCount;

#ifdef mTEST_STORE_ALLOCATIONS
  RemoveFromBucket(pUserData, pData);
#endif

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
#ifdef mTEST_STORE_ALLOCATIONS
  mTestAllocatorUserData *pUsrDat = reinterpret_cast<mTestAllocatorUserData *>(pAllocator->pUserData);

  if (pUsrDat->allocationCount != 0)
  {
    mTestAllocator_PrintRemainingMemoryAllocations(pAllocator);
    mFAIL(mFormat("Memory leak detected: not all memory allocated by this allocator has been released (", pUsrDat->allocationCount, " Allocations)."));
  }

  for (size_t i = 0; i < mARRAYSIZE(pUsrDat->buckets); i++)
    free(pUsrDat->buckets[i].pValues);
#else
  mUnused(pAllocator);
  mASSERT((reinterpret_cast<mTestAllocatorUserData *>(pUserData))->allocationCount == 0, mFormat("Memory leak detected: not all memory allocated by this allocator has been released (", (reinterpret_cast<mTestAllocatorUserData *>(pUserData))->allocationCount, " Allocations)."));
#endif
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

#ifdef mTEST_STORE_ALLOCATIONS
mFUNCTION(mTestAllocator_PrintRemainingMemoryAllocations, mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mTestAllocatorUserData *pUserData = reinterpret_cast<mTestAllocatorUserData *>(pAllocator->pUserData);

  mERROR_IF(pUserData == nullptr, mR_ArgumentNull);

  for (size_t b = 0; b < mARRAYSIZE(pUserData->buckets); b++)
  {
    for (size_t i = 0; i < pUserData->buckets[b].capacity; i++)
    {
      const size_t value = pUserData->buckets[b].pValues[i];

      if (value == 0)
        continue;

      uint8_t *pData = reinterpret_cast<uint8_t *>(value);

      printf("0x%" PRIx64 ": (%" PRIu64 " bytes)\n", value, *reinterpret_cast<size_t *>(pData - mTestAllocator_BeginOffset));

      const mDebugSymbolInfo_CallStack *pCallStack = reinterpret_cast<mDebugSymbolInfo_CallStack *>(pData - mTestAllocator_BeginOffset + sizeof(size_t));
      char stackTrace[1024 * 8];
      
      if (mSUCCEEDED(mDebugSymbolInfo_GetStackTraceFromCallStack(pCallStack, stackTrace, mARRAYSIZE(stackTrace))))
        puts(stackTrace);

      puts("");
    }
  }

  mRETURN_SUCCESS();
}
#endif

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
