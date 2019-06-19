#include "mediaLib.h"
#include "mHashMap.h"

#ifdef mDEBUG_MEMORY_ALLOCATIONS
void mAllocatorDebugging_PrintOnExit()
{
  mPRINT("\nRemaining Memory Allocations:\n\n");

  mAllocatorDebugging_PrintAllRemainingMemoryAllocations();

  mPRINT("\nEnd of Remaining Memory Allocations.\n");

#ifdef mDEBUG_MEMORY_ALLOCATIONS_PRINT_ALLOCATIONS_WAIT_ON_EXIT
  getchar();
#endif
}

std::atomic<uint64_t> &mAllocatorDebugging_GetDebugMemoryAllocationCount()
{
  static std::atomic<uint64_t> instance;
  return instance;
}

std::recursive_mutex &mAllocatorDebugging_GetDebugMemoryAllocationMutex()
{
  static std::recursive_mutex instance;
  return instance;
};

std::map<mAllocator *, std::map<size_t, std::string>> &mAllocatorDebugging_GetDebugMemoryAllocationMap()
{
  static std::map<mAllocator *, std::map<size_t, std::string>> instance;
  
#ifdef mDEBUG_MEMORY_ALLOCATIONS_PRINT_ALLOCATIONS_ON_EXIT
  static std::atomic<bool> enqueuedPrintOnExit = false;

  if (!enqueuedPrintOnExit)
  {
    atexit(mAllocatorDebugging_PrintOnExit);
    enqueuedPrintOnExit = true;
  }
#endif

  return instance;
};

void mAllocatorDebugging_PrintRemainingMemoryAllocations(IN mAllocator *pAllocator)
{
  mAllocatorDebugging_GetDebugMemoryAllocationMutex().lock();

  auto item = mAllocatorDebugging_GetDebugMemoryAllocationMap().find(pAllocator);
  
  if (item == mAllocatorDebugging_GetDebugMemoryAllocationMap().end())
  {
    mAllocatorDebugging_GetDebugMemoryAllocationMutex().unlock();
    return;
  }

  auto allocatedMemory = item->second;

  for (const auto &allocation : allocatedMemory)
    mLOG("[Memory Allocation at 0x%" PRIx64 "]: %s\n", (uint64_t)allocation.first, allocation.second.c_str());

  mAllocatorDebugging_GetDebugMemoryAllocationMutex().unlock();
}

void mAllocatorDebugging_PrintAllRemainingMemoryAllocations()
{
  mAllocatorDebugging_GetDebugMemoryAllocationMutex().lock();

  for (const auto &allocator : mAllocatorDebugging_GetDebugMemoryAllocationMap())
  {
    if (allocator.second.size() > 0)
    {
      mPRINT("Allocator 0x%" PRIx64 ":\n");

      mAllocatorDebugging_PrintRemainingMemoryAllocations(allocator.first);

      mPRINT("\n\n\n");
    }
  }

  mAllocatorDebugging_GetDebugMemoryAllocationMutex().unlock();
}

void mAllocatorDebugging_PrintMemoryAllocationInfo(IN mAllocator *pAllocator, IN const void *pAllocation)
{
  mAllocatorDebugging_GetDebugMemoryAllocationMutex().lock();

  auto item = mAllocatorDebugging_GetDebugMemoryAllocationMap().find(pAllocator);

  if (item == mAllocatorDebugging_GetDebugMemoryAllocationMap().end())
  {
    mLOG("ERROR: Allocator not found.\n");
    mAllocatorDebugging_GetDebugMemoryAllocationMutex().unlock();
    return;
  }

  auto allocatedMemory = item->second;

  auto allocation = allocatedMemory.find(reinterpret_cast<size_t>(pAllocation));

  if (allocation == allocatedMemory.end())
  {
    mLOG("ERROR: Allocation not found in this allocator.\n");
    mAllocatorDebugging_GetDebugMemoryAllocationMutex().unlock();
    return;
  }

  mLOG("[Memory Allocation at 0x%" PRIx64 "]: %s\n", (uint64_t)(*allocation).first, (*allocation).second.c_str());

  mAllocatorDebugging_GetDebugMemoryAllocationMutex().unlock();
}

void mAllocatorDebugging_StoreAllocateCall(IN mAllocator *pAllocator, IN const void *pData, IN const char *information)
{
  mAllocatorDebugging_GetDebugMemoryAllocationMutex().lock();
  mDEFER(mAllocatorDebugging_GetDebugMemoryAllocationMutex().unlock());

  auto entry = mAllocatorDebugging_GetDebugMemoryAllocationMap().find(pAllocator);

  if (entry == mAllocatorDebugging_GetDebugMemoryAllocationMap().end())
  {
    mAllocatorDebugging_GetDebugMemoryAllocationMap().insert(std::pair<mAllocator *, std::map<size_t, std::string>>(pAllocator, std::map<size_t, std::string>()));
    entry = mAllocatorDebugging_GetDebugMemoryAllocationMap().find(pAllocator);
  }

  if (entry->second.find((size_t)pData) == entry->second.end())
    entry->second.insert(std::pair<size_t, std::string>((size_t)pData, std::string(information)));
}

void mAllocatorDebugging_StoreReallocateCall(IN mAllocator *pAllocator, const size_t originalPointer, IN const void *pData, IN const char *information)
{
  mAllocatorDebugging_GetDebugMemoryAllocationMutex().lock();
  mDEFER(mAllocatorDebugging_GetDebugMemoryAllocationMutex().unlock());

  auto entry = mAllocatorDebugging_GetDebugMemoryAllocationMap().find(pAllocator);

  if (entry == mAllocatorDebugging_GetDebugMemoryAllocationMap().end())
  {
    mAllocatorDebugging_GetDebugMemoryAllocationMap().insert(std::pair<mAllocator *, std::map<size_t, std::string>>(pAllocator, std::map<size_t, std::string>()));
    entry = mAllocatorDebugging_GetDebugMemoryAllocationMap().find(pAllocator);
  }

  entry->second.erase(originalPointer);

  if (entry->second.find((size_t)pData) == entry->second.end())
    entry->second.insert(std::pair<size_t, std::string>((size_t)pData, std::string(information)));
}

void mAllocatorDebugging_StoreFreeCall(IN mAllocator *pAllocator, const size_t originalPointer)
{
  mAllocatorDebugging_GetDebugMemoryAllocationMutex().lock();
  mDEFER(mAllocatorDebugging_GetDebugMemoryAllocationMutex().unlock());

  auto entry = mAllocatorDebugging_GetDebugMemoryAllocationMap().find(pAllocator);

  if (entry == mAllocatorDebugging_GetDebugMemoryAllocationMap().end())
  {
    mAllocatorDebugging_GetDebugMemoryAllocationMap().insert(std::pair<mAllocator *, std::map<size_t, std::string>>(pAllocator, std::map<size_t, std::string>()));
    entry = mAllocatorDebugging_GetDebugMemoryAllocationMap().find(pAllocator);
  }

  entry->second.erase(originalPointer);
}

#endif

#ifndef MEDIA_LIB_CUSTOM_DEFAULT_ALLOCATOR
mAllocator mDefaultAllocator = mAllocator_StaticCreate(&mDefaultAllocator_Alloc, &mDefaultAllocator_Realloc, &mDefaultAllocator_Free, &mDefaultAllocator_Move, &mDefaultAllocator_Copy, &mDefaultAllocator_AllocZero);
mAllocator mDefaultTempAllocator = mDefaultAllocator;

mFUNCTION(mDefaultAllocator_Alloc, OUT uint8_t **ppData, const size_t size, const size_t count, IN void *)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mAlloc(ppData, size * count));

  mRETURN_SUCCESS();
}

mFUNCTION(mDefaultAllocator_AllocZero, OUT uint8_t **ppData, const size_t size, const size_t count, IN void *)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mAllocZero(ppData, size * count));

  mRETURN_SUCCESS();
}

mFUNCTION(mDefaultAllocator_Realloc, OUT uint8_t **ppData, const size_t size, const size_t count, IN void *)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mRealloc(ppData, size * count));

  mRETURN_SUCCESS();
}

mFUNCTION(mDefaultAllocator_Free, OUT uint8_t *pData, IN void *)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mFree(pData));

  mRETURN_SUCCESS();
}

mFUNCTION(mDefaultAllocator_Move, IN_OUT uint8_t *pDestimation, IN uint8_t *pSource, const size_t size, const size_t count, IN void *)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mMemmove(pDestimation, pSource, size * count));

  mRETURN_SUCCESS();
}

mFUNCTION(mDefaultAllocator_Copy, IN_OUT uint8_t *pDestimation, IN const uint8_t *pSource, const size_t size, const size_t count, IN void *)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mMemcpy(pDestimation, pSource, size * count));

  mRETURN_SUCCESS();
}
#endif

//////////////////////////////////////////////////////////////////////////

mPtr<mHashMap<size_t, nullptr_t>> nullAllocatorData = nullptr;

mFUNCTION(mNullAllocator_Alloc, OUT uint8_t **ppData, const size_t size, const size_t count, IN void *)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mAlloc(ppData, size * count));

  if (nullAllocatorData == nullptr)
    mERROR_CHECK(mHashMap_Create(&nullAllocatorData, &mDefaultAllocator, 32));

  nullptr_t null = nullptr;
  mERROR_CHECK(mHashMap_Add(nullAllocatorData, (size_t)*ppData, &null));

  mRETURN_SUCCESS();
}

mFUNCTION(mNullAllocator_Realloc, OUT uint8_t **ppData, const size_t size, const size_t count, IN void *)
{
  mFUNCTION_SETUP();

  if (*ppData == nullptr || nullAllocatorData == nullptr)
  {
    allocate:
    mERROR_CHECK(mNullAllocator_Alloc(ppData, size, count, nullptr));
    mRETURN_SUCCESS();
  }

  nullptr_t unused;
  const mResult result = mHashMap_Remove(nullAllocatorData, (size_t)*ppData, &unused);

  if (mFAILED(result))
  {
    if (result == mR_ResourceNotFound)
      goto allocate;
    else
      mRETURN_RESULT(result);
  }

  mERROR_CHECK(mRealloc(ppData, size * count));

  mRETURN_SUCCESS();
}

mFUNCTION(mNullAllocator_Free, uint8_t *pData, IN void *) 
{
  mFUNCTION_SETUP();

  mERROR_IF(pData == nullptr || nullAllocatorData == nullptr, mR_Success);

  nullptr_t unused;
  const mResult result = mHashMap_Remove(nullAllocatorData, (size_t)pData, &unused);

  if (mFAILED(result))
  {
    if (result == mR_ResourceNotFound)
      mRETURN_SUCCESS();
    else
      mRETURN_RESULT(result);
  }

  mERROR_CHECK(mFree(pData));

  mRETURN_SUCCESS();
}

mAllocator mNullAllocator = mAllocator_StaticCreate(mNullAllocator_Alloc, mNullAllocator_Realloc, mNullAllocator_Free);

//////////////////////////////////////////////////////////////////////////

mAllocator mAllocator_StaticCreate(IN mAllocator_AllocFunction *pAllocFunction, IN mAllocator_AllocFunction *pReallocFunction, IN mAllocator_FreeFunction *pFree, IN OPTIONAL mAllocator_MoveFunction *pMove, IN OPTIONAL mAllocator_CopyFunction *pCopy, IN OPTIONAL mAllocator_AllocFunction *pAllocZeroFunction, IN OPTIONAL mAllocator_DestroyAllocator *pDestroyAllocator, IN OPTIONAL void *pUserData)
{
  mAllocator allocator;

  mASSERT_DEBUG((pAllocFunction != nullptr || pAllocZeroFunction != nullptr) && pFree != nullptr && pReallocFunction != nullptr, "Invalid static allocator. Allocate/AllocateZero, Free and Realloc cannot be nullptr.");

  allocator.pAllocate = pAllocFunction;
  allocator.pAllocateZero = pAllocZeroFunction;
  allocator.pReallocate = pReallocFunction;
  allocator.pFree = pFree;
  allocator.pMove = pMove;
  allocator.pCopy = pCopy;
  allocator.pDestroyAllocator = pDestroyAllocator;
  allocator.pUserData = pUserData;

  allocator.initialized = true;

  return allocator;
}

mFUNCTION(mAllocator_Create, OUT mAllocator *pAllocator, IN mAllocator_AllocFunction *pAllocFunction, IN mAllocator_AllocFunction *pReallocFunction, IN mAllocator_FreeFunction *pFree, IN OPTIONAL mAllocator_MoveFunction *pMove /* = nullptr */, IN OPTIONAL mAllocator_CopyFunction *pCopy /* = nullptr */, IN OPTIONAL mAllocator_AllocFunction *pAllocZeroFunction /* = nullptr */, IN OPTIONAL mAllocator_DestroyAllocator *pDestroyAllocator /* = nullptr */, IN OPTIONAL void *pUserData /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAllocator == nullptr || (pAllocFunction == nullptr && pAllocZeroFunction == nullptr) || pReallocFunction == nullptr || pFree == nullptr, mR_ArgumentNull);

  pAllocator->pAllocate = pAllocFunction;
  pAllocator->pAllocateZero = pAllocZeroFunction;
  pAllocator->pReallocate = pReallocFunction;
  pAllocator->pFree = pFree;
  pAllocator->pMove = pMove;
  pAllocator->pCopy = pCopy;
  pAllocator->pDestroyAllocator = pDestroyAllocator;
  pAllocator->pUserData = pUserData;

  pAllocator->initialized = true;

  mRETURN_SUCCESS();
}

mFUNCTION(mAllocator_Destroy, IN_OUT mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAllocator == nullptr, mR_ArgumentNull);

  pAllocator->initialized = false;
  pAllocator->pDestroyAllocator(pAllocator, pAllocator->pUserData);

  mRETURN_SUCCESS();
}
