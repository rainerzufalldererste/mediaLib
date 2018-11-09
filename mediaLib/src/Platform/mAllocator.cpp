#include "mAllocator.h"
#include "mHashMap.h"

#ifdef mDEBUG_MEMORY_ALLOCATIONS
uint64_t mAllocatorDebugging_DebugMemoryAllocationCount = 0;
std::recursive_mutex mAllocatorDebugging_DebugMemoryAllocationMutex;
std::map<mAllocator *, std::map<size_t, std::string>> mAllocatorDebugging_DebugMemoryAllocationMap;

void mAllocatorDebugging_PrintRemainingMemoryAllocations(mAllocator *pAllocator)
{
  mAllocatorDebugging_DebugMemoryAllocationMutex.lock();

  auto item = mAllocatorDebugging_DebugMemoryAllocationMap.find(pAllocator);
  
  if (item == mAllocatorDebugging_DebugMemoryAllocationMap.end())
  {
    mAllocatorDebugging_DebugMemoryAllocationMutex.unlock();
    return;
  }

  auto allocatedMemory = item->second;

  for (const auto &allocation : allocatedMemory)
    mPRINT("0x%" PRIx64 ": %s\n\n", (uint64_t)allocation.first, allocation.second.c_str());

  mAllocatorDebugging_DebugMemoryAllocationMutex.unlock();
}

void mAllocatorDebugging_PrintAllRemainingMemoryAllocations()
{
  mAllocatorDebugging_DebugMemoryAllocationMutex.lock();

  for (const auto &allocator : mAllocatorDebugging_DebugMemoryAllocationMap)
  {
    mPRINT("Allocator 0x%" PRIx64 ":\n");

    mAllocatorDebugging_PrintRemainingMemoryAllocations(allocator.first);

    mPRINT("\n\n\n");
  }

  mAllocatorDebugging_DebugMemoryAllocationMutex.unlock();
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
      mERROR_IF(true, result);
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
      mERROR_IF(true, result);
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
