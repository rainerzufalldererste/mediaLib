#ifndef mAllocator_h__
#define mAllocator_h__

#include "mResult.h"
#include "mediaLib.h"

//#define mDEBUG_MEMORY_ALLOCATIONS

#ifdef mDEBUG_MEMORY_ALLOCATIONS
//#define mDEBUG_MEMORY_ALLOCATIONS_PRINT_ALLOCATIONS
//#define mDEBUG_MEMORY_ALLOCATIONS_PRINT_ALLOCATIONS_ON_EXIT

#ifdef mDEBUG_MEMORY_ALLOCATIONS_PRINT_ALLOCATIONS_ON_EXIT
//#define mDEBUG_MEMORY_ALLOCATIONS_PRINT_ALLOCATIONS_WAIT_ON_EXIT
#endif

#if defined(GIT_BUILD)
#error mDEBUG_MEMORY_ALLOCATIONS cannot be enabled in a GIT_BUILD.
#endif

#include <map>
#include <mutex>
#include <atomic>

#include "mDebugSymbolInfo.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "ZGBcUKtRyNNbXwr0Bq0Tr1wN1cPTb8Tt22covdNAWifpg5JSinweE2CFb/XQl1mRCpGk7R8F14mir8aZ"
#endif

struct mAllocator;

std::atomic<uint64_t> &mAllocatorDebugging_GetDebugMemoryAllocationCount();

void mAllocatorDebugging_PrintRemainingMemoryAllocations(IN mAllocator *pAllocator);
void mAllocatorDebugging_PrintAllRemainingMemoryAllocations();
void mAllocatorDebugging_PrintMemoryAllocationInfo(IN mAllocator *pAllocator, IN const void *pAllocation);

void mAllocatorDebugging_StoreAllocateCall(IN mAllocator *pAllocator, IN const void *pData, IN const char *information);
void mAllocatorDebugging_StoreReallocateCall(IN mAllocator *pAllocator, const size_t originalPointer, IN const void *pData, IN const char *information);
void mAllocatorDebugging_StoreFreeCall(IN mAllocator *pAllocator, const size_t originalPointer);

void mAllocatorDebugging_ClearAllAllocations();
void mAllocatorDebugging_ClearAllocations(IN mAllocator *pAllocator);

void mAllocatorDebugging_SetStoreNewAllocations(const bool storeNewAllocations);
bool mAllocatorDebugging_GetStoreNewAllocations();
#endif

struct mAllocator;

// Parameters:
//   OUT uint8_t **ppData: pointer to pointer to allocate to.
//   const size_t size: size of a single object.
//   const size_t count: amount of objects.
//   IN void *pUserData: associated user data with the custom allocator.
typedef mResult(mAllocator_AllocFunction) (OUT uint8_t **, const size_t, const size_t, IN void *);

// Parameters:
//   IN uint8_t *ppData: pointer to free memory at.
//   IN void *pUserData: associated user data with the custom allocator.
typedef mResult(mAllocator_FreeFunction) (IN uint8_t *, IN void *);

// Parameters:
//   IN_OUT uint8_t *pDestination: pointer to the destination.
//   IN_OUT uint8_t *pSource: size of a single object.
//   const size_t size: size of a single object.
//   const size_t count: amount of objects.
//   IN void *pUserData: associated user data with the custom allocator.
typedef mResult(mAllocator_MoveFunction) (IN_OUT uint8_t *, IN_OUT uint8_t *, const size_t, const size_t, IN void *);

// Parameters:
//   IN_OUT uint8_t *pDestination: pointer to the destination.
//   IN const uint8_t *pSource: size of a single object.
//   const size_t size: size of a single object.
//   const size_t count: amount of objects.
//   IN void *pUserData: associated user data with the custom allocator.
typedef mResult(mAllocator_CopyFunction) (IN_OUT uint8_t *, IN const uint8_t *, const size_t, const size_t, IN void *);

// Parameters:
//   IN mAllocator *pAllocator: the allocator to be destroyed. (this pointer shouldn't be freed!)
//   IN void *pUserData: associated user data with the custom allocator.
typedef mResult(mAllocator_DestroyAllocator) (IN mAllocator *, IN void *);

mFUNCTION(mDefaultAllocator_Alloc, OUT uint8_t **ppData, const size_t size, const size_t count, IN void *pUserData = nullptr);
mFUNCTION(mDefaultAllocator_AllocZero, OUT uint8_t **ppData, const size_t size, const size_t count, IN void *pUserData = nullptr);
mFUNCTION(mDefaultAllocator_Realloc, OUT uint8_t **ppData, const size_t size, const size_t count, IN void *pUserData = nullptr);
mFUNCTION(mDefaultAllocator_Free, OUT uint8_t *pData, IN void *pUserData = nullptr);
mFUNCTION(mDefaultAllocator_Move, IN_OUT uint8_t *pDestimation, IN uint8_t *pSource, const size_t size, const size_t count, IN void *pUserData = nullptr);
mFUNCTION(mDefaultAllocator_Copy, IN_OUT uint8_t *pDestimation, IN const uint8_t *pSource, const size_t size, const size_t count, IN void *pUserData = nullptr);

extern mAllocator mDefaultAllocator;
extern mAllocator mDefaultTempAllocator;
extern mAllocator mNullAllocator; // This allocator will only free pointers it has allocated.

struct mAllocator
{
  mAllocator_AllocFunction *pAllocate = nullptr;
  mAllocator_AllocFunction *pAllocateZero = nullptr;
  mAllocator_AllocFunction *pReallocate = nullptr;
  mAllocator_FreeFunction *pFree = nullptr;
  mAllocator_DestroyAllocator *pDestroyAllocator = nullptr;

  void *pUserData = nullptr;
  bool initialized = false;
};

mAllocator mAllocator_StaticCreate(IN mAllocator_AllocFunction *pAllocFunction, IN mAllocator_AllocFunction *pReallocFunction, IN mAllocator_FreeFunction *pFree, IN OPTIONAL mAllocator_AllocFunction *pAllocZeroFunction = nullptr, IN OPTIONAL mAllocator_DestroyAllocator *pDestroyAllocator = nullptr, IN OPTIONAL void *pUserData = nullptr);

mFUNCTION(mAllocator_Create, OUT mAllocator *pAllocator, IN mAllocator_AllocFunction *pAllocFunction, IN mAllocator_AllocFunction *pReallocFunction, IN mAllocator_FreeFunction *pFree, IN OPTIONAL mAllocator_AllocFunction *pAllocZeroFunction = nullptr, IN OPTIONAL mAllocator_DestroyAllocator *pDestroyAllocator = nullptr, IN OPTIONAL void *pUserData = nullptr);
mFUNCTION(mAllocator_Destroy, IN_OUT mAllocator *pAllocator);

template <typename T>
mFUNCTION(mAllocator_Allocate, IN OPTIONAL mAllocator *pAllocator, OUT T **pData, const size_t count);

template <typename T>
mFUNCTION(mAllocator_AllocateZero, IN OPTIONAL mAllocator *pAllocator, OUT T **ppData, const size_t count);

template <typename T>
mFUNCTION(mAllocator_Reallocate, IN OPTIONAL mAllocator *pAllocator, OUT T **ppData, const size_t count);

template <typename T>
mFUNCTION(mAllocator_FreePtr, IN OPTIONAL mAllocator *pAllocator, IN_OUT T **ppData);

template <typename T>
mFUNCTION(mAllocator_Free, IN OPTIONAL mAllocator *pAllocator, IN T *pData);

//////////////////////////////////////////////////////////////////////////

template <typename T>
mFUNCTION(mAllocator_Allocate, IN OPTIONAL mAllocator *pAllocator, OUT T **ppData, const size_t count)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppData == nullptr, mR_ArgumentNull);
  mERROR_IF(count == 0, mR_ArgumentOutOfBounds);
  mERROR_IF(pAllocator != nullptr && (!pAllocator->initialized || (pAllocator->pAllocate == nullptr && pAllocator->pAllocateZero == nullptr)), mR_ResourceInvalid);
  mDEFER_CALL_ON_ERROR(ppData, mSetToNullptr);

#ifdef mDEBUG_MEMORY_ALLOCATIONS_PRINT_ALLOCATIONS
  mLOG("Allocating ", sizeof(T) * count, " bytes for ", count, " elements of type ", typeid(T).name(), " (to zero).");
#endif

  if (pAllocator == nullptr)
  {
    if (mDefaultAllocator.pAllocate == nullptr)
      mERROR_CHECK(mDefaultAllocator_Alloc(reinterpret_cast<uint8_t **>(ppData), sizeof(T), count)); // This is only here for the CPU to pre-cache the function.
    else  
      mERROR_CHECK(mDefaultAllocator.pAllocate(reinterpret_cast<uint8_t **>(ppData), sizeof(T), count, mDefaultAllocator.pUserData));
  }
  else if (pAllocator->pAllocate != nullptr)
  {
    mERROR_CHECK(pAllocator->pAllocate(reinterpret_cast<uint8_t **>(ppData), sizeof(T), count, pAllocator->pUserData));
  }
  else
  {
    mERROR_CHECK(pAllocator->pAllocateZero(reinterpret_cast<uint8_t **>(ppData), sizeof(T), count, pAllocator->pUserData));
  }

#ifdef mDEBUG_MEMORY_ALLOCATIONS
  if (mAllocatorDebugging_GetStoreNewAllocations())
  {
    char stacktrace[1024 * 15];
    stacktrace[0] = '\0';
    mDebugSymbolInfo_GetStackTrace(stacktrace, mARRAYSIZE(stacktrace));
    char text[1024 * 16];
    sprintf_s(text, "[#%" PRIu64 "] Type: %s, %" PRIu64 " x %" PRIu64 " Bytes = %" PRIu64 " Bytes.\n@ STACK TRACE:\n%s\n\n", mAllocatorDebugging_GetDebugMemoryAllocationCount()++, typeid(T).name(), count, sizeof(T), count * sizeof(T), stacktrace);

    mAllocatorDebugging_StoreAllocateCall(pAllocator, *ppData, text);
  }
#endif

  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mAllocator_AllocateZero, IN OPTIONAL mAllocator *pAllocator, OUT T **ppData, const size_t count)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppData == nullptr, mR_ArgumentNull);
  mERROR_IF(count == 0, mR_ArgumentOutOfBounds);
  mERROR_IF(pAllocator != nullptr && (!pAllocator->initialized || (pAllocator->pAllocate == nullptr && pAllocator->pAllocateZero == nullptr)), mR_ResourceInvalid);
  mDEFER_CALL_ON_ERROR(ppData, mSetToNullptr);

#ifdef mDEBUG_MEMORY_ALLOCATIONS_PRINT_ALLOCATIONS
  mLOG("Allocating ", sizeof(T) * count, " bytes for ", count, " elements of type ", typeid(T).name(), ".");
#endif

  if (pAllocator == nullptr)
  {
    if (mDefaultAllocator.pAllocateZero == nullptr && mDefaultAllocator.pAllocate == nullptr)
    {
      mERROR_CHECK(mDefaultAllocator_AllocZero(reinterpret_cast<uint8_t **>(ppData), sizeof(T), count)); // This is only here for the CPU to pre-cache the function.
    }
    else if (mDefaultAllocator.pAllocateZero != nullptr)
    {
      mERROR_CHECK(mDefaultAllocator.pAllocateZero(reinterpret_cast<uint8_t **>(ppData), sizeof(T), count, mDefaultAllocator.pUserData));
    }
    else
    {
      mERROR_CHECK(mDefaultAllocator.pAllocate(reinterpret_cast<uint8_t **>(ppData), sizeof(T), count, mDefaultAllocator.pUserData));
      mERROR_CHECK(mMemset(*ppData, count, 0));
    }
  }
  else if (pAllocator->pAllocateZero != nullptr)
  {
    mERROR_CHECK(pAllocator->pAllocateZero(reinterpret_cast<uint8_t **>(ppData), sizeof(T), count, pAllocator->pUserData));
  }
  else
  {
    mERROR_CHECK(pAllocator->pAllocate(reinterpret_cast<uint8_t **>(ppData), sizeof(T), count, pAllocator->pUserData));
    mERROR_CHECK(mMemset(*ppData, count, 0));
  }

#ifdef mDEBUG_MEMORY_ALLOCATIONS
  if (mAllocatorDebugging_GetStoreNewAllocations())
  {
    char stacktrace[1024 * 15];
    stacktrace[0] = '\0';
    mDebugSymbolInfo_GetStackTrace(stacktrace, mARRAYSIZE(stacktrace));
    char text[1024 * 16];
    sprintf_s(text, "[#%" PRIu64 "] Type: %s, %" PRIu64 " x %" PRIu64 " Bytes = %" PRIu64 " Bytes.\n@ STACK TRACE:\n%s\n\n", mAllocatorDebugging_GetDebugMemoryAllocationCount()++, typeid(T).name(), count, sizeof(T), count * sizeof(T), stacktrace);

    mAllocatorDebugging_StoreAllocateCall(pAllocator, *ppData, text);
  }
#endif

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mAllocator_Reallocate, IN OPTIONAL mAllocator *pAllocator, OUT T **ppData, const size_t count)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppData == nullptr, mR_ArgumentNull);
  mERROR_IF(pAllocator != nullptr && (!pAllocator->initialized || pAllocator->pReallocate == nullptr), mR_ResourceInvalid);

#ifdef mDEBUG_MEMORY_ALLOCATIONS
  const size_t originalPointer = reinterpret_cast<size_t>(*ppData);
#endif

#ifdef mDEBUG_MEMORY_ALLOCATIONS_PRINT_ALLOCATIONS
  mLOG("Reallocating ", sizeof(T) * count, " bytes for ", count, " elements of type ", typeid(T).name(), ".");
#endif

  mSTATIC_ASSERT(mIsTriviallyMemoryMovable<T>::value, "This type is not trivially memory movable.");

  if (pAllocator == nullptr)
  {
    if (mDefaultAllocator.pReallocate == nullptr)
      mERROR_CHECK(mDefaultAllocator_Realloc(reinterpret_cast<uint8_t **>(ppData), sizeof(T), count)); // This is only here for the CPU to pre-cache the function.
    else  
      mERROR_CHECK(mDefaultAllocator.pReallocate(reinterpret_cast<uint8_t **>(ppData), sizeof(T), count, mDefaultAllocator.pUserData));
  }
  else
  {
    mERROR_CHECK(pAllocator->pReallocate(reinterpret_cast<uint8_t **>(ppData), sizeof(T), count, pAllocator->pUserData));
  }

#ifdef mDEBUG_MEMORY_ALLOCATIONS
  if (mAllocatorDebugging_GetStoreNewAllocations())
  {
    char stacktrace[1024 * 15];
    stacktrace[0] = '\0';
    mDebugSymbolInfo_GetStackTrace(stacktrace, mARRAYSIZE(stacktrace));
    char text[1024 * 16];
    sprintf_s(text, "[#%" PRIu64 "] Type: %s, %" PRIu64 " x %" PRIu64 " Bytes = %" PRIu64 " Bytes.\n@ STACK TRACE:\n%s\n\n", mAllocatorDebugging_GetDebugMemoryAllocationCount()++, typeid(T).name(), count, sizeof(T), count * sizeof(T), stacktrace);

    mAllocatorDebugging_StoreReallocateCall(pAllocator, originalPointer, *ppData, text);
  }
#endif

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mAllocator_FreePtr, IN OPTIONAL mAllocator *pAllocator, IN_OUT T **ppData)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppData == nullptr, mR_ArgumentNull);

#ifdef mDEBUG_MEMORY_ALLOCATIONS
  const size_t originalPointer = reinterpret_cast<size_t>(*ppData);
#endif

  mDEFER_CALL(ppData, mSetToNullptr);

#ifdef mDEBUG_MEMORY_ALLOCATIONS_PRINT_ALLOCATIONS
  mLOG("Freeing element(s) of type ", typeid(T).name(), ".");
#endif

  if (*ppData == nullptr)
    mRETURN_SUCCESS();

  mERROR_IF(pAllocator != nullptr && (!pAllocator->initialized || pAllocator->pFree == nullptr), mR_ResourceInvalid);

  if (pAllocator == nullptr)
  {
    if (mDefaultAllocator.pFree == nullptr)
      mERROR_CHECK(mDefaultAllocator_Free(reinterpret_cast<uint8_t *>(*ppData))); // This is only here for the CPU to pre-cache the function.
    else
      mERROR_CHECK(mDefaultAllocator.pFree(reinterpret_cast<uint8_t *>(*ppData), mDefaultAllocator.pUserData));
  }
  else
  {
    mERROR_CHECK(pAllocator->pFree(reinterpret_cast<uint8_t *>(*ppData), pAllocator->pUserData));
  }

#ifdef mDEBUG_MEMORY_ALLOCATIONS
  mAllocatorDebugging_StoreFreeCall(pAllocator, originalPointer);
#endif

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mAllocator_Free, IN OPTIONAL mAllocator *pAllocator, IN T *pData)
{
  mFUNCTION_SETUP();

  if (pData == nullptr)
    mRETURN_SUCCESS();

  mERROR_IF(pAllocator != nullptr && !pAllocator->initialized, mR_ResourceInvalid);

#ifdef mDEBUG_MEMORY_ALLOCATIONS
  const size_t originalPointer = reinterpret_cast<size_t>(pData);
#endif

#ifdef mDEBUG_MEMORY_ALLOCATIONS_PRINT_ALLOCATIONS
  mLOG("Freeing element(s) of type ", typeid(T).name(), ".");
#endif

  mERROR_IF(pAllocator != nullptr && (!pAllocator->initialized || pAllocator->pFree == nullptr), mR_ResourceInvalid);

  if (pAllocator == nullptr)
  {
    if (mDefaultAllocator.pFree == nullptr)
      mERROR_CHECK(mDefaultAllocator_Free(reinterpret_cast<uint8_t *>(pData))); // This is only here for the CPU to pre-cache the function.
    else
      mERROR_CHECK(mDefaultAllocator.pFree(reinterpret_cast<uint8_t *>(pData), mDefaultAllocator.pUserData));
  }
  else
  {
    mERROR_CHECK(pAllocator->pFree(reinterpret_cast<uint8_t *>(pData), pAllocator->pUserData));
  }

#ifdef mDEBUG_MEMORY_ALLOCATIONS
  mAllocatorDebugging_StoreFreeCall(pAllocator, originalPointer);
#endif

  mRETURN_SUCCESS();
}

template <typename T>
inline mFUNCTION(mAllocator_AllocateWithSize, IN OPTIONAL mAllocator *pAllocator, OUT T **ppData, const size_t size)
{
  return mAllocator_Allocate(pAllocator, reinterpret_cast<uint8_t **>(ppData), size);
}

template <typename T>
inline mFUNCTION(mAllocator_AllocateZeroWithSize, IN OPTIONAL mAllocator *pAllocator, OUT T **ppData, const size_t size)
{
  return mAllocator_AllocateZero(pAllocator, reinterpret_cast<uint8_t **>(ppData), size);
}

#endif // mAllocator_h__
