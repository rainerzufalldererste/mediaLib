#ifndef mAllocator_h__
#define mAllocator_h__

#include "mResult.h"
#include "mediaLib.h"

//#define mDEBUG_MEMORY_ALLOCATIONS

#ifdef mDEBUG_MEMORY_ALLOCATIONS
//#define mDEBUG_MEMORY_ALLOCATIONS_PRINT_ALLOCATIONS
#include <map>
#include <mutex>

struct mAllocator;

extern uint64_t mAllocatorDebugging_DebugMemoryAllocationCount;

extern std::recursive_mutex mAllocatorDebugging_DebugMemoryAllocationMutex;
extern std::map<mAllocator *, std::map<size_t, std::string>> mAllocatorDebugging_DebugMemoryAllocationMap;

void mAllocatorDebugging_PrintRemainingMemoryAllocations(mAllocator *pAllocator);
void mAllocatorDebugging_PrintAllRemainingMemoryAllocations();
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

struct mAllocator
{
  mAllocator_AllocFunction *pAllocate = nullptr;
  mAllocator_AllocFunction *pAllocateZero = nullptr;
  mAllocator_AllocFunction *pReallocate = nullptr;
  mAllocator_FreeFunction *pFree = nullptr;
  mAllocator_MoveFunction *pMove = nullptr;
  mAllocator_CopyFunction *pCopy = nullptr;
  mAllocator_DestroyAllocator *pDestroyAllocator = nullptr;

  void *pUserData = nullptr;
  bool initialized = false;
};

mAllocator mAllocator_StaticCreate(IN mAllocator_AllocFunction *pAllocFunction, IN mAllocator_AllocFunction *pReallocFunction, IN mAllocator_FreeFunction *pFree, IN OPTIONAL mAllocator_MoveFunction *pMove = nullptr, IN OPTIONAL mAllocator_CopyFunction *pCopy = nullptr, IN OPTIONAL mAllocator_AllocFunction *pAllocZeroFunction = nullptr, IN OPTIONAL mAllocator_DestroyAllocator *pDestroyAllocator = nullptr, IN OPTIONAL void *pUserData = nullptr);

mFUNCTION(mAllocator_Create, OUT mAllocator *pAllocator, IN mAllocator_AllocFunction *pAllocFunction, IN mAllocator_AllocFunction *pReallocFunction, IN mAllocator_FreeFunction *pFree, IN OPTIONAL mAllocator_MoveFunction *pMove = nullptr, IN OPTIONAL mAllocator_CopyFunction *pCopy = nullptr, IN OPTIONAL mAllocator_AllocFunction *pAllocZeroFunction = nullptr, IN OPTIONAL mAllocator_DestroyAllocator *pDestroyAllocator = nullptr, IN OPTIONAL void *pUserData = nullptr);
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

template <typename T>
mFUNCTION(mAllocator_Move, IN OPTIONAL mAllocator *pAllocator, IN_OUT T *pDestination, IN_OUT T *pSource, const size_t count);

template <typename T>
mFUNCTION(mAllocator_Copy, IN OPTIONAL mAllocator *pAllocator, IN_OUT T *pDestination, IN const T *pSource, const size_t count);

//////////////////////////////////////////////////////////////////////////

template <typename T>
mFUNCTION(mAllocator_Allocate, IN OPTIONAL mAllocator *pAllocator, OUT T **ppData, const size_t count)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppData == nullptr, mR_ArgumentNull);
  mDEFER_CALL_ON_ERROR(ppData, mSetToNullptr);

#ifdef mDEBUG_MEMORY_ALLOCATIONS_PRINT_ALLOCATIONS
  mLOG("Allocating %" PRIu64 " bytes for %" PRIu64 " elements of type %s (to zero).\n", sizeof(T) * count, count, typeid(T).name());
#endif

  if (pAllocator == nullptr || !pAllocator->initialized || (pAllocator->pAllocate == nullptr && pAllocator->pAllocateZero == nullptr))
    mERROR_CHECK(mDefaultAllocator_Alloc((uint8_t **)ppData, sizeof(T), count));
  else if (pAllocator->pAllocate != nullptr)
    mERROR_CHECK(pAllocator->pAllocate((uint8_t **)ppData, sizeof(T), count, pAllocator->pUserData));
  else
    mERROR_CHECK(pAllocator->pAllocateZero((uint8_t **)ppData, sizeof(T), count, pAllocator->pUserData));

#ifdef mDEBUG_MEMORY_ALLOCATIONS
  char text[1024];
  sprintf_s(text, "[%" PRIu64 "] Type: %s, Count: %" PRIu64 ", Size: %" PRIu64 ".", mAllocatorDebugging_DebugMemoryAllocationCount++, typeid(T).name(), count, count * sizeof(T));

  auto entry = mAllocatorDebugging_DebugMemoryAllocationMap.find(pAllocator);

  if (entry == mAllocatorDebugging_DebugMemoryAllocationMap.end())
  {
    mAllocatorDebugging_DebugMemoryAllocationMap.insert(std::pair<mAllocator *, std::map<size_t, std::string>>(pAllocator, std::map<size_t, std::string>()));
    entry = mAllocatorDebugging_DebugMemoryAllocationMap.find(pAllocator);
  }

  if (entry->second.find((size_t)*ppData) == entry->second.end())
    entry->second.insert(std::pair<size_t, std::string>((size_t)*ppData, std::string(text)));
#endif

  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mAllocator_AllocateZero, IN OPTIONAL mAllocator *pAllocator, OUT T **ppData, const size_t count)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppData == nullptr, mR_ArgumentNull);
  mDEFER_CALL_ON_ERROR(ppData, mSetToNullptr);

#ifdef mDEBUG_MEMORY_ALLOCATIONS_PRINT_ALLOCATIONS
  mLOG("Allocating %" PRIu64 " bytes for %" PRIu64 " elements of type %s.\n", sizeof(T) * count, count, typeid(T).name());
#endif

  if (pAllocator == nullptr || !pAllocator->initialized || (pAllocator->pAllocate == nullptr && pAllocator->pAllocateZero == nullptr))
  {
    mERROR_CHECK(mDefaultAllocator_AllocZero((uint8_t **)ppData, sizeof(T), count));
  }
  else if (pAllocator->pAllocateZero != nullptr)
  {
    mERROR_CHECK(pAllocator->pAllocateZero((uint8_t **)ppData, sizeof(T), count, pAllocator->pUserData));
  }
  else
  {
    mERROR_CHECK(pAllocator->pAllocate((uint8_t **)ppData, sizeof(T), count, pAllocator->pUserData));
    mERROR_CHECK(mMemset(*ppData, count, 0));
  }

#ifdef mDEBUG_MEMORY_ALLOCATIONS
  char text[1024];
  sprintf_s(text, "[%" PRIu64 "] Type: %s, Count: %" PRIu64 ", Size: %" PRIu64 ".", mAllocatorDebugging_DebugMemoryAllocationCount++, typeid(T).name(), count, count * sizeof(T));

  auto entry = mAllocatorDebugging_DebugMemoryAllocationMap.find(pAllocator);

  if (entry == mAllocatorDebugging_DebugMemoryAllocationMap.end())
  {
    mAllocatorDebugging_DebugMemoryAllocationMap.insert(std::pair<mAllocator *, std::map<size_t, std::string>>(pAllocator, std::map<size_t, std::string>()));
    entry = mAllocatorDebugging_DebugMemoryAllocationMap.find(pAllocator);
  }

  if (entry->second.find((size_t)*ppData) == entry->second.end())
    entry->second.insert(std::pair<size_t, std::string>((size_t)*ppData, std::string(text)));
#endif

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mAllocator_Reallocate, IN OPTIONAL mAllocator *pAllocator, OUT T **ppData, const size_t count)
{
  mFUNCTION_SETUP();

#ifdef mDEBUG_MEMORY_ALLOCATIONS
  size_t originalPointer = (size_t)*ppData;
#endif

#ifdef mDEBUG_MEMORY_ALLOCATIONS_PRINT_ALLOCATIONS
  mLOG("Reallocating %" PRIu64 " bytes for %" PRIu64 " elements of type %s.\n", sizeof(T) * count, count, typeid(T).name());
#endif

  mSTATIC_ASSERT(mIsTriviallyMemoryMovable<T>::value, "This type is not trivially memory movable.");

  if (pAllocator == nullptr || !pAllocator->initialized || pAllocator->pReallocate == nullptr)
    mERROR_CHECK(mDefaultAllocator_Realloc((uint8_t **)ppData, sizeof(T), count));
  else
    mERROR_CHECK(pAllocator->pReallocate((uint8_t **)ppData, sizeof(T), count, pAllocator->pUserData));

#ifdef mDEBUG_MEMORY_ALLOCATIONS
  char text[1024];
  sprintf_s(text, "[%" PRIu64 "] Type: %s, Count: %" PRIu64 ", Size: %" PRIu64 ".", mAllocatorDebugging_DebugMemoryAllocationCount++, typeid(T).name(), count, count * sizeof(T));

  auto entry = mAllocatorDebugging_DebugMemoryAllocationMap.find(pAllocator);

  if (entry == mAllocatorDebugging_DebugMemoryAllocationMap.end())
  {
    mAllocatorDebugging_DebugMemoryAllocationMap.insert(std::pair<mAllocator *, std::map<size_t, std::string>>(pAllocator, std::map<size_t, std::string>()));
    entry = mAllocatorDebugging_DebugMemoryAllocationMap.find(pAllocator);
  }

  entry->second.erase(originalPointer);

  if (entry->second.find((size_t)*ppData) == entry->second.end())
    entry->second.insert(std::pair<size_t, std::string>((size_t)*ppData, std::string(text)));
#endif

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mAllocator_FreePtr, IN OPTIONAL mAllocator *pAllocator, IN_OUT T **ppData)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppData == nullptr, mR_ArgumentNull);

#ifdef mDEBUG_MEMORY_ALLOCATIONS
  size_t originalPointer = (size_t)*ppData;
#endif

  mDEFER_CALL(ppData, mSetToNullptr);

#ifdef mDEBUG_MEMORY_ALLOCATIONS_PRINT_ALLOCATIONS
  mLOG("Freeing element(s) of type %s.\n", typeid(T).name());
#endif

  if (*ppData == nullptr)
    mRETURN_SUCCESS();

  if (pAllocator == nullptr || !pAllocator->initialized || pAllocator->pFree == nullptr)
    mERROR_CHECK(mDefaultAllocator_Free((uint8_t *)*ppData));
  else
    mERROR_CHECK(pAllocator->pFree((uint8_t *)*ppData, pAllocator->pUserData));

#ifdef mDEBUG_MEMORY_ALLOCATIONS
  auto entry = mAllocatorDebugging_DebugMemoryAllocationMap.find(pAllocator);

  if (entry == mAllocatorDebugging_DebugMemoryAllocationMap.end())
  {
    mAllocatorDebugging_DebugMemoryAllocationMap.insert(std::pair<mAllocator *, std::map<size_t, std::string>>(pAllocator, std::map<size_t, std::string>()));
    entry = mAllocatorDebugging_DebugMemoryAllocationMap.find(pAllocator);
  }

  entry->second.erase(originalPointer);
#endif

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mAllocator_Free, IN OPTIONAL mAllocator *pAllocator, IN T *pData)
{
  mFUNCTION_SETUP();

  if (pData == nullptr)
    mRETURN_SUCCESS();

#ifdef mDEBUG_MEMORY_ALLOCATIONS
  size_t originalPointer = (size_t)*ppData;
#endif

#ifdef mDEBUG_MEMORY_ALLOCATIONS_PRINT_ALLOCATIONS
  mLOG("Freeing element(s) of type %s.\n", typeid(T).name());
#endif

  if (pAllocator == nullptr || !pAllocator->initialized || pAllocator->pFree == nullptr)
    mERROR_CHECK(mDefaultAllocator_Free((uint8_t *)pData));
  else
    mERROR_CHECK(pAllocator->pFree((uint8_t *)pData, pAllocator->pUserData));

#ifdef mDEBUG_MEMORY_ALLOCATIONS
  auto entry = mAllocatorDebugging_DebugMemoryAllocationMap.find(pAllocator);

  if (entry == mAllocatorDebugging_DebugMemoryAllocationMap.end())
  {
    mAllocatorDebugging_DebugMemoryAllocationMap.insert(std::pair<mAllocator *, std::map<size_t, std::string>>(pAllocator, std::map<size_t, std::string>()));
    entry = mAllocatorDebugging_DebugMemoryAllocationMap.find(pAllocator);
  }

  entry->second.erase(originalPointer);
#endif

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mAllocator_Move, IN OPTIONAL mAllocator *pAllocator, IN_OUT T *pDestination, IN_OUT T *pSource, const size_t count)
{
  mFUNCTION_SETUP();

  mERROR_IF(pDestination == nullptr || pSource == nullptr, mR_ArgumentNull);

  if (mIsTriviallyMemoryMovable<T>::value)
  {
    if (pAllocator == nullptr || !pAllocator->initialized || pAllocator->pMove == nullptr)
      mERROR_CHECK(mDefaultAllocator_Move((uint8_t *)pDestination, (uint8_t *)pSource, sizeof(T), count));
    else
      mERROR_CHECK(pAllocator->pMove((uint8_t *)pDestination, (uint8_t *)pSource, sizeof(T), count, pAllocator->pUserData));
  }
  else
  {
    mMoveConstructMultiple(pDestination, pSource, count);
  }

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mAllocator_Copy, IN OPTIONAL mAllocator *pAllocator, IN_OUT T *pDestination, IN const T *pSource, const size_t count)
{
  mFUNCTION_SETUP();

  mERROR_IF(pDestination == nullptr || pSource == nullptr, mR_ArgumentNull);

  if (pAllocator == nullptr || !pAllocator->initialized || pAllocator->pCopy == nullptr)
    mERROR_CHECK(mDefaultAllocator_Copy((uint8_t *)pDestination, (uint8_t *)pSource, sizeof(T), count));
  else
    mERROR_CHECK(pAllocator->pCopy((uint8_t *)pDestination, (uint8_t *)pSource, sizeof(T), count, pAllocator->pUserData));

  mRETURN_SUCCESS();
}

#endif // mAllocator_h__
