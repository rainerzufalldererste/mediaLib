#ifndef mResourceManager_h__
#define mResourceManager_h__

#include "mediaLib.h"
#include "mHashMap.h"
#include "mPool.h"
#include "mMutex.h"

//#define mPRINT_RESOURCE_MANAGER_LOG

template <typename TValue, typename TKey>
mFUNCTION(mCreateResource, OUT TValue *pResource, TKey param);

template <typename TValue>
mFUNCTION(mDestroyResource, IN_OUT TValue *pResource);

template <typename TValue>
mFUNCTION(mResourceGetPreferredAllocator, OUT mAllocator **ppAllocator);

//////////////////////////////////////////////////////////////////////////

template <typename TValue, typename TKey>
struct mResourceManager
{
  static mPtr<mResourceManager<TValue, TKey>> &Instance();

  struct ResourceData
  {
    TValue resource;
    typename mSharedPointer<TValue>::PointerParams pointerParams;
    mPtr<TValue> sharedPointer;

    ResourceData() {}

    ResourceData(ResourceData &&move) :
      pointerParams(std::move(move.pointerParams)),
      sharedPointer(std::move(move.sharedPointer)),
      resource(std::move(move.resource))
    {
      if(sharedPointer != nullptr && sharedPointer.m_pParams != nullptr)
        sharedPointer.m_pParams = &pointerParams;

      move.~ResourceData();
    }

    ResourceData &operator =(ResourceData &&move)
    {
      pointerParams = std::move(move.pointerParams);
      sharedPointer = std::move(move.sharedPointer);
      resource = std::move(move.resource);

      if (sharedPointer != nullptr && sharedPointer.m_pParams != nullptr)
        sharedPointer.m_pParams = &pointerParams;

      move.~ResourceData();

      return *this;
    }
  };

  mPtr<mHashMap<TKey, size_t>> keys;
  mPtr<mPool<ResourceData>> data;
  mAllocator *pAllocator;
  mMutex *pMutex;
};

//////////////////////////////////////////////////////////////////////////

// To be available for the resource manager, expose two functions: 
//  'mResult mCreateResource(OUT TValue* pResource, TKey param);'
//  and
//  'mResult mDestroyResource(IN_OUT TValue *pResource);'

template <typename TValue, typename TKey>
mFUNCTION(mCreateResource, OUT TValue *pResource, TKey param)
{
  mSTATIC_ASSERT(false, "This type does not support resource creation yet.");
}

// To be available for the resource manager, expose two functions: 
//  'mResult mCreateResource(OUT TValue* pResource, TKey param);'
//  and
//  'mResult mDestroyResource(IN_OUT TValue *pResource);'

template <typename TValue>
mFUNCTION(mDestroyResource, IN_OUT TValue *pResource)
{
  mSTATIC_ASSERT(false, "This type does not support resource destruction yet.");
}

template <typename TValue>
mFUNCTION(mResourceGetPreferredAllocator, OUT mAllocator **ppAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppAllocator == nullptr, mR_ArgumentNull);

  *ppAllocator = nullptr;

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

template<typename TValue, typename TKey>
inline mFUNCTION(mResourceManager_GetResource, OUT mPtr<TValue> *pValue, TKey key)
{
  mFUNCTION_SETUP();

  typedef mResourceManager<TValue, TKey> CurrentResourceManagerType;

  mERROR_IF(pValue == nullptr, mR_ArgumentNull);

  if (CurrentResourceManagerType::Instance() == nullptr)
  {
    mAllocator *pAllocator;
    mERROR_CHECK((mResourceGetPreferredAllocator<TValue>)(&pAllocator));
    mERROR_CHECK((mResourceManager_CreateResourceManager_Explicit<TKey, TValue>(pAllocator)));
  }

  mERROR_CHECK(mMutex_Lock(CurrentResourceManagerType::Instance()->pMutex));
  mDEFER_CALL(CurrentResourceManagerType::Instance()->pMutex, mMutex_Unlock);

  bool contains = false;
  size_t resourceIndex = 0;
  mERROR_CHECK(mHashMap_Contains(CurrentResourceManagerType::Instance()->keys, key, &contains, &resourceIndex));

  if (contains)
  {
    CurrentResourceManagerType::ResourceData *pResourceData = nullptr;
    mERROR_CHECK(mPool_PointerAt(CurrentResourceManagerType::Instance()->data, resourceIndex, &pResourceData));
    *pValue = pResourceData->sharedPointer;
    mRETURN_SUCCESS();
  }

  resourceIndex = 0;
  CurrentResourceManagerType::ResourceData resourceData;
  mERROR_CHECK(mMemset(&resourceData, 1));
  mERROR_CHECK(mPool_Add(CurrentResourceManagerType::Instance()->data, std::forward<CurrentResourceManagerType::ResourceData>(resourceData), &resourceIndex));
  mERROR_CHECK(mHashMap_Add(CurrentResourceManagerType::Instance()->keys, key, &resourceIndex));

#if defined(mPRINT_RESOURCE_MANAGER_LOG)
  mPRINT("Creating Resource %" PRIu64 " of type %s at index %" PRIu64 ".\n", resourceIndex, typeid(TValue *).name(), resourceIndex);
#endif

  CurrentResourceManagerType::ResourceData *pRetrievedResourceData = nullptr;
  mERROR_CHECK(mPool_PointerAt(CurrentResourceManagerType::Instance()->data, resourceIndex, &pRetrievedResourceData));

  std::function<void(TValue *)> cleanupFunc = [resourceIndex, key](TValue * /*pData*/)
  {
    size_t resourceIndex0 = 0;
    mResult result = mHashMap_Remove(CurrentResourceManagerType::Instance()->keys, key, &resourceIndex0);

    if (mFAILED(result))
      return;

    mASSERT(resourceIndex == resourceIndex0, "Corrupted ResourceManager data.");

    CurrentResourceManagerType::ResourceData resourceData0;
    result = mPool_RemoveAt(CurrentResourceManagerType::Instance()->data, resourceIndex, &resourceData0);

    if (mFAILED(result))
      return;

    resourceData0.sharedPointer.m_pData = nullptr;
    resourceData0.sharedPointer.m_pParams = nullptr;

#if defined(mPRINT_RESOURCE_MANAGER_LOG)
    mPRINT("Destroying Resource %" PRIu64 " of type %s at index %" PRIu64 ".\n", resourceIndex, typeid(TValue *).name(), resourceIndex);
#endif

    mDestruct(&resourceData0);
  };

  mERROR_CHECK(mSharedPointer_CreateInplace(
    &pRetrievedResourceData->sharedPointer,
    &pRetrievedResourceData->pointerParams,
    &pRetrievedResourceData->resource,
    mSHARED_POINTER_FOREIGN_RESOURCE, 
    cleanupFunc));

  mERROR_CHECK(mCreateResource(&pRetrievedResourceData->resource, key));
  *pValue = pRetrievedResourceData->sharedPointer;

  // The resource manager does not own the resource!
  --pRetrievedResourceData->pointerParams.referenceCount;

  mRETURN_SUCCESS();
}

template<typename TKey, typename TValue>
inline mFUNCTION(mResourceManager_CreateResourceManager_Explicit, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  typedef mResourceManager<TValue, TKey> CurrentResourceManagerType;

  // only the hashmap & the pool will be allocated by pAllocator!

  if (CurrentResourceManagerType::Instance() != nullptr)
    mRETURN_RESULT(mR_ResourceStateInvalid);

  std::function<void(CurrentResourceManagerType *)> destructionFunc = 
    [](CurrentResourceManagerType *pData)
  {
    mPool_Destroy(&pData->data);
    mHashMap_Destroy(&pData->keys);
    mMutex_Destroy(&pData->pMutex);
  };

  mERROR_CHECK(mSharedPointer_Allocate(&CurrentResourceManagerType::Instance(), nullptr, destructionFunc, 1));

  mERROR_CHECK(mMutex_Create(&CurrentResourceManagerType::Instance()->pMutex, nullptr));
  mERROR_CHECK(mHashMap_Create(&CurrentResourceManagerType::Instance()->keys, pAllocator, 64));
  mERROR_CHECK(mPool_Create(&CurrentResourceManagerType::Instance()->data, pAllocator));
  CurrentResourceManagerType::Instance()->pAllocator = pAllocator;

  mRETURN_SUCCESS();
}

template<typename TValue, typename TKey>
inline mPtr<mResourceManager<TValue, TKey>>& mResourceManager<TValue, TKey>::Instance()
{
  static mPtr<mResourceManager<TValue, TKey>> singletonInstance;
  return singletonInstance;
}

template<typename TValue, typename TKey>
mFUNCTION(mDestruct, struct mResourceManager<TValue, TKey>::ResourceData *pResourceData)
{
  mFUNCTION_SETUP();

  mERROR_IF(pResourceData == nullptr, mR_Success);

  mSharedPointer_Destroy(&pResourceData->sharedPointer);
  pResourceData->pointerParams.cleanupFunction.~function();
  mDestroyResource(&pResourceData->resource);
  mERROR_CHECK(mMemset(pResourceData, 1));

  mRETURN_SUCCESS();
}

#endif // mResourceManager_h__
