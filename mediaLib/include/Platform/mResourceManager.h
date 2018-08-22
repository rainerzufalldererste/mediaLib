// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef mResourceManager_h__
#define mResourceManager_h__

#include "default.h"
#include "mHashMap.h"
#include "mPool.h"
#include "mMutex.h"

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
  static_assert(false, "This type does not support resource creation yet.");
}

// To be available for the resource manager, expose two functions: 
//  'mResult mCreateResource(OUT TValue* pResource, TKey param);'
//  and
//  'mResult mDestroyResource(IN_OUT TValue *pResource);'

template <typename TValue>
mFUNCTION(mDestroyResource, IN_OUT TValue *pResource)
{
  static_assert(false, "This type does not support resource destruction yet.");
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
    mERROR_CHECK((mResourceManager_CreateRessourceManager_Explicit<TKey, TValue>(pAllocator)));
  }

  mERROR_CHECK(mMutex_Lock(CurrentResourceManagerType::Instance()->pMutex));
  mDEFER_DESTRUCTION(CurrentResourceManagerType::Instance()->pMutex, mMutex_Unlock);

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
  mERROR_CHECK(mPool_Add(CurrentResourceManagerType::Instance()->data, &resourceData, &resourceIndex));
  mERROR_CHECK(mHashMap_Add(CurrentResourceManagerType::Instance()->keys, key, &resourceIndex));

  CurrentResourceManagerType::ResourceData *pRetrievedResourceData = nullptr;
  mERROR_CHECK(mPool_PointerAt(CurrentResourceManagerType::Instance()->data, resourceIndex, &pRetrievedResourceData));

  mERROR_CHECK(mSharedPointer_CreateInplace(
    &pRetrievedResourceData->sharedPointer,
    &pRetrievedResourceData->pointerParams,
    &pRetrievedResourceData->resource,
    mSHARED_POINTER_FOREIGN_RESOURCE, 
    (std::function<void (TValue *)>)[=](TValue *pData)
      {
        //mPRINT("Destroying Resource %" PRIu64 " of type %s.\n", resourceIndex, typeid(TValue *).name());
        mDestroyResource(pData);
        
        size_t resourceIndex0 = 0;
        mResult result = mHashMap_Remove(CurrentResourceManagerType::Instance()->keys, key, &resourceIndex0);

        mASSERT_DEBUG(resourceIndex == resourceIndex0, "Corrupted ResourceManager data.");

        CurrentResourceManagerType::ResourceData resourceData0; // we don't care.
        result = mPool_RemoveAt(CurrentResourceManagerType::Instance()->data, resourceIndex, &resourceData0);
      }));

  mERROR_CHECK(mCreateResource(&pRetrievedResourceData->resource, key));
  *pValue = pRetrievedResourceData->sharedPointer;

  // The resource manager does not own the resource!
  --pRetrievedResourceData->pointerParams.referenceCount;

  mRETURN_SUCCESS();
}

template<typename TKey, typename TValue>
inline mFUNCTION(mResourceManager_CreateRessourceManager_Explicit, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  typedef mResourceManager<TValue, TKey> CurrentResourceManagerType;

  // only the hashmap & the pool will be allocated by pAllocator!

  if (CurrentResourceManagerType::Instance() != nullptr)
    mRETURN_RESULT(mR_ResourceStateInvalid);

  mERROR_CHECK(mSharedPointer_Allocate(
    &CurrentResourceManagerType::Instance(), 
    nullptr, 
    (std::function<void(CurrentResourceManagerType *)>)[](CurrentResourceManagerType *pData) 
      {
        mHashMap_Destroy(&pData->keys);
        mPool_Destroy(&pData->data);
        mMutex_Destroy(&pData->pMutex); 
      }, 
    1));

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

#endif // mResourceManager_h__
