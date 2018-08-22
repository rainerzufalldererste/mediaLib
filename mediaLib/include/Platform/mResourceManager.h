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
#include "mMutex.h"

template <typename TValue, typename TKey>
mCreateResource(OUT TValue *pResource, TKey param);

template <typename TValue>
mDestroyResource(IN_OUT TValue *pResource);

template <typename TValue>
mResourceGetPreferredAllocator(OUT mAllocator **ppAllocator);

//////////////////////////////////////////////////////////////////////////

template <typename TValue, typename TKey>
struct mResourceManager
{
  static mPtr<mResourceManager> Instance;

  struct ResourceData
  {
    TValue resource;
    mSharedPointer<TValue>::PointerParams pointerParams;
    mPtr<TValue> sharedPointer;
  };

  mPtr<mHashMap<TKey, ResourceData>> data;
  mAllocator *pAllocator;
  mMutex *pMutex;
};

template <typename TKey, typename TValue>
mFUNCTION(mResourceManager_GetResource, OUT mPtr<TValue> *pValue, TKey key);

template <typename TKey, typename TValue>
mFUNCTION(mResourceManager_CreateExplicit, IN mAllocator *pAllocator);

//////////////////////////////////////////////////////////////////////////

// To be available for the resource manager, expose two functions: 
//  'mCreateResource(OUT TValue* pResource, TKey param);'
//  and
//  'mDestroyResource(IN_OUT TValue *pResource);'

template <typename TValue, typename TKey>
mCreateResource(OUT TValue *pResource, TKey param)
{
  static_assert(false, "This type does not support resource creation yet.");
}

// To be available for the resource manager, expose two functions: 
//  'mCreateResource(OUT TValue* pResource, TKey param);'
//  and
//  'mDestroyResource(IN_OUT TValue *pResource);'

template <typename TValue>
mDestroyResource(IN_OUT TValue *pResource)
{
  static_assert(false, "This type does not support resource destruction yet.");
}

template <typename TValue>
mResourceGetPreferredAllocator(OUT mAllocator **ppAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppAllocator == nullptr, mR_ArgumentNull);

  *ppAllocator = nullptr;

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

template<typename TKey, typename TValue>
inline mFUNCTION(mResourceManager_GetResource, OUT mPtr<TValue> *pValue, TKey key)
{
  mFUNCTION_SETUP();

  mERROR_IF(pValue == nullptr, mR_ArgumentNull);

  if (mResourceManager<TKey, TValue>::Instance == nullptr)
  {
    mAllocator *pAllocator;
    mERROR_CHECK(mResourceGetPreferredAllocator<TValue>(&pAllocator));

    mERROR_CHECK(mResourceManager_CreateExplicit<TKey, TValue>(pAllocator));
  }

  mERROR_CHECK(mMutex_Lock(mResourceManager<TKey, TValue>::Instance->pMutex));
  mDEFER_DESTRUCTION(mResourceManager<TKey, TValue>::Instance->pMutex, mMutex_Unlock);

  bool contains = false;
  mResourceManager<TKey, TValue>::ResourceData *pResourceData;
  mERROR_CHECK(mHashMap_ContainsGetPointer(mResourceManager<TKey, TValue>::Instance->data, key, &contains, &pResourceData));

  if (contains)
  {
    *pValue = pResourceData->sharedPointer;
    mRETURN_SUCCESS();
  }

  mResourceManager<TKey, TValue>::ResourceData resourceData;
  mERROR_CHECK(mMemset(&resourceData, 1));
  mERROR_CHECK(mHashMap_Add(mResourceManager<TKey, TValue>::Instance->data, key, &resourceData));

  mERROR_CHECK(mHashMap_ContainsGetPointer(mResourceManager<TKey, TValue>::Instance->data, key, &contains, &pResourceData));
  mERROR_IF(!contains, mR_InternalError);

  mERROR_CHECK(mSharedPointer_CreateInplace(
    &pResourceData->sharedPointer,
    &pResourceData->pointerParams, 
    &pResourceData->resource,
    mSHARED_POINTER_FOREIGN_RESOURCE, 
    [=](TValue *pData) 
      { 
        mDestroyResource(pData); 
        
        mResourceManager<TKey, TValue>::ResourceData resourceData0; // we don't care.
        mHashMap_Remove(mResourceManager<TKey, TValue>::Instance->data, key, &resourceData0);
      }));

  mERROR_CHECK(mCreateResource(&pResourceData->resource, key));
  *pValue = pResourceData->sharedPointer;

  // The resource manager does not own the resource!
  --pResourceData->pointerParams.referenceCount;

  mRETURN_SUCCESS();
}

template<typename TKey, typename TValue>
inline mFUNCTION(mResourceManager_CreateExplicit, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  // only the hashmap will be allocated by pAllocator!

  if (mResourceManager<TKey, TValue>::Instance != nullptr)
    mRETURN_RESULT(mR_ResourceStateInvalid);

  mERROR_CHECK(mSharedPointer_Allocate(
    &mResourceManager<TKey, TValue>::Instance, 
    nullptr, 
    (std::function<void(mResourceManager<TKey, TValue> *)>)[](mResourceManager<TKey, TValue> *pData) 
      { 
        mHashMap_Destroy(&pData->data);
        mMutex_Destroy(&pData->pMutex); 
      }, 
    1));

  mERROR_CHECK(mMutex_Create(&mResourceManager<TKey, TValue>::Instance->pMutex, nullptr));
  mERROR_CHECK(mHashMap_Create(&mResourceManager<TKey, TValue>::Instance->pMutex, pAllocator, 64));
  mResourceManager<TKey, TValue>::Instance->pAllocator = pAllocator;

  mRETURN_SUCCESS();
}

#endif // mResourceManager_h__
