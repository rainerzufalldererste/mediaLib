// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "mHashMap.h"
#include "mHash.h"

template <typename TKey, typename TValue>
mFUNCTION(mHashMap_Destroy_Internal, IN mHashMap<TKey, TValue> *pHashMap);

template <typename TKey, typename TValue>
mFUNCTION(mHashMap_Hash_Internal, mPtr<mHashMap<TKey, TValue>> &hashMap, IN TKey *pKey, OUT size_t *pHashedIndex);

//////////////////////////////////////////////////////////////////////////

template <typename TKey, typename TValue>
mFUNCTION(mHashMap_Create, OUT mPtr<mHashMap<TKey, TValue>> *pHashMap, IN mAllocator *pAllocator, const size_t hashMapSize)
{
  mFUNCTION_SETUP();

  mERROR_IF(pHashMap == nullptr, mR_ArgumentNull);
  mERROR_IF(hashMapSize < 1, mR_ArgumentOutOfBounds);

  mERROR_CHECK(mSharedPointer_Allocate(pHashMap, pAllocator, (std::function<void(mHashMap<TKey, TValue> *)>)[](mHashMap<TKey, TValue> *pData) {mHashMap_Destroy_Internal(pData);}, 1));

  (*pHashMap)->pAllocator = pAllocator;
  (*pHashMap)->hashMapSize = hashMapSize;

  mERROR_CHECK(mQueue_Create(&(*pHashMap)->data, pAllocator));
  mERROR_CHECK(mQueue_Reserve((*pHashMap)->data, hashMapSize));

  for (size_t i = 0; i < hashMapSize; ++i)
  {
    mPtr<mChunkedArray<mKeyValuePair<TKey, TValue>>> chunkedArray;
    mDEFER_DESTRUCTION(&chunkedArray, mChunkedArray_Destroy);
    mERROR_CHECK(mChunkedArray_Create(&chunkedArray, pAllocator));

    mERROR_CHECK(mQueue_PushBack((*pHashMap)->data, &chunkedArray));
  }

  mRETURN_SUCCESS();
}

template <typename TKey, typename TValue>
mFUNCTION(mHashMap_Destroy, IN_OUT mPtr<mHashMap<TKey, TValue>> *pHashMap)
{
  mFUNCTION_SETUP();

  mERROR_IF(pHashMap == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Destroy(pHashMap));

  mRETURN_SUCCESS();
}

template <typename TKey, typename TValue>
mFUNCTION(mHashMap_Contains, mPtr<mHashMap<TKey, TValue>> &hashMap, TKey key, OUT bool *pContains, OUT TValue *pValueIfExistent)
{
  mFUNCTION_SETUP();

  mERROR_IF(hashMap == nullptr || pContains == nullptr || pValueIfExistent == nullptr, mR_ArgumentNull);

  size_t index = 0;
  mERROR_CHECK(mHashMap_Hash_Internal(hashMap, &key, &index));

  mPtr<mChunkedArray<mKeyValuePair<TKey, TValue>>> chunkedArray;
  mDEFER_DESTRUCTION(&chunkedArray, mChunkedArray_Destroy);
  mERROR_CHECK(mQueue_PeekAt(hashMap->data, index, &chunkedArray));

  size_t count = 0;
  mERROR_CHECK(mChunkedArray_GetCount(chunkedArray, &count));

  for (size_t i = 0; i < count; ++i)
  {
    mKeyValuePair<TKey, TValue> *pKVPair = nullptr;
    mERROR_CHECK(mChunkedArray_PointerAt(chunkedArray, i, &pKVPair));

    if (pKVPair->key == key)
    {
      *pContains = true;
      *pValueIfExistent = pKVPair->value;
      mRETURN_SUCCESS();
    }
  }

  *pContains = false;

  mRETURN_SUCCESS();
}

template <typename TKey, typename TValue>
mFUNCTION(mHashMap_ContainsGetPointer, mPtr<mHashMap<TKey, TValue>> &hashMap, TKey key, OUT bool *pContains, OUT TValue **ppValueIfExistent)
{
  mFUNCTION_SETUP();

  mERROR_IF(hashMap == nullptr || pContains == nullptr || ppValueIfExistent == nullptr, mR_ArgumentNull);

  size_t index = 0;
  mERROR_CHECK(mHashMap_Hash_Internal(hashMap, &key, &index));

  mPtr<mChunkedArray<mKeyValuePair<TKey, TValue>>> chunkedArray;
  mDEFER_DESTRUCTION(&chunkedArray, mChunkedArray_Destroy);
  mERROR_CHECK(mQueue_PeekAt(hashMap->data, index, &chunkedArray));

  size_t count = 0;
  mERROR_CHECK(mChunkedArray_GetCount(chunkedArray, &count));

  for (size_t i = 0; i < count; ++i)
  {
    mKeyValuePair<TKey, TValue> *pKVPair = nullptr;
    mERROR_CHECK(mChunkedArray_PointerAt(chunkedArray, i, &pKVPair));

    if (pKVPair->key == key)
    {
      *pContains = true;
      *ppValueIfExistent = &pKVPair->value;
      mRETURN_SUCCESS();
    }
  }

  *pContains = false;
  *ppValueIfExistent = nullptr;

  mRETURN_SUCCESS();
}

template <typename TKey, typename TValue>
mFUNCTION(mHashMap_Add, mPtr<mHashMap<TKey, TValue>> &hashMap, TKey key, IN TValue *pValue)
{
  mFUNCTION_SETUP();

  mERROR_IF(hashMap == nullptr || pValue == nullptr, mR_ArgumentNull);

  size_t index = 0;
  mERROR_CHECK(mHashMap_Hash_Internal(hashMap, &key, &index));

  mPtr<mChunkedArray<mKeyValuePair<TKey, TValue>>> chunkedArray;
  mDEFER_DESTRUCTION(&chunkedArray, mChunkedArray_Destroy);
  mERROR_CHECK(mQueue_PeekAt(hashMap->data, index, &chunkedArray));

  mKeyValuePair<TKey, TValue> kvpair;
  mDEFER_DESTRUCTION(&kvpair, mKeyValuePair_Destroy); // does nothing.
  mERROR_CHECK(mKeyValuePair_Create(&kvpair, key, *pValue));

  size_t chunkedArrayIndex = 0; // we don't care anyways.
  mERROR_CHECK(mChunkedArray_Push(chunkedArray, &kvpair, &chunkedArrayIndex));

  mRETURN_SUCCESS();
}

template <typename TKey, typename TValue>
mFUNCTION(mHashMap_Get, mPtr<mHashMap<TKey, TValue>> &hashMap, TKey key, OUT TValue *pValue)
{
  mFUNCTION_SETUP();

  mERROR_IF(hashMap == nullptr || pValue == nullptr, mR_ArgumentNull);

  bool existent = false;
  mERROR_CHECK(mHashMap_Contains(hashMap, key, &existent, pValue));

  mERROR_IF(!existent, mR_ResourceNotFound);

  mRETURN_SUCCESS();
}

template <typename TKey, typename TValue>
mFUNCTION(mHashMap_Remove, mPtr<mHashMap<TKey, TValue>> &hashMap, TKey key, OUT TValue *pValue)
{
  mFUNCTION_SETUP();

  mERROR_IF(hashMap == nullptr || pValue == nullptr, mR_ArgumentNull);

  size_t index = 0;
  mERROR_CHECK(mHashMap_Hash_Internal(hashMap, &key, &index));

  mPtr<mChunkedArray<mKeyValuePair<TKey, TValue>>> chunkedArray;
  mDEFER_DESTRUCTION(&chunkedArray, mChunkedArray_Destroy);
  mERROR_CHECK(mQueue_PeekAt(hashMap->data, index, &chunkedArray));

  size_t count = 0;
  mERROR_CHECK(mChunkedArray_GetCount(chunkedArray, &count));

  for (size_t i = 0; i < count; ++i)
  {
    mKeyValuePair<TKey, TValue> *pKVPair = nullptr;
    mERROR_CHECK(mChunkedArray_PointerAt(chunkedArray, i, &pKVPair));

    if (pKVPair->key == key)
    {
      mERROR_CHECK(mChunkedArray_PopAt(chunkedArray, i, pKVPair));
      *pValue = std::move(pKVPair->value);
      mRETURN_SUCCESS();
    }
  }

  mRETURN_RESULT(mR_ResourceNotFound);
}

template <typename TKey, typename TValue>
mFUNCTION(mHashMap_SetKeyValuePairDestructionFunction, mPtr<mHashMap<TKey, TValue>> &hashMap, const std::function<void(mKeyValuePair<TKey, TValue> *)> &destructionFunction)
{
  mFUNCTION_SETUP();

  mERROR_IF(hashMap == nullptr, mR_ArgumentNull);

  for (size_t i = 0; i < hashMapSize; ++i)
  {
    mPtr<mChunkedArray<mKeyValuePair<TKey, TValue>>> chunkedArray;
    mDEFER_DESTRUCTION(chunkedArray, mChunkedArray_Destroy);
    mERROR_CHECK(mQueue_PeekAt(hashMap->data, i, &chunkedArray));
    mERROR_CHECK(mChunkedArray_SetDestructionFunction(chunkedArray, destructionFunction));
  }

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

template<typename TKey, typename TValue>
inline mFUNCTION(mHashMap_Destroy_Internal, IN mHashMap<TKey, TValue> *pHashMap)
{
  mFUNCTION_SETUP();

  mERROR_IF(pHashMap == nullptr, mR_ArgumentNull);

  for (size_t i = 0; i < pHashMap->hashMapSize; ++i)
  {
    mPtr<mChunkedArray<mKeyValuePair<TKey, TValue>>> chunkedArray;
    mERROR_CHECK(mQueue_PopFront(pHashMap->data, &chunkedArray));
    mERROR_CHECK(mChunkedArray_Destroy(&chunkedArray));
  }

  mERROR_CHECK(mQueue_Destroy(&pHashMap->data));

  mRETURN_SUCCESS();
}

template<typename TKey, typename TValue>
inline mFUNCTION(mHashMap_Hash_Internal, mPtr<mHashMap<TKey, TValue>> &hashMap, IN TKey *pKey, OUT size_t *pHashedIndex)
{
  mFUNCTION_SETUP();

  mERROR_IF(pKey == nullptr || pHashedIndex == nullptr, mR_ArgumentNull);

  uint64_t hash = 0;

  mERROR_CHECK(mHash(pKey, &hash));

  *pHashedIndex = (size_t)hash % hashMap->hashMapSize;

  mRETURN_SUCCESS();
}
