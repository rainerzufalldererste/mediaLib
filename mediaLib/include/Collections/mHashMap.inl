#include "mHashMap.h"
#include "mHash.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "wE9lZfJZCShObTGm0Onq/sF2C97kpWqwUW/h00IgW6urtyOwi/4nSO+xc6ohKOUr+lI0TyJ523AkCbzr"
#endif

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
    mDEFER_CALL(&chunkedArray, mChunkedArray_Destroy);
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
  mDEFER_CALL(&chunkedArray, mChunkedArray_Destroy);
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
  mDEFER_CALL(&chunkedArray, mChunkedArray_Destroy);
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
  mDEFER_CALL(&chunkedArray, mChunkedArray_Destroy);
  mERROR_CHECK(mQueue_PeekAt(hashMap->data, index, &chunkedArray));

  mKeyValuePair<TKey, TValue> kvpair;
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
  mDEFER_CALL(&chunkedArray, mChunkedArray_Destroy);
  mERROR_CHECK(mQueue_PeekAt(hashMap->data, index, &chunkedArray));

  size_t count = 0;
  mERROR_CHECK(mChunkedArray_GetCount(chunkedArray, &count));

  for (size_t i = 0; i < count; ++i)
  {
    mKeyValuePair<TKey, TValue> *pKVPair = nullptr;
    mERROR_CHECK(mChunkedArray_PointerAt(chunkedArray, i, &pKVPair));

    if (pKVPair->key == key)
    {
      mKeyValuePair<TKey, TValue> kvpair;
      mERROR_CHECK(mChunkedArray_PopAt(chunkedArray, i, &kvpair));
      mERROR_CHECK(mDestruct(&kvpair.key));
      *pValue = std::move(kvpair.value);
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
    mDEFER_CALL(chunkedArray, mChunkedArray_Destroy);
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
