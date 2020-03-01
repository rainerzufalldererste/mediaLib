#ifndef mHashMap_h__
#define mHashMap_h__

#include "mediaLib.h"
#include "mQueue.h"
#include "mChunkedArray.h"
#include "mKeyValuePair.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "NsyZ/4oO/uRhWPGVwHU+pKXPpyaEnt/4cHH7dAlnYzDcozuwO724meP38dSCAY7zxocN5WvqxmnMHdTB"
#endif

template <typename TKey, typename TValue>
struct mHashMap
{
  mAllocator *pAllocator;
  mPtr<mQueue<mPtr<mChunkedArray<mKeyValuePair<TKey, TValue>>>>> data;
  size_t hashMapSize;
};

template <typename TKey, typename TValue>
mFUNCTION(mHashMap_Create, OUT mPtr<mHashMap<TKey, TValue>> *pHashMap, IN mAllocator *pAllocator, const size_t hashMapSize);

template <typename TKey, typename TValue>
mFUNCTION(mHashMap_Destroy, IN_OUT mPtr<mHashMap<TKey, TValue>> *pHashMap);

template <typename TKey, typename TValue>
mFUNCTION(mHashMap_Contains, mPtr<mHashMap<TKey, TValue>> &hashMap, TKey key, OUT bool *pContains, OUT TValue *pValueIfExistent);

template <typename TKey, typename TValue>
mFUNCTION(mHashMap_ContainsGetPointer, mPtr<mHashMap<TKey, TValue>> &hashMap, TKey key, OUT bool *pContains, OUT TValue **ppValueIfExistent);

template <typename TKey, typename TValue>
mFUNCTION(mHashMap_Add, mPtr<mHashMap<TKey, TValue>> &hashMap, TKey key, IN TValue *pValue);

template <typename TKey, typename TValue>
mFUNCTION(mHashMap_Get, mPtr<mHashMap<TKey, TValue>> &hashMap, TKey key, OUT TValue *pValue);

template <typename TKey, typename TValue>
mFUNCTION(mHashMap_Remove, mPtr<mHashMap<TKey, TValue>> &hashMap, TKey key, OUT TValue *pValue);

template <typename TKey, typename TValue>
mFUNCTION(mHashMap_SetKeyValuePairDestructionFunction, mPtr<mHashMap<TKey, TValue>> &hashMap, const std::function<mResult (mKeyValuePair<TKey, TValue> *)> &destructionFunction);

#include "mHashMap.inl"

#endif // mHashMap_h__
