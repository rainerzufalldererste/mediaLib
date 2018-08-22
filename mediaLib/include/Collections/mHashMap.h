// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef mHashMap_h__
#define mHashMap_h__

#include "default.h"
#include "mQueue.h"
#include "mChunkedArray.h"
#include "mKeyValuePair.h"

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
mFUNCTION(mHashMap_Contains, mHashMap<TKey, TValue> &hashMap, TKey key, OUT bool *pContains, OUT TValue *pValueIfExistent);

template <typename TKey, typename TValue>
mFUNCTION(mHashMap_ContainsGetPointer, mHashMap<TKey, TValue> &hashMap, TKey key, OUT bool *pContains, OUT TValue **ppValueIfExistent);

template <typename TKey, typename TValue>
mFUNCTION(mHashMap_Add, mHashMap<TKey, TValue> &hashMap, TKey key, IN TValue *pValue);

template <typename TKey, typename TValue>
mFUNCTION(mHashMap_Get, mHashMap<TKey, TValue> &hashMap, TKey key, OUT TValue *pValue);

template <typename TKey, typename TValue>
mFUNCTION(mHashMap_Remove, mHashMap<TKey, TValue> &hashMap, TKey key, OUT TValue *pValue);

template <typename TKey, typename TValue>
mFUNCTION(mHashMap_SetKeyValuePairDestructionFunction, mHashMap<TKey, TValue> &hashMap, const std::function<mResult (mKeyValuePair<TKey, TValue> *)> &destructionFunction);

#include "mHashMap.inl"

#endif // mHashMap_h__
