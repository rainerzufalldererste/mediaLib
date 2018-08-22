// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef mKeyValuePair_h__
#define mKeyValuePair_h__

#include "default.h"

template <typename TKey, typename TValue>
struct mKeyValuePair
{
  TKey key;
  TValue value;

  mKeyValuePair() = default;
  mKeyValuePair(TKey key, TValue value) : key(key), value(value) {}
};

template <typename TKey, typename TValue>
mFUNCTION(mKeyValuePair_Create, OUT mKeyValuePair *pPair, TKey key, TValue value);

template <typename TKey, typename TValue>
mFUNCTION(mKeyValuePair_Destroy, IN_OUT mKeyValuePair *pPair);

//////////////////////////////////////////////////////////////////////////

template <typename TKey, typename TValue>
mFUNCTION(mKeyValuePair_Create, OUT mKeyValuePair<TKey, TValue> *pPair, TKey key, TValue value)
{
  mFUNCTION_SETUP();

  mERROR_IF(pPair == nullptr, mR_ArgumentNull);

  *pPair->key = key;
  *pPair->value = value;

  mRETURN_SUCCESS();
}

template<typename TKey, typename TValue>
inline mFUNCTION(mKeyValuePair_Destroy, IN_OUT mKeyValuePair *pPair)
{
  mFUNCTION_SETUP();

  mRETURN_SUCCESS();
}

#endif // mKeyValuePair_h__
