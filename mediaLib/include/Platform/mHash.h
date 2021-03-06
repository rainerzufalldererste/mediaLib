// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef mHash_h__
#define mHash_h__

#include "default.h"

template<typename T>
mFUNCTION(mHash, mPtr<T> *pData, OUT uint64_t *pHash);

template<typename T>
mFUNCTION(mHash, mPtr<T> data, OUT uint64_t *pHash);

template<typename T>
mFUNCTION(mHash, std::basic_string<T> *pString, OUT uint64_t *pHash);

template<typename T>
mFUNCTION(mHash, const std::basic_string<T> &string, OUT uint64_t *pHash);

template <typename T>
mFUNCTION(mHash, const T *pData, OUT uint64_t *pHash, const size_t count = 1);

inline mFUNCTION(mHash, const mString &string, OUT uint64_t *pHash);

template<size_t TCount>
inline mFUNCTION(mHash, const mInplaceString<TCount> &string, OUT uint64_t *pHash);

uint64_t mMurmurHash2(const void *pData, const size_t length, const uint64_t seed = 0xD0D93DEADC0FFEE5);

//////////////////////////////////////////////////////////////////////////

template<typename T>
inline mFUNCTION(mHash, mPtr<T> *pData, OUT uint64_t *pHash)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mHash(pData->GetPointer(), pHash));

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mHash, mPtr<T> data, OUT uint64_t *pHash)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mHash(pData.GetPointer(), pHash));

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mHash, std::basic_string<T>* pString, OUT uint64_t * pHash)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mHash(pString->c_str(), pHash, pString->length()));

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mHash, const std::basic_string<T> &string, OUT uint64_t *pHash)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mHash(string.c_str(), pHash, string.length()));

  mRETURN_SUCCESS();
}

inline mFUNCTION(mHash, const mString &string, OUT uint64_t *pHash)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mHash(string.c_str(), pHash, string.bytes));

  mRETURN_SUCCESS();
}

template<size_t TCount>
inline mFUNCTION(mHash, const mInplaceString<TCount> &string, OUT uint64_t *pHash)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mHash(string.c_str(), pHash, string.bytes));

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mHash, const T *pData, OUT uint64_t *pHash, const size_t count /* = 1 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pData == nullptr || pHash == nullptr, mR_ArgumentNull);

  const size_t size = sizeof(T) * count;

  if (size >= 4)
  {
    *pHash = mMurmurHash2(pData, size);
  }
  else
  {
    uint8_t data[4] = { 0x13, 0xFA, 0x36, 0x92 };
    const uint8_t *pDataUint8 = (uint8_t *)pData;

    for (size_t i = 0; i < size; i++)
      data[i] = pDataUint8[i];

    *pHash = mMurmurHash2(data, sizeof(data));
  }

  mRETURN_SUCCESS();
}

#endif // mHash_h__
