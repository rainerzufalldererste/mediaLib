// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef mBinaryChunk_h__
#define mBinaryChunk_h__

#include "default.h"

struct mBinaryChunk
{
  uint8_t *pData;
  size_t size;
  size_t writeBytes;
  size_t readBytes;
  mAllocator *pAllocator;
};

mFUNCTION(mBinaryChunk_Create, OUT mPtr<mBinaryChunk> *pBinaryChunk, IN mAllocator *pAllocator);
mFUNCTION(mBinaryChunk_Destroy, IN_OUT mPtr<mBinaryChunk> *pBinaryChunk);
mFUNCTION(mBinaryChunk_GrowBack, mPtr<mBinaryChunk> &binaryChunk, const size_t sizeToGrow);

template <typename T>
mFUNCTION(mBinaryChunk_Write, mPtr<mBinaryChunk> &binaryChunk, T *pItem);

template <typename T>
mFUNCTION(mBinaryChunk_WriteData, mPtr<mBinaryChunk> &binaryChunk, T item);

template <typename T>
mFUNCTION(mBinaryChunk_Read, mPtr<mBinaryChunk> &binaryChunk, T *pItem);

mFUNCTION(mBinaryChunk_ResetWrite, mPtr<mBinaryChunk> &binaryChunk);
mFUNCTION(mBinaryChunk_ResetRead, mPtr<mBinaryChunk> &binaryChunk);

//////////////////////////////////////////////////////////////////////////

template<typename T>
inline mFUNCTION(mBinaryChunk_Write, mPtr<mBinaryChunk>& binaryChunk, T *pItem)
{
  mFUNCTION_SETUP();

  mERROR_IF(binaryChunk == nullptr || pItem == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mBinaryChunk_GrowBack(binaryChunk, sizeof(T)));

  uint8_t *pData = &binaryChunk->pData[binaryChunk->writeBytes];
  size_t writtenSize = 0;
  uint8_t *pItemData = (uint8_t *)pItem;

#if defined(SSE2)
  while (writtenSize + sizeof(__m128i) <= sizeof(T))
  {
    *((__m128i *)pData) = *(__m128i *)pItemData;
    writtenSize += sizeof(__m128i);
    pData += sizeof(__m128i);
    pItemData += sizeof(__m128i);
  }
#endif

  while (writtenSize + sizeof(size_t) <= sizeof(T))
  {
    *((size_t *)pData) = *(size_t *)pItemData;
    writtenSize += sizeof(size_t);
    pData += sizeof(size_t);
    pItemData += sizeof(size_t);
  }

  while (writtenSize < sizeof(T))
  {
    *pData = *pItemData;
    ++writtenSize;
    ++pData;
    ++pItemData;
  }

  binaryChunk->writeBytes += sizeof(T);

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mBinaryChunk_WriteData, mPtr<mBinaryChunk>& binaryChunk, T item)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mBinaryChunk_Write<T>(binaryChunk, &item));

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mBinaryChunk_Read, mPtr<mBinaryChunk>& binaryChunk, T *pItem)
{
  mFUNCTION_SETUP();

  mERROR_IF(binaryChunk == nullptr || pItem == nullptr, mR_ArgumentNull);
  mERROR_IF(writeCount + sizeof(T) > readCount, mR_IndexOutOfBounds);

  uint8_t *pData = &binaryChunk->pData[binaryChunk->writeBytes];
  size_t writtenSize = 0;
  uint8_t *pItemData = (uint8_t *)pItem;

#if defined(SSE2)
  while (writtenSize + sizeof(__m128i) <= sizeof(T))
  {
    *(__m128i *)pItemData = *((__m128i *)pData);
    writtenSize += sizeof(__m128i);
    pData += sizeof(__m128i);
    pItemData += sizeof(__m128i);
  }
#else
  while (writtenSize + sizeof(size_t) <= sizeof(T))
  {
    *(size_t *)pItemData = *((size_t *)pData);
    writtenSize += sizeof(size_t);
    pData += sizeof(size_t);
    pItemData += sizeof(size_t);
  }
#endif

  while (writtenSize < sizeof(T))
  {
    *pItemData = *pData;
    ++writtenSize;
    ++pData;
    ++pItemData;
  }

  binaryChunk->readBytes += sizeof(T);

  mRETURN_SUCCESS();
}

#endif // mBinaryChunk_h__
