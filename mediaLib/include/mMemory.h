// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef mMemory_h__
#define mMemory_h__

#include <memory.h>
#include <stdint.h>
#include "mResult.h"
#include "mDefer.h"
#include <climits>

template <typename T>
void mSetToNullptr(T **ppData)
{
  if(ppData != nullptr)
    *ppData = nullptr;
}

template <typename T> mFUNCTION(mMemset, IN_OUT T *pData, size_t count, uint8_t data = 0)
{
  mFUNCTION_SETUP();
  mERROR_IF(pData == nullptr, mR_ArgumentNull);
  mERROR_IF(sizeof(T) * count > INT_MAX, mR_InvalidParameter);

  memset(pData, (int)(sizeof(T) * count), (int)data);

  mRETURN_SUCCESS();
}

template <typename T> mFUNCTION(mAlloc, OUT T **ppData, const size_t count)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppData == nullptr, mR_ArgumentNull);
  mDEFER_DESTRUCTION_ON_ERROR(ppData, mSetToNullptr);

  T *pData = (T *)malloc(sizeof(T) * count);
  mERROR_IF(pData == nullptr, mR_MemoryAllocationFailure);

  *ppData = pData;
  mRETURN_SUCCESS();
}

template <typename T> mFUNCTION(mRealloc, OUT T **ppData, const size_t count)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppData == nullptr, mR_ArgumentNull);
  mDEFER_DESTRUCTION_ON_ERROR(ppData, mSetToNullptr);

  if (*ppData == nullptr)
  {
    mERROR_CHECK(mAlloc(ppData, count));
  }
  else
  {
    T *pData = (T *)realloc(*ppData, sizeof(T) * count);
    mERROR_IF(pData == nullptr, mR_MemoryAllocationFailure);

    *ppData = pData;
  }

  mRETURN_SUCCESS();
}

template <typename T> mFUNCTION(mAllocZero, OUT T **ppData, const size_t count)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppData == nullptr, mR_ArgumentNull);
  mERROR_CHECK(mAlloc(ppData, count));
  mERROR_CHECK(mMemset(*ppData, count));

  mRETURN_SUCCESS();
}

template <typename T> mFUNCTION(mAllocStack, OUT T **ppData, const size_t count)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppData == nullptr, mR_ArgumentNull);
  mDEFER_DESTRUCTION_ON_ERROR(ppData, mSetToNullptr);

  T *pData = (T *)_malloca(sizeof(T) * count);
  mERROR_IF(pData == nullptr, mR_MemoryAllocationFailure);

  *ppData = pData;
  mRETURN_SUCCESS();
}

template <typename T> mFUNCTION(mAllocStackZero, OUT T **ppData, const size_t count)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppData == nullptr, mR_ArgumentNull);
  mERROR_CHECK(mAllocStack(ppData, count));
  mERROR_CHECK(mMemset(*ppData, count));

  mRETURN_SUCCESS();
}

template <typename T> mFUNCTION(mFreePtr, IN_OUT T **ppData)
{
  mFUNCTION_SETUP();

  if (ppData != nullptr)
  {
    mDEFER_DESTRUCTION(ppData, mSetToNullptr);

    if (*ppData != nullptr)
      free(*ppData);
  }

  mRETURN_SUCCESS();
}

template <typename T> mFUNCTION(mFree, IN_OUT T *pData)
{
  mFUNCTION_SETUP();

  if (pData != nullptr)
    free(pData);

  mRETURN_SUCCESS();
}

template <typename T> mFUNCTION(mFreePtrStack, IN_OUT T **ppData)
{
  mFUNCTION_SETUP();

  if (ppData != nullptr)
  {
    mDEFER_DESTRUCTION(ppData, mSetToNullptr);

    if (*ppData != nullptr)
      _freea(*ppData);
  }

  mRETURN_SUCCESS();
}

template <typename T> mFUNCTION(mFreeStack, IN_OUT T *pData)
{
  mFUNCTION_SETUP();

  if (pData != nullptr)
    _freea(pData);

  mRETURN_SUCCESS();
}

template <typename T> mFUNCTION(mMemmove, T *pDst, T *pSrc, const size_t size)
{
  mFUNCTION_SETUP();
  mERROR_IF(pDst == nullptr || pSrc == nullptr, mR_ArgumentNull);

  if (size == 0)
    mRETURN_SUCCESS();

  memmove(pDst, pSrc, sizeof(T) * size);

  mRETURN_SUCCESS();
}

template <typename T> mFUNCTION(mMemcpy, T *pDst, T *pSrc, const size_t size)
{
  mFUNCTION_SETUP();
  mERROR_IF(pDst == nullptr || pSrc == nullptr, mR_ArgumentNull);

  if (size == 0)
    mRETURN_SUCCESS();

  memcpy(pDst, pSrc, sizeof(T) * size);

  mRETURN_SUCCESS();
}

#endif // mMemory_h__
