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

  memset(pData, (int)data, (int)(sizeof(T) * count));

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

template <typename T> mFUNCTION(mAllocAlligned, OUT T **ppData, const size_t count, const size_t alignment)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppData == nullptr, mR_ArgumentNull);
  mDEFER_DESTRUCTION_ON_ERROR(ppData, mSetToNullptr);

  T *pData = (T *)_aligned_malloc(sizeof(T) * count);
  mERROR_IF(pData == nullptr, mR_MemoryAllocationFailure);

  *ppData = pData;
  mRETURN_SUCCESS();
}

template <typename T> mFUNCTION(mReallocAlligned, OUT T **ppData, const size_t count, const size_t alignment)
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
    T *pData = (T *)_aligned_realloc(*ppData, sizeof(T) * count);
    mERROR_IF(pData == nullptr, mR_MemoryAllocationFailure);

    *ppData = pData;
  }

  mRETURN_SUCCESS();
}

template <typename T> mFUNCTION(mAllocAllignedZero, OUT T **ppData, const size_t count, const size_t alignment)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppData == nullptr, mR_ArgumentNull);
  mERROR_CHECK(mAllocAlligned(ppData, count, alignment));
  mERROR_CHECK(mMemset(*ppData, count));

  mRETURN_SUCCESS();
}

template <typename T> mFUNCTION(mFreeAllignedPtr, IN_OUT T **ppData)
{
  mFUNCTION_SETUP();

  if (ppData != nullptr)
  {
    mDEFER_DESTRUCTION(ppData, mSetToNullptr);

    if (*ppData != nullptr)
      _aligned_free(*ppData);
  }

  mRETURN_SUCCESS();
}

template <typename T> mFUNCTION(mAllignedFree, IN_OUT T *pData)
{
  mFUNCTION_SETUP();

  if (pData != nullptr)
    _aligned_free(pData);

  mRETURN_SUCCESS();
}

template <typename T> mFUNCTION(mMemmove, T *pDst, T *pSrc, const size_t count)
{
  mFUNCTION_SETUP();
  mERROR_IF(pDst == nullptr || pSrc == nullptr, mR_ArgumentNull);

  if (count == 0)
    mRETURN_SUCCESS();

  memmove(pDst, pSrc, sizeof(T) * count);

  mRETURN_SUCCESS();
}

template <typename T> mFUNCTION(mMemcpy, T *pDst, T *pSrc, const size_t count)
{
  mFUNCTION_SETUP();
  mERROR_IF(pDst == nullptr || pSrc == nullptr, mR_ArgumentNull);

  if (count == 0)
    mRETURN_SUCCESS();

  memcpy(pDst, pSrc, sizeof(T) * count);

  mRETURN_SUCCESS();
}

template <size_t TCount>
mFUNCTION(mStringLength, const char text[TCount], OUT size_t *pLength)
{
  mFUNCTION_SETUP();

  mERROR_IF(text == nullptr || pLength == nullptr, mR_ArgumentNull);

  *pLength = strnlen_s(text, TCount);

  mRETURN_SUCCESS();
}

inline mFUNCTION(mStringLength, const char *text, const size_t maxLength, OUT size_t *pLength)
{
  mFUNCTION_SETUP();

  mERROR_IF(text == nullptr || pLength == nullptr, mR_ArgumentNull);

  *pLength = strnlen_s(text, maxLength);

  mRETURN_SUCCESS();
}

template <size_t TCount>
mFUNCTION(mStringChar, const char text[TCount], const char character, OUT size_t *pLength)
{
  mFUNCTION_SETUP();

  mERROR_IF(text == nullptr || pLength == nullptr, mR_ArgumentNull);

  *pLength = (size_t)((char *)memchr((void *)text, (int)character, strnlen_s(text, TCount)) - (char *)text);

  mRETURN_SUCCESS();
}

mFUNCTION(mStringChar, const char *text, const size_t maxLength, const char character, OUT size_t *pLength)
{
  mFUNCTION_SETUP();

  mERROR_IF(text == nullptr || pLength == nullptr, mR_ArgumentNull);

  *pLength = (size_t)((char *)memchr((void *)text, (int)character, strnlen_s(text, maxLength)) - (char *)text);

  mRETURN_SUCCESS();
}

/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2018 Yermalayeu Ihar.
*               2016-2016 Sintegrial Technologies.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

mINLINE mFUNCTION(mMemoryAlignHiAny, const size_t size, const size_t align, OUT size_t *pResult)
{
  mFUNCTION_SETUP();

  mERROR_IF(pResult == nullptr, mR_ArgumentNull);

  *pResult = (size + align - 1) / align * align;

  mRETURN_SUCCESS();
}

mINLINE mFUNCTION(mMemoryAlignLoAny, const size_t size, const size_t align, OUT size_t *pResult)
{
  mFUNCTION_SETUP();

  mERROR_IF(pResult == nullptr, mR_ArgumentNull);

  *pResult = size / align * align;

  mRETURN_SUCCESS();
}

mINLINE mFUNCTION(mMemoryAlignHi, const size_t size, const size_t align, OUT size_t *pResult)
{
  mFUNCTION_SETUP();

  mERROR_IF(pResult == nullptr, mR_ArgumentNull);

  *pResult = (size + align - 1) & ~(align - 1);

  mRETURN_SUCCESS();
}

mINLINE mFUNCTION(mMemoryAlignHi, const void *pData, const size_t align, OUT void **ppResult)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppResult == nullptr, mR_ArgumentNull);

  *ppResult = (void *)((((size_t)pData) + align - 1) & ~(align - 1));

  mRETURN_SUCCESS();
}

mINLINE mFUNCTION(mMemoryAlignLo, const size_t size, const size_t align, OUT size_t *pResult)
{
  mFUNCTION_SETUP();

  mERROR_IF(pResult == nullptr, mR_ArgumentNull);

  *pResult = size & ~(align - 1);

  mRETURN_SUCCESS();
}

mINLINE mFUNCTION(mMemoryAlignLo, const void *pData, const size_t align, OUT void **ppResult)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppResult == nullptr, mR_ArgumentNull);

  *ppResult = (void *)(((size_t)pData) & ~(align - 1));

  mRETURN_SUCCESS();
}

mINLINE mFUNCTION(mMemoryIsAligned, const size_t size, const size_t align, OUT bool *pResult)
{
  mFUNCTION_SETUP();

  mERROR_IF(pResult == nullptr, mR_ArgumentNull);

  size_t allignedSize;
  mERROR_CHECK(mMemoryAlignLo(size, align, &allignedSize));

  *pResult = (size == allignedSize);

  mRETURN_SUCCESS();
}

mINLINE mFUNCTION(mMemoryIsAligned, const void *pData, const size_t align, OUT bool *pResult)
{
  mFUNCTION_SETUP();

  mERROR_IF(pResult == nullptr, mR_ArgumentNull);

  void *pAligned;
  mERROR_CHECK(mMemoryAlignLo(pData, align, &pAligned));

  *pResult = (pData == pAligned);

  mRETURN_SUCCESS();
}

#endif // mMemory_h__
