// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "mBinaryChunk.h"

mFUNCTION(mBinaryChunk_Destroy_Internal, mBinaryChunk *pBinaryChunk);

mFUNCTION(mBinaryChunk_Create, OUT mPtr<mBinaryChunk> *pBinaryChunk, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pBinaryChunk == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Allocate(pBinaryChunk, pAllocator, (std::function<void(mBinaryChunk *)>)[](mBinaryChunk *pData) { mBinaryChunk_Destroy_Internal(pData); }, 1));
  (*pBinaryChunk)->pAllocator = pAllocator;

  mRETURN_SUCCESS();
}

mFUNCTION(mBinaryChunk_Destroy, IN_OUT mPtr<mBinaryChunk> *pBinaryChunk)
{
  mFUNCTION_SETUP();

  mERROR_IF(pBinaryChunk == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Destroy(pBinaryChunk));

  mRETURN_SUCCESS();
}

mFUNCTION(mBinaryChunk_GrowBack, mPtr<mBinaryChunk> &binaryChunk, const size_t sizeToGrow)
{
  mFUNCTION_SETUP();

  mERROR_IF(binaryChunk == nullptr, mR_ArgumentNull);

  if (binaryChunk->pData == nullptr)
  {
    mERROR_CHECK(mAllocator_AllocateZero(binaryChunk->pAllocator, &binaryChunk->pData, sizeToGrow));
    binaryChunk->size = sizeToGrow;
  }
  else if (binaryChunk->writeBytes + sizeToGrow > binaryChunk->size)
  {
    size_t newSize;

    if (binaryChunk->writeBytes + sizeToGrow > binaryChunk->size * 2)
      newSize = binaryChunk->writeBytes + sizeToGrow;
    else
      newSize = binaryChunk->size * 2;

    mERROR_CHECK(mAllocator_Reallocate(binaryChunk->pAllocator, &binaryChunk->pData, newSize));
    mERROR_CHECK(mMemset(&binaryChunk->pData[binaryChunk->size], newSize - binaryChunk->size));

    binaryChunk->size = newSize;
  }

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mBinaryChunk_Destroy_Internal, mBinaryChunk *pBinaryChunk)
{
  mFUNCTION_SETUP();

  if (pBinaryChunk->pData != nullptr)
  {
    mAllocator *pAllocator = pBinaryChunk->pAllocator;
    mERROR_CHECK(mAllocator_FreePtr(pAllocator, &pBinaryChunk->pData));
  }

  mRETURN_SUCCESS();
}
