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

mFUNCTION(mBinaryChunk_WriteBytes, mPtr<mBinaryChunk> &binaryChunk, IN const uint8_t *pItems, const size_t bytes)
{
  mFUNCTION_SETUP();

  mERROR_IF(binaryChunk == nullptr || pItems == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mBinaryChunk_GrowBack(binaryChunk, bytes));

  uint8_t *pData = &binaryChunk->pData[binaryChunk->writeBytes];
  size_t writtenSize = 0;

#if defined(SSE2)
  while (writtenSize + sizeof(__m128i) <= bytes)
  {
    *((__m128i *)pData) = *(__m128i *)pItems;
    writtenSize += sizeof(__m128i);
    pData += sizeof(__m128i);
    pItems += sizeof(__m128i);
  }
#else
  while (writtenSize + sizeof(size_t) <= bytes)
  {
    *((size_t *)pData) = *(size_t *)pItems;
    writtenSize += sizeof(size_t);
    pData += sizeof(size_t);
    pItems += sizeof(size_t);
  }
#endif

  while (writtenSize < bytes)
  {
    *pData = *pItems;
    ++writtenSize;
    ++pData;
    ++pItems;
  }

  binaryChunk->writeBytes += bytes;

  mRETURN_SUCCESS();
}

mFUNCTION(mBinaryChunk_ReadBytes, mPtr<mBinaryChunk> &binaryChunk, OUT uint8_t *pItems, const size_t bytes)
{
  mFUNCTION_SETUP();

  mERROR_IF(binaryChunk == nullptr || pItems == nullptr, mR_ArgumentNull);
  mERROR_IF(binaryChunk->readBytes + bytes > binaryChunk->writeBytes, mR_IndexOutOfBounds);

  uint8_t *pData = &binaryChunk->pData[binaryChunk->writeBytes];
  size_t writtenSize = 0;

#if defined(SSE2)
  while (writtenSize + sizeof(__m128i) <= bytes)
  {
    *(__m128i *)pItems = *((__m128i *)pData);
    writtenSize += sizeof(__m128i);
    pData += sizeof(__m128i);
    pItems += sizeof(__m128i);
  }
#else
  while (writtenSize + sizeof(size_t) <= bytes)
  {
    *(size_t *)pItems = *((size_t *)pData);
    writtenSize += sizeof(size_t);
    pData += sizeof(size_t);
    pItems += sizeof(size_t);
  }
#endif

  while (writtenSize < bytes)
  {
    *pItems = *pData;
    ++writtenSize;
    ++pData;
    ++pItems;
  }

  binaryChunk->readBytes += bytes;

  mRETURN_SUCCESS();
}

mFUNCTION(mBinaryChunk_ResetWrite, mPtr<mBinaryChunk> &binaryChunk)
{
  mFUNCTION_SETUP();

  mERROR_IF(binaryChunk == nullptr, mR_ArgumentNull);

  binaryChunk->writeBytes = 0;

  mRETURN_SUCCESS();
}

mFUNCTION(mBinaryChunk_ResetRead, mPtr<mBinaryChunk> &binaryChunk)
{
  mFUNCTION_SETUP();

  mERROR_IF(binaryChunk == nullptr, mR_ArgumentNull);

  binaryChunk->readBytes = 0;

  mRETURN_SUCCESS();
}

mFUNCTION(mBinaryChunk_GetWriteBytes, mPtr<mBinaryChunk> &binaryChunk, OUT size_t *pSize)
{
  mFUNCTION_SETUP();

  mERROR_IF(binaryChunk == nullptr || pSize == nullptr, mR_ArgumentNull);

  *pSize = binaryChunk->writeBytes;

  mRETURN_SUCCESS();
}

mFUNCTION(mBinaryChunk_GetReadBytes, mPtr<mBinaryChunk> &binaryChunk, OUT size_t *pSize)
{
  mFUNCTION_SETUP();

  mERROR_IF(binaryChunk == nullptr || pSize == nullptr, mR_ArgumentNull);

  *pSize = binaryChunk->readBytes;

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
