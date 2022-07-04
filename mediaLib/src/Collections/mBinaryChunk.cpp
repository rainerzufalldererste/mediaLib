#include "mBinaryChunk.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "sFhKy5lOa3kejJouLspA0hEKRaJC3MjigM8ThoUO+aVIfM/UalYLAD37kAevPzxxncRJ7+tO2I2WpDby"
#endif

static mFUNCTION(mBinaryChunk_Destroy_Internal, mBinaryChunk *pBinaryChunk);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mBinaryChunk_Create, OUT mPtr<mBinaryChunk> *pBinaryChunk, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pBinaryChunk == nullptr, mR_ArgumentNull);

  mDEFER_CALL_ON_ERROR(pBinaryChunk, mSharedPointer_Destroy);
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

  mMemcpy(&binaryChunk->pData[binaryChunk->writeBytes], pItems, bytes);
  binaryChunk->writeBytes += bytes;

  mRETURN_SUCCESS();
}

mFUNCTION(mBinaryChunk_ReadBytes, mPtr<mBinaryChunk> &binaryChunk, OUT uint8_t *pItems, const size_t bytes)
{
  mFUNCTION_SETUP();

  mERROR_IF(binaryChunk == nullptr || pItems == nullptr, mR_ArgumentNull);
  mERROR_IF(binaryChunk->readBytes + bytes > binaryChunk->writeBytes, mR_IndexOutOfBounds);

  mMemcpy(pItems, &binaryChunk->pData[binaryChunk->writeBytes], bytes);
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

mFUNCTION(mBinaryChunk_GetWriteBytes, const mPtr<mBinaryChunk> &binaryChunk, OUT size_t *pSize)
{
  mFUNCTION_SETUP();

  mERROR_IF(binaryChunk == nullptr || pSize == nullptr, mR_ArgumentNull);

  *pSize = binaryChunk->writeBytes;

  mRETURN_SUCCESS();
}

mFUNCTION(mBinaryChunk_GetReadBytes, const mPtr<mBinaryChunk> &binaryChunk, OUT size_t *pSize)
{
  mFUNCTION_SETUP();

  mERROR_IF(binaryChunk == nullptr || pSize == nullptr, mR_ArgumentNull);

  *pSize = binaryChunk->readBytes;

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

static mFUNCTION(mBinaryChunk_Destroy_Internal, mBinaryChunk *pBinaryChunk)
{
  mFUNCTION_SETUP();

  if (pBinaryChunk->pData != nullptr)
  {
    mAllocator *pAllocator = pBinaryChunk->pAllocator;
    mERROR_CHECK(mAllocator_FreePtr(pAllocator, &pBinaryChunk->pData));
  }

  mRETURN_SUCCESS();
}
