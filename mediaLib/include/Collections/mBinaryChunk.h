#ifndef mBinaryChunk_h__
#define mBinaryChunk_h__

#include "mediaLib.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "De6gmXydgRjhIYW/hHVzSD/FImMHqkuUNobg+0CHS/89JmIJZILo2SZfWIiXlc8sxsmRzZFK1IjXpt5H"
#endif

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

mFUNCTION(mBinaryChunk_WriteBytes, mPtr<mBinaryChunk> &binaryChunk, IN const uint8_t *pItems, const size_t bytes);
mFUNCTION(mBinaryChunk_ReadBytes, mPtr<mBinaryChunk> &binaryChunk, OUT uint8_t *pItems, const size_t bytes);

mFUNCTION(mBinaryChunk_ResetWrite, mPtr<mBinaryChunk> &binaryChunk);
mFUNCTION(mBinaryChunk_ResetRead, mPtr<mBinaryChunk> &binaryChunk);

// Retrieves the number of bytes written to the binary chunk.
mFUNCTION(mBinaryChunk_GetWriteBytes, mPtr<mBinaryChunk> &binaryChunk, OUT size_t *pSize);

// Retrieves the number of bytes read from the binary chunk.
mFUNCTION(mBinaryChunk_GetReadBytes, mPtr<mBinaryChunk> &binaryChunk, OUT size_t *pSize);

//////////////////////////////////////////////////////////////////////////

template<typename T>
inline mFUNCTION(mBinaryChunk_Write, mPtr<mBinaryChunk> &binaryChunk, T *pItem)
{
  mFUNCTION_SETUP();

  mERROR_IF(binaryChunk == nullptr || pItem == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mBinaryChunk_GrowBack(binaryChunk, sizeof(T)));

  T *pData = (T *)&(binaryChunk->pData[binaryChunk->writeBytes]);
  *pData = *pItem;

  binaryChunk->writeBytes += sizeof(T);

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mBinaryChunk_WriteData, mPtr<mBinaryChunk> &binaryChunk, T item)
{
  mFUNCTION_SETUP();

  mERROR_IF(binaryChunk == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mBinaryChunk_GrowBack(binaryChunk, sizeof(T)));

  T *pData = (T *)&(binaryChunk->pData[binaryChunk->writeBytes]);
  *pData = item;

  binaryChunk->writeBytes += sizeof(T);

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mBinaryChunk_Read, mPtr<mBinaryChunk> &binaryChunk, T *pItem)
{
  mFUNCTION_SETUP();

  mERROR_IF(binaryChunk == nullptr || pItem == nullptr, mR_ArgumentNull);
  mERROR_IF(binaryChunk->readBytes + sizeof(T) > binaryChunk->writeBytes, mR_IndexOutOfBounds);

  T *pData = (T *)&(binaryChunk->pData[binaryChunk->readBytes]);
  *pItem = *pData;

  binaryChunk->readBytes += sizeof(T);

  mRETURN_SUCCESS();
}

#endif // mBinaryChunk_h__
