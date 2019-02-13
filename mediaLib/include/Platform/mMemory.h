#ifndef mMemory_h__
#define mMemory_h__

#include "mediaLib.h"

template <typename T>
void mSetToNullptr(T **ppData)
{
  if(ppData != nullptr)
    *ppData = nullptr;
}

template <typename T>
mFUNCTION(mMemset, IN_OUT T *pData, size_t count, uint8_t data = 0)
{
  mFUNCTION_SETUP();
  mERROR_IF(pData == nullptr, mR_ArgumentNull);

  memset(pData, (int)data, sizeof(T) * count);

  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mZeroMemory, IN_OUT T *pData, size_t count = 1)
{
  mFUNCTION_SETUP();
  mERROR_IF(pData == nullptr, mR_ArgumentNull);

  memset(pData, 0, sizeof(T) * count);

  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mAlloc, OUT T **ppData, const size_t count)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppData == nullptr, mR_ArgumentNull);
  mDEFER_CALL_ON_ERROR(ppData, mSetToNullptr);

  T *pData = (T *)malloc(sizeof(T) * count);
  mERROR_IF(pData == nullptr, mR_MemoryAllocationFailure);

  *ppData = pData;
  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mRealloc, OUT T **ppData, const size_t count)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppData == nullptr, mR_ArgumentNull);
  mDEFER_CALL_ON_ERROR(ppData, mSetToNullptr);

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

template <typename T>
mFUNCTION(mAllocZero, OUT T **ppData, const size_t count)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppData == nullptr, mR_ArgumentNull);
  mERROR_CHECK(mAlloc(ppData, count));
  mERROR_CHECK(mMemset(*ppData, count));

  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mAllocStack, OUT T **ppData, const size_t count)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppData == nullptr, mR_ArgumentNull);
  mDEFER_CALL_ON_ERROR(ppData, mSetToNullptr);

  T *pData = (T *)_malloca(sizeof(T) * count);
  mERROR_IF(pData == nullptr, mR_MemoryAllocationFailure);

  *ppData = pData;
  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mAllocStackZero, OUT T **ppData, const size_t count)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppData == nullptr, mR_ArgumentNull);
  mERROR_CHECK(mAllocStack(ppData, count));
  mERROR_CHECK(mMemset(*ppData, count));

  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mFreePtr, IN_OUT T **ppData)
{
  mFUNCTION_SETUP();

  if (ppData != nullptr)
  {
    mDEFER_CALL(ppData, mSetToNullptr);

    if (*ppData != nullptr)
      free(*ppData);
  }

  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mFree, IN_OUT T *pData)
{
  mFUNCTION_SETUP();

  if (pData != nullptr)
    free(pData);

  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mFreePtrStack, IN_OUT T **ppData)
{
  mFUNCTION_SETUP();

  if (ppData != nullptr)
  {
    mDEFER_CALL(ppData, mSetToNullptr);

    if (*ppData != nullptr)
      _freea(*ppData);
  }

  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mFreeStack, IN_OUT T *pData)
{
  mFUNCTION_SETUP();

  if (pData != nullptr)
    _freea(pData);

  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mAllocAlligned, OUT T **ppData, const size_t count, const size_t alignment)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppData == nullptr, mR_ArgumentNull);
  mDEFER_CALL_ON_ERROR(ppData, mSetToNullptr);

  T *pData = (T *)_aligned_malloc(sizeof(T) * count);
  mERROR_IF(pData == nullptr, mR_MemoryAllocationFailure);

  *ppData = pData;
  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mReallocAlligned, OUT T **ppData, const size_t count, const size_t alignment)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppData == nullptr, mR_ArgumentNull);
  mDEFER_CALL_ON_ERROR(ppData, mSetToNullptr);

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

template <typename T>
mFUNCTION(mAllocAllignedZero, OUT T **ppData, const size_t count, const size_t alignment)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppData == nullptr, mR_ArgumentNull);
  mERROR_CHECK(mAllocAlligned(ppData, count, alignment));
  mERROR_CHECK(mMemset(*ppData, count));

  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mFreeAllignedPtr, IN_OUT T **ppData)
{
  mFUNCTION_SETUP();

  if (ppData != nullptr)
  {
    mDEFER_CALL(ppData, mSetToNullptr);

    if (*ppData != nullptr)
      _aligned_free(*ppData);
  }

  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mAllignedFree, IN_OUT T *pData)
{
  mFUNCTION_SETUP();

  if (pData != nullptr)
    _aligned_free(pData);

  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mMemmove, T *pDst, T *pSrc, const size_t count)
{
  mFUNCTION_SETUP();
  mERROR_IF(pDst == nullptr || pSrc == nullptr, mR_ArgumentNull);

  if (count == 0)
    mRETURN_SUCCESS();

  memmove(pDst, pSrc, sizeof(T) * count);

  mRETURN_SUCCESS();
}

template <typename T>
mFUNCTION(mMemcpy, T *pDst, const T *pSrc, const size_t count)
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

mFUNCTION(mStringLength, const char *text, const size_t maxLength, OUT size_t *pLength);
mFUNCTION(mSprintf, OUT char *buffer, const size_t bufferLength, const char *formatString, ...);
mFUNCTION(mSprintfWithLength, OUT char *buffer, const size_t bufferLength, const char *formatString, OUT size_t *pLength, ...);
mFUNCTION(mStringCopy, OUT char *buffer, const size_t bufferLength, const char *source, const size_t sourceLength);

template <size_t TCount>
mFUNCTION(mStringChar, const char text[TCount], const char character, OUT size_t *pLength)
{
  mFUNCTION_SETUP();

  mERROR_IF(text == nullptr || pLength == nullptr, mR_ArgumentNull);

  *pLength = (size_t)((char *)memchr((void *)text, (int)character, strnlen_s(text, TCount)) - (char *)text);

  mRETURN_SUCCESS();
}

mFUNCTION(mStringChar, const char *text, const size_t maxLength, const char character, OUT size_t *pLength);

template <typename ...Args>
inline mFUNCTION(mSScanf, const char *text, const char *format, Args ...args)
{
  mFUNCTION_SETUP();

  const int length = sscanf(text, format, args...);

  if (length < 0)
    mRETURN_RESULT(mR_Failure);
  else if (sizeof...(args) != length)
    mRETURN_RESULT(mR_InvalidParameter);

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mMemoryAlignHiAny, const size_t size, const size_t align, OUT size_t *pResult);
mFUNCTION(mMemoryAlignLoAny, const size_t size, const size_t align, OUT size_t *pResult);
mFUNCTION(mMemoryAlignHi, const size_t size, const size_t align, OUT size_t *pResult);
mFUNCTION(mMemoryAlignHi, IN const void *pData, const size_t align, OUT void **ppResult);
mFUNCTION(mMemoryAlignLo, const size_t size, const size_t align, OUT size_t *pResult);
mFUNCTION(mMemoryAlignLo, IN const void *pData, const size_t align, OUT void **ppResult);
mFUNCTION(mMemoryIsAligned, const size_t size, const size_t align, OUT bool *pResult);
mFUNCTION(mMemoryIsAligned, IN const void *pData, const size_t align, OUT bool *pResult);

#endif // mMemory_h__
