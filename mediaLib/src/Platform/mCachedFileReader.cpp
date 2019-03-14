#include "mCachedFileReader.h"
#include "mFile.h"

#include <io.h>
#include <fcntl.h>

mFUNCTION(mCachedFileReader_Destroy_Internal, mCachedFileReader *pCachedFileReader);
mFUNCTION(mCachedFileReader_ReadFrom_Internal, mPtr<mCachedFileReader> &cachedFileReader, const size_t location, const size_t requestedSize);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mCachedFileReader_Create, OUT mPtr<mCachedFileReader> *pCachedFileReader, IN mAllocator *pAllocator, const mString &filename, const size_t maxCacheSize /* = 1024 * 1024 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pCachedFileReader == nullptr, mR_ArgumentNull);

  wchar_t wfilename[1024];
  mERROR_CHECK(mString_ToWideString(filename, wfilename, mARRAYSIZE(wfilename)));

  // Does the file exist?
  {
    bool fileExists = false;
    mERROR_CHECK(mFile_Exists(wfilename, &fileExists));
    mERROR_IF(!fileExists, mR_ResourceNotFound);
  }

  mERROR_CHECK(mSharedPointer_Allocate(pCachedFileReader, pAllocator, (std::function<void(mCachedFileReader *)>)[](mCachedFileReader *pData) {mCachedFileReader_Destroy_Internal(pData);}, 1));
  mDEFER_CALL_ON_ERROR(pCachedFileReader, mSharedPointer_Destroy);

  (*pCachedFileReader)->pAllocator = pAllocator;
  (*pCachedFileReader)->maxCacheSize = maxCacheSize;

  const errno_t result = _wsopen_s(&(*pCachedFileReader)->fileHandle, wfilename, _O_BINARY, _SH_DENYNO, _S_IREAD);
  mERROR_IF(result != 0, mR_ResourceNotFound);

  const int64_t fileSize = _lseeki64((*pCachedFileReader)->fileHandle, 0, SEEK_END);
  mERROR_IF(fileSize < 0, mR_IOFailure);
  (*pCachedFileReader)->fileSize = (size_t)fileSize;

  mERROR_IF(0 != _lseeki64((*pCachedFileReader)->fileHandle, 0, SEEK_SET), mR_IOFailure); // Reset read position.

  (*pCachedFileReader)->cachePosition = (size_t)-1;
  (*pCachedFileReader)->cacheSize = (size_t)0;

  mRETURN_SUCCESS();
}

mFUNCTION(mCachedFileReader_Destroy, IN_OUT mPtr<mCachedFileReader> *pCachedFileReader)
{
  mFUNCTION_SETUP();

  mERROR_IF(pCachedFileReader == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Destroy(pCachedFileReader));

  mRETURN_SUCCESS();
}

mFUNCTION(mCachedFileReader_GetSize, mPtr<mCachedFileReader> &cachedFileReader, OUT size_t *pFileSize)
{
  mFUNCTION_SETUP();

  mERROR_IF(pFileSize == nullptr || cachedFileReader == nullptr, mR_ArgumentNull);

  *pFileSize = cachedFileReader->fileSize;

  mRETURN_SUCCESS();
}

mFUNCTION(mCachedFileReader_ReadAt, mPtr<mCachedFileReader> &cachedFileReader, const size_t location, const size_t size, OUT uint8_t *pBuffer)
{
  mFUNCTION_SETUP();

  mERROR_IF(cachedFileReader == nullptr || pBuffer == nullptr, mR_ArgumentNull);
  mERROR_IF(location >= cachedFileReader->fileSize, mR_IndexOutOfBounds);
  mERROR_IF(location + size > cachedFileReader->fileSize, mR_ArgumentOutOfBounds);

  size_t readSize = 0;
  size_t currentLocation = location;
  uint8_t *pCurrentBufferPosition = pBuffer;

  do 
  {
    const size_t remainingSize = size - readSize;

    if (remainingSize == 0)
    {
      mRETURN_SUCCESS();
    }
    else if (cachedFileReader->cachePosition <= currentLocation && (int64_t)cachedFileReader->cacheSize - (int64_t)(currentLocation - cachedFileReader->cachePosition) >= (int64_t)remainingSize) // If it's currently cached.
    {
      // If we could preload the next block already.
      if (remainingSize < (cachedFileReader->maxCacheSize >> 1) && cachedFileReader->cachePosition + (cachedFileReader->maxCacheSize >> 1) <= currentLocation && cachedFileReader->cachePosition + cachedFileReader->maxCacheSize < cachedFileReader->fileSize)
      {
        // Preload half of the maxCacheSize.
        mERROR_CHECK(mCachedFileReader_ReadFrom_Internal(cachedFileReader, cachedFileReader->cachePosition + cachedFileReader->maxCacheSize, 1));
        mERROR_CHECK(mMemcpy(pCurrentBufferPosition, cachedFileReader->pCache + (currentLocation - cachedFileReader->cachePosition), remainingSize));
        mRETURN_SUCCESS();
      }
      else
      {
        // Serve from cache.
        mERROR_CHECK(mMemcpy(pCurrentBufferPosition, cachedFileReader->pCache + (currentLocation - cachedFileReader->cachePosition), remainingSize));
        mRETURN_SUCCESS();
      }
    }
    else if (remainingSize <= cachedFileReader->maxCacheSize)
    {
      mERROR_CHECK(mCachedFileReader_ReadFrom_Internal(cachedFileReader, currentLocation, remainingSize));
      mERROR_CHECK(mMemcpy(pCurrentBufferPosition, cachedFileReader->pCache + (currentLocation - cachedFileReader->cachePosition), remainingSize));
      mRETURN_SUCCESS();
    }
    else
    {
      mERROR_CHECK(mCachedFileReader_ReadFrom_Internal(cachedFileReader, currentLocation, remainingSize));
      mERROR_CHECK(mMemcpy(pCurrentBufferPosition, cachedFileReader->pCache + (currentLocation - cachedFileReader->cachePosition), cachedFileReader->maxCacheSize));

      pCurrentBufferPosition += cachedFileReader->maxCacheSize;
      currentLocation += cachedFileReader->maxCacheSize;
      readSize += cachedFileReader->maxCacheSize;
    }
  } while (true);

  mRETURN_SUCCESS();
}

mFUNCTION(mCachedFileReader_PointerAt, mPtr<mCachedFileReader> &cachedFileReader, const size_t location, const size_t size, OUT uint8_t **ppBuffer)
{
  mFUNCTION_SETUP();

  mERROR_IF(cachedFileReader == nullptr || ppBuffer == nullptr, mR_ArgumentNull);
  mERROR_IF(location >= cachedFileReader->fileSize, mR_IndexOutOfBounds);
  mERROR_IF(location + size > cachedFileReader->fileSize, mR_ArgumentOutOfBounds);
  mERROR_IF(size > cachedFileReader->maxCacheSize, mR_ArgumentOutOfBounds);
  mERROR_IF(size == 0, mR_Success);

   if (cachedFileReader->cachePosition <= location && (int64_t)cachedFileReader->cacheSize - (int64_t)(location - cachedFileReader->cachePosition) >= (int64_t)size)
  {
    // Serve from cache.
  }
  else // if size < cachedFileReader->maxCacheSize
  {
    mERROR_CHECK(mCachedFileReader_ReadFrom_Internal(cachedFileReader, location, size));
  }

  *ppBuffer = cachedFileReader->pCache + (location - cachedFileReader->cachePosition);

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mCachedFileReader_Destroy_Internal, mCachedFileReader *pCachedFileReader)
{
  mFUNCTION_SETUP();

  mERROR_IF(pCachedFileReader == nullptr, mR_ArgumentNull);

  if (pCachedFileReader->fileHandle)
    /* errno_t errorCode = */ _close(pCachedFileReader->fileHandle);

  mERROR_CHECK(mAllocator_FreePtr(pCachedFileReader->pAllocator, &pCachedFileReader->pCache));

  mRETURN_SUCCESS();
}

mFUNCTION(mCachedFileReader_ReadFrom_Internal, mPtr<mCachedFileReader> &cachedFileReader, const size_t location, const size_t requestedSize)
{
  mFUNCTION_SETUP();

  mERROR_IF(location >= cachedFileReader->fileSize, mR_IndexOutOfBounds);

  if (cachedFileReader->pCache == nullptr)
    mERROR_CHECK(mAllocator_AllocateZero(cachedFileReader->pAllocator, &cachedFileReader->pCache, cachedFileReader->maxCacheSize));

  size_t readSize = cachedFileReader->maxCacheSize;
  size_t readPos = location;
  size_t readOffset = 0;

  if (cachedFileReader->cachePosition == (size_t)-1)
  {
    // Read whole block.
    readSize = mMin(cachedFileReader->fileSize - location, cachedFileReader->maxCacheSize);

    cachedFileReader->cachePosition = location;
    cachedFileReader->cacheSize = readSize;
  }
  else
  {
    const int64_t diff = (int64_t)location - (int64_t)cachedFileReader->cachePosition;
    const size_t diffSize = (size_t)mAbs(diff);

    if (diff > 0 && diffSize < (readSize >> 1) && requestedSize < (readSize >> 1))
    {
      mRETURN_SUCCESS();
    }
    else if (requestedSize < (readSize >> 1) && diffSize <= readSize && diff > 0) // cachePosition < location
    {
      const size_t target = (size_t)mMax((int64_t)location - (int64_t)(readSize >> 1), (int64_t)0);

      if (target == cachedFileReader->cachePosition)
        mRETURN_SUCCESS();

      const size_t move = target - cachedFileReader->cachePosition;
      const size_t moveSize = readSize - move;

      mERROR_CHECK(mMemmove(cachedFileReader->pCache, cachedFileReader->pCache + move, moveSize));

      readSize = move;
      readOffset = moveSize;
      readPos = location;

      cachedFileReader->cachePosition = target;
      cachedFileReader->cacheSize = mMin(cachedFileReader->fileSize - target, cachedFileReader->maxCacheSize);
    }
    else if (requestedSize < (readSize >> 1) && diffSize < readSize) // cachePosition > location
    {
      const size_t target = (size_t)mClamp((int64_t)location - (int64_t)(readSize >> 1), 0LL, (int64_t)cachedFileReader->fileSize - 1);

      if (target == cachedFileReader->cachePosition)
        mRETURN_SUCCESS();

      const size_t move = cachedFileReader->cachePosition - target;
      const size_t moveSize = readSize - move;

      mERROR_CHECK(mMemmove(cachedFileReader->pCache + move, cachedFileReader->pCache, moveSize));

      readSize = move;
      readOffset = 0;
      readPos = target;

      cachedFileReader->cachePosition = target;
      cachedFileReader->cacheSize = mMin(cachedFileReader->fileSize - target, cachedFileReader->maxCacheSize);
    }
    else
    {
      // Read whole block.
      readSize = mMin(cachedFileReader->fileSize - location, cachedFileReader->maxCacheSize);

      cachedFileReader->cachePosition = location;
      cachedFileReader->cacheSize = readSize;
    }
  }

  const size_t readPosition = _lseeki64(cachedFileReader->fileHandle, (int64_t)readPos, SEEK_SET); // Set read position.

  mERROR_IF(readPosition != readPos, mR_IndexOutOfBounds);

  const int bytesRead = _read(cachedFileReader->fileHandle, cachedFileReader->pCache + readOffset, (uint32_t)readSize);

  if ((int64_t)bytesRead != (int64_t)readSize)
  {
    cachedFileReader->cachePosition = (size_t)-1;
    cachedFileReader->cacheSize = 0;

    mRETURN_RESULT(mR_ArgumentOutOfBounds);
  }

  mRETURN_SUCCESS();
}
