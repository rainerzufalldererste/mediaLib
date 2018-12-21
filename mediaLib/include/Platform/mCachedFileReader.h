#ifndef mCachedFileReader_h__
#define mCachedFileReader_h__

#include "mediaLib.h"

struct mCachedFileReader
{
  int fileHandle;
  mAllocator *pAllocator;
  size_t fileSize;

  size_t cachePosition;
  size_t cacheSize;
  uint8_t *pCache;

  size_t maxCacheSize;
};

mFUNCTION(mCachedFileReader_Create, OUT mPtr<mCachedFileReader> *pCachedFileReader, IN mAllocator *pAllocator, const mString &fileName, const size_t maxCacheSize = 1024 * 1024);
mFUNCTION(mCachedFileReader_Destroy, IN_OUT mPtr<mCachedFileReader> *pCachedFileReader);

mFUNCTION(mCachedFileReader_GetSize, mPtr<mCachedFileReader> &cachedFileReader, OUT size_t *pFileSize);
mFUNCTION(mCachedFileReader_ReadAt, mPtr<mCachedFileReader> &cachedFileReader, const size_t location, const size_t size, OUT uint8_t *pBuffer);
mFUNCTION(mCachedFileReader_PointerAt, mPtr<mCachedFileReader> &cachedFileReader, const size_t location, const size_t size, OUT uint8_t **ppBuffer);

#endif // mCachedFileReader_h__
