#include "mTestLib.h"
#include "mFile.h"
#include "mCachedFileReader.h"

mTEST(mCachedFileReader, TestRead)
{
  mTEST_ALLOCATOR_SETUP();

  const size_t fileSize = 1024 * 12;
  const size_t maxCacheSize = 1024;
  const mString filename = "mCachedFileReaderTest.bin";

  size_t *pData = nullptr;
  mDEFER(mAllocator_FreePtr(pAllocator, &pData));
  mTEST_ASSERT_SUCCESS(mAllocator_AllocateZero(pAllocator, &pData, fileSize));

  for (size_t i = 0; i < fileSize; i++)
    pData[i] = i;

  mTEST_ASSERT_SUCCESS(mFile_WriteRaw(filename, pData, fileSize));
  mDEFER(mFile_Delete(filename));

  mPtr<mCachedFileReader> fileReader;
  mDEFER_CALL(&fileReader, mCachedFileReader_Destroy);
  mTEST_ASSERT_SUCCESS(mCachedFileReader_Create(&fileReader, pAllocator, filename, maxCacheSize));

  size_t actualFileSize = 0;
  mTEST_ASSERT_SUCCESS(mCachedFileReader_GetSize(fileReader, &actualFileSize));
  mTEST_ASSERT_EQUAL(actualFileSize, fileSize * sizeof(size_t));

  // Read from various places in the file.
  for (size_t run = 0; run < 2; run++)
  {
    for (size_t i = 0; i < fileSize; i++)
    {
      size_t value;
      mTEST_ASSERT_SUCCESS(mCachedFileReader_ReadAt(fileReader, i * sizeof(size_t), sizeof(size_t), (uint8_t *)&value));
      mTEST_ASSERT_EQUAL(i, value);

      const size_t size = fileReader->cacheSize / sizeof(size_t);
      const size_t offset = fileReader->cachePosition / sizeof(size_t);

      for (size_t index = 0; index < size; index++)
        mTEST_ASSERT_EQUAL(index + offset, ((size_t *)fileReader->pCache)[index]);
    }
  }

  for (size_t run = 0; run < 2; run++)
  {
    for (int64_t i = fileSize - 1; i >= 0; i--)
    {
      size_t value;
      mTEST_ASSERT_SUCCESS(mCachedFileReader_ReadAt(fileReader, i * sizeof(size_t), sizeof(size_t), (uint8_t *)&value));
      mTEST_ASSERT_EQUAL((size_t)i, value);

      const size_t size = fileReader->cacheSize / sizeof(size_t);
      const size_t offset = fileReader->cachePosition / sizeof(size_t);

      for (size_t index = 0; index < size; index++)
        mTEST_ASSERT_EQUAL(index + offset, ((size_t *)fileReader->pCache)[index]);
    }
  }

  for (size_t run = 0; run < 2; run++)
  {
    for (size_t i = 0; i < fileSize; i++)
    {
      size_t *pValue = nullptr;
      mTEST_ASSERT_SUCCESS(mCachedFileReader_PointerAt(fileReader, i * sizeof(size_t), sizeof(size_t), (uint8_t **)&pValue));
      mTEST_ASSERT_EQUAL(i, *pValue);

      const size_t size = fileReader->cacheSize / sizeof(size_t);
      const size_t offset = fileReader->cachePosition / sizeof(size_t);

      for (size_t index = 0; index < size; index++)
        mTEST_ASSERT_EQUAL(index + offset, ((size_t *)fileReader->pCache)[index]);
    }
  }


  for (size_t run = 0; run < 2; run++)
  {
    for (int64_t i = fileSize - 1; i >= 0; i--)
    {
      size_t *pValue = nullptr;
      mTEST_ASSERT_SUCCESS(mCachedFileReader_PointerAt(fileReader, i * sizeof(size_t), sizeof(size_t), (uint8_t **)&pValue));
      mTEST_ASSERT_EQUAL((size_t)i, *pValue);

      const size_t size = fileReader->cacheSize / sizeof(size_t);
      const size_t offset = fileReader->cachePosition / sizeof(size_t);

      for (size_t index = 0; index < size; index++)
        mTEST_ASSERT_EQUAL(index + offset, ((size_t *)fileReader->pCache)[index]);
    }
  }

  uint8_t unusedValue;
  uint8_t *pBuffer = nullptr;

  mPtr<mCachedFileReader> nullFileReader = nullptr;
  mDEFER_CALL(&nullFileReader, mSharedPointer_Destroy);

  mTEST_ASSERT_EQUAL(mR_IndexOutOfBounds, mCachedFileReader_ReadAt(fileReader, fileSize * sizeof(size_t), 1, &unusedValue));
  mTEST_ASSERT_EQUAL(mR_IndexOutOfBounds, mCachedFileReader_PointerAt(fileReader, fileSize * sizeof(size_t), 1, &pBuffer));
  mTEST_ASSERT_EQUAL(mR_ArgumentNull, mCachedFileReader_ReadAt(fileReader, 0, 1, nullptr));
  mTEST_ASSERT_EQUAL(mR_ArgumentNull, mCachedFileReader_ReadAt(nullFileReader, 0, 1, &unusedValue));
  mTEST_ASSERT_EQUAL(mR_ArgumentNull, mCachedFileReader_PointerAt(fileReader, 0, 1, nullptr));
  mTEST_ASSERT_EQUAL(mR_ArgumentNull, mCachedFileReader_PointerAt(nullFileReader, 0, 1, &pBuffer));
  mTEST_ASSERT_EQUAL(mR_ArgumentOutOfBounds, mCachedFileReader_PointerAt(fileReader, 0, maxCacheSize + 1, &pBuffer));

  mTEST_ASSERT_SUCCESS(mMemset(pData, fileSize, 0));
  mTEST_ASSERT_SUCCESS(mCachedFileReader_ReadAt(fileReader, 0, sizeof(size_t) * fileSize, (uint8_t *)pData));

  for (size_t i = 0; i < fileSize; i++)
    mTEST_ASSERT_EQUAL(i, pData[i]);

  mTEST_ASSERT_SUCCESS(mMemset(pData, fileSize, 0));
  mTEST_ASSERT_SUCCESS(mCachedFileReader_ReadAt(fileReader, sizeof(size_t) * maxCacheSize, sizeof(size_t) * (fileSize - maxCacheSize), (uint8_t *)(pData + maxCacheSize)));

  for (size_t i = maxCacheSize; i < fileSize; i++)
    mTEST_ASSERT_EQUAL(i, pData[i]);

  mTEST_ALLOCATOR_ZERO_CHECK();
}
