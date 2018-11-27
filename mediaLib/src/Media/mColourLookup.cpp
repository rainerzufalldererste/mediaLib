#include "mColourLookup.h"
#include "mFile.h"
#include "mQueue.h"

mFUNCTION(mColourLookup_Destroy_Internal, IN mColourLookup *pColourLookup);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mColourLookup_CreateFromFile, OUT mPtr<mColourLookup> *pColourLookup, IN mAllocator *pAllocator, const mString &filename)
{
  mFUNCTION_SETUP();

  mERROR_IF(pColourLookup == nullptr, mR_ArgumentNull);

  mDEFER_CALL_ON_ERROR(pColourLookup, mSharedPointer_Destroy);

  mERROR_CHECK(mSharedPointer_Allocate(pColourLookup, pAllocator, (std::function<void (mColourLookup *)>)[](mColourLookup *pData) { mColourLookup_Destroy_Internal(pData); }, 1));

  (*pColourLookup)->pAllocator = pAllocator;

  mERROR_CHECK(mColourLookup_LoadFromFile(*pColourLookup, filename));

  mRETURN_SUCCESS();
}

mFUNCTION(mColourLookup_Destroy, IN_OUT mPtr<mColourLookup> *pColourLookup)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mSharedPointer_Destroy(pColourLookup));

  mRETURN_SUCCESS();
}

mFUNCTION(mColourLookup_LoadFromFile, mPtr<mColourLookup> &colourLookup, const mString &filename)
{
  mFUNCTION_SETUP();

  mERROR_IF(colourLookup == nullptr, mR_ArgumentNull);
  mERROR_IF(filename.hasFailed, mR_InvalidParameter);

  char *fileContents = nullptr;
  mDEFER(mAllocator_FreePtr(&mDefaultTempAllocator, &fileContents));
  size_t length = 0;
  mERROR_CHECK(mFile_ReadRaw(filename, &fileContents, &mDefaultTempAllocator, &length));

  mERROR_IF(length == 0 || fileContents == nullptr, mR_ResourceNotFound);
  
  const char lutSizeName[] = "LUT_3D_SIZE ";
  char *currentChar = fileContents;
  size_t requiredCapacity = 0;

  // Find dimensions.
  while (currentChar < fileContents + length)
  {
    if (strncmp(lutSizeName, currentChar, mARRAYSIZE(lutSizeName) - 1) == 0)
    {
      currentChar += mARRAYSIZE(lutSizeName) - 1;
      const size_t size = atoi(currentChar);
      colourLookup->resolution = mVec3s(size);

      requiredCapacity = colourLookup->resolution.x * colourLookup->resolution.y * colourLookup->resolution.z;

      if (requiredCapacity > colourLookup->capacity)
      {
        mERROR_CHECK(mAllocator_Reallocate(colourLookup->pAllocator, &colourLookup->pData, requiredCapacity));
        colourLookup->capacity = requiredCapacity;
      }

      // Move to next line
      for (; currentChar < fileContents + length; currentChar++)
      {
        if (*currentChar == '\0')
          mRETURN_RESULT(mR_ResourceIncompatible);

        if (*currentChar == '\r' || *currentChar == '\n')
        {
          currentChar++;
          break;
        }
      }

      break;
    }

    ++currentChar;

    if (*currentChar == '\0')
      mRETURN_RESULT(mR_ResourceIncompatible);
  }

  mVec3f *pData = colourLookup->pData;

  for (size_t i = 0; i < requiredCapacity; i++)
  {
    // Move to the next number.
    for (; currentChar < fileContents + length; currentChar++)
    {
      if (*currentChar == '\0')
        mRETURN_RESULT(mR_ResourceIncompatible);

      if (*currentChar >= '-' && *currentChar <= '9' && *currentChar != '/')
        break;
    }

    for (size_t index = 0; index < 3; index++)
    {
      pData->asArray[index] = (float_t)atof(currentChar);

      if (index != 2)
      {
        // Move to next number
        for (; currentChar < fileContents + length; currentChar++)
        {
          if (*currentChar == '\0')
            mRETURN_RESULT(mR_ResourceIncompatible);

          if (*currentChar == ' ')
          {
            currentChar++;
            break;
          }
        }
      }
    }

    // Move to next line
    for (; currentChar < fileContents + length; currentChar++)
    {
      if (*currentChar == '\0')
        mRETURN_RESULT(mR_ResourceIncompatible);

      if (*currentChar == '\r' || *currentChar == '\n')
      {
        currentChar++;
        break;
      }
    }

    pData++;
  }

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mColourLookup_Destroy_Internal, IN mColourLookup *pColourLookup)
{
  mFUNCTION_SETUP();

  mERROR_IF(pColourLookup == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mAllocator_FreePtr(pColourLookup->pAllocator, &pColourLookup->pData));
  
  pColourLookup->capacity = 0;
  pColourLookup->resolution = mVec3s(0);

  mRETURN_SUCCESS();
}
