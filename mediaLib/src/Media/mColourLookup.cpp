#include "mColourLookup.h"
#include "mFile.h"
#include "mQueue.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "7i1fMiMlihfUmO475fdfoo8P8x+ExWgNW3dx4SdnDn0TEzWzrVfbA8E939cVxWSuz39b+0HyBiB9gcq6"
#endif

static mFUNCTION(mColourLookup_Destroy_Internal, IN mColourLookup *pColourLookup);

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

mFUNCTION(mColourLookup_At, mPtr<mColourLookup> &colourLookup, const mVec3f position, OUT mVec3f *pColour)
{
  mFUNCTION_SETUP();

  mERROR_IF(colourLookup == nullptr || pColour == nullptr, mR_ArgumentNull);

  mVec3f pos = mVec3f(mClamp(position.x, 0.f, 1.f), mClamp(position.y, 0.f, 1.f), mClamp(position.z, 0.f, 1.f)) * (mVec3f(colourLookup->resolution - mVec3s(1)) - mVec3f(mSmallest<float_t>((float_t)mMax(colourLookup->resolution))));

  const float_t fracx = modff(pos.x, &pos.x);
  const float_t fracy = modff(pos.y, &pos.y);
  const float_t fracz = modff(pos.z, &pos.z);

  const size_t x = (size_t)pos.x;
  const size_t y = (size_t)pos.y;
  const size_t z = (size_t)pos.z;

  *pColour = mTriLerp(
    colourLookup->pData[(x + 0) + ((y + 0) + ((z + 0) * colourLookup->resolution.y)) * colourLookup->resolution.x],
    colourLookup->pData[(x + 1) + ((y + 0) + ((z + 0) * colourLookup->resolution.y)) * colourLookup->resolution.x],
    colourLookup->pData[(x + 0) + ((y + 1) + ((z + 0) * colourLookup->resolution.y)) * colourLookup->resolution.x],
    colourLookup->pData[(x + 1) + ((y + 1) + ((z + 0) * colourLookup->resolution.y)) * colourLookup->resolution.x],
    colourLookup->pData[(x + 0) + ((y + 0) + ((z + 1) * colourLookup->resolution.y)) * colourLookup->resolution.x],
    colourLookup->pData[(x + 1) + ((y + 0) + ((z + 1) * colourLookup->resolution.y)) * colourLookup->resolution.x],
    colourLookup->pData[(x + 0) + ((y + 1) + ((z + 1) * colourLookup->resolution.y)) * colourLookup->resolution.x],
    colourLookup->pData[(x + 1) + ((y + 1) + ((z + 1) * colourLookup->resolution.y)) * colourLookup->resolution.x],
    fracx,
    fracy,
    fracz
  );

  mRETURN_SUCCESS();
}

mFUNCTION(mColourLookup_WriteToFile, mPtr<mColourLookup> &colourLookup, const mString &filename)
{
  mFUNCTION_SETUP();

  mERROR_IF(colourLookup == nullptr, mR_ArgumentNull);
  mERROR_IF(filename.hasFailed, mR_InvalidParameter);
  mERROR_IF(colourLookup->resolution.x != colourLookup->resolution.y || colourLookup->resolution.x != colourLookup->resolution.z, mR_InvalidParameter);

  mUniqueContainer<mFileWriter> fileWriter;
  mERROR_CHECK(mFileWriter_Create(&fileWriter, filename));

  const char startText[] = "TITLE \"Generated By Euclideon Holographics\"\r\nLUT_3D_SIZE ";

  mString tempText;
  mERROR_CHECK(mString_Create(&tempText, startText, sizeof(startText), &mDefaultTempAllocator));
  mERROR_CHECK(mFileWriter_Write(fileWriter, tempText.c_str(), tempText.bytes - 1));

  mERROR_CHECK(mString_CreateFormat(&tempText, tempText.pAllocator, "%" PRIu64 "\r\n\r\n", colourLookup->resolution.x));
  mERROR_CHECK(mFileWriter_Write(fileWriter, tempText.c_str(), tempText.bytes - 1));

  for (size_t i = 0; i < colourLookup->resolution.x * colourLookup->resolution.y * colourLookup->resolution.z; i++)
  {
    const mVec3f &colour = colourLookup->pData[i];
    mERROR_CHECK(mString_CreateFormat(&tempText, tempText.pAllocator, "%f %f %f\r\n", colour.x, colour.y, colour.z));
    mERROR_CHECK(mFileWriter_Write(fileWriter, tempText.c_str(), tempText.bytes - 1));
  }

  mERROR_CHECK(mFileWriter_Write(fileWriter, "", 1)); // Null Terminator.

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

static mFUNCTION(mColourLookup_Destroy_Internal, IN mColourLookup *pColourLookup)
{
  mFUNCTION_SETUP();

  mERROR_IF(pColourLookup == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mAllocator_FreePtr(pColourLookup->pAllocator, &pColourLookup->pData));
  
  pColourLookup->capacity = 0;
  pColourLookup->resolution = mVec3s(0);

  mRETURN_SUCCESS();
}
