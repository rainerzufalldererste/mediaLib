// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "mPixelFormat.h"

mFUNCTION(mPixelFormat_HasSubBuffers, const mPixelFormat pixelFormat, OUT bool *pValue)
{
  mFUNCTION_SETUP();

  mERROR_IF(pValue == nullptr, mR_ArgumentNull);

  switch (pixelFormat)
  {
  case mPF_B8G8R8:
  case mPF_B8G8R8A8:
  case mPF_Monochrome8:
  case mPF_Monochrome16:
    *pValue = false;
    break;

  case mPF_YUV422:
    *pValue = true;
    break;

  default:
    mRETURN_RESULT(mR_InvalidParameter);
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mPixelFormat_IsChromaSubsampled, const mPixelFormat pixelFormat, OUT bool *pValue)
{
  mFUNCTION_SETUP();

  mERROR_IF(pValue == nullptr, mR_ArgumentNull);

  switch (pixelFormat)
  {
  case mPF_B8G8R8:
  case mPF_B8G8R8A8:
  case mPF_Monochrome8:
  case mPF_Monochrome16:
    *pValue = false;
    break;

  case mPF_YUV422:
    *pValue = true;
    break;

  default:
    mRETURN_RESULT(mR_InvalidParameter);
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mPixelFormat_GetSize, const mPixelFormat pixelFormat, const mVec2s &size, OUT size_t *pBytes)
{
  mFUNCTION_SETUP();

  mERROR_IF(pBytes == nullptr, mR_ArgumentNull);

  switch (pixelFormat)
  {
  case mPF_B8G8R8:
    *pBytes = sizeof(uint8_t) * 3 * size.x * size.y;
    break;

  case mPF_B8G8R8A8:
    *pBytes = sizeof(uint32_t) * size.x * size.y;
    break;

  case mPF_YUV422:
    *pBytes = sizeof(uint8_t) * size.x * size.y * 3 / 2;
    break;

  case mPF_Monochrome8:
    *pBytes = sizeof(uint8_t) * size.x * size.y;
    break;

  case mPF_Monochrome16:
    *pBytes = sizeof(uint16_t) * size.x * size.y;
    break;

  default:
    mRETURN_RESULT(mR_InvalidParameter);
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mPixelFormat_GetUnitSize, const mPixelFormat pixelFormat, OUT size_t *pBytes)
{
  mFUNCTION_SETUP();

  mERROR_IF(pBytes == nullptr, mR_ArgumentNull);

  switch (pixelFormat)
  {
  case mPF_B8G8R8:
    *pBytes = sizeof(uint8_t) * 3;
    break;

  case mPF_B8G8R8A8:
    *pBytes = sizeof(uint32_t);
    break;

  case mPF_Monochrome8:
    *pBytes = sizeof(uint8_t);
    break;

  case mPF_Monochrome16:
    *pBytes = sizeof(uint16_t);
    break;

  default:
    mRETURN_RESULT(mR_InvalidParameter);
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mPixelFormat_GetSubBufferCount, const mPixelFormat pixelFormat, OUT size_t *pBufferCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(pBufferCount == nullptr, mR_ArgumentNull);

  switch (pixelFormat)
  {
  case mPF_B8G8R8:
  case mPF_B8G8R8A8:
  case mPF_Monochrome8:
  case mPF_Monochrome16:
    *pBufferCount = 1;
    break;

  case mPF_YUV422:
    *pBufferCount = 3;

  default:
    mRETURN_RESULT(mR_InvalidParameter);
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mPixelFormat_GetSubBufferOffset, const mPixelFormat pixelFormat, const size_t bufferIndex, const mVec2s &size, OUT size_t *pOffset)
{
  mFUNCTION_SETUP();

  mERROR_IF(pOffset == nullptr, mR_ArgumentNull);
  
  size_t subBufferCount = 0;
  mERROR_CHECK(mPixelFormat_GetSubBufferCount(pixelFormat, &subBufferCount));
  mERROR_IF(subBufferCount <= bufferIndex, mR_IndexOutOfBounds);

  *pOffset = 0;

  if (bufferIndex > 0)
  {
    for (size_t i = 0; i <= bufferIndex; i++)
    {
      mPixelFormat subBufferPixelFormat;
      mERROR_CHECK(mPixelFormat_GetSubBufferPixelFormat(pixelFormat, bufferIndex, &subBufferPixelFormat));

      size_t subBufferUnitSize;
      mERROR_CHECK(mPixelFormat_GetUnitSize(subBufferPixelFormat, &subBufferUnitSize));

      mVec2s subBufferSize;
      mERROR_CHECK(mPixelFormat_GetSubBufferSize(pixelFormat, bufferIndex, size, &subBufferSize));

      *pOffset += subBufferSize.x * subBufferSize.y * subBufferUnitSize;
    }
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mPixelFormat_GetSubBufferSize, const mPixelFormat pixelFormat, const size_t bufferIndex, const mVec2s &size, OUT mVec2s *pSubBufferSize)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSubBufferSize == nullptr, mR_ArgumentNull);

  size_t subBufferCount = 0;
  mERROR_CHECK(mPixelFormat_GetSubBufferCount(pixelFormat, &subBufferCount));
  mERROR_IF(subBufferCount <= bufferIndex, mR_IndexOutOfBounds);

  switch (pixelFormat)
  {
  case mPF_YUV422:
    switch (bufferIndex)
    {
    case 0:
      *pSubBufferSize = size;
      break;

    case 1:
    case 2:
      *pSubBufferSize = mVec2s(size.x, size.y / 2);
      break;

    default:
      mRETURN_RESULT(mR_IndexOutOfBounds);
    }
    break;

  case mPF_B8G8R8:
  case mPF_B8G8R8A8:
  case mPF_Monochrome8:
  case mPF_Monochrome16:
    *pSubBufferSize = size;
    break;

  default:
    mRETURN_RESULT(mR_InvalidParameter);
    break;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mPixelFormat_GetSubBufferPixelFormat, const mPixelFormat pixelFormat, const size_t bufferIndex, OUT mPixelFormat *pSubBufferPixelFormat)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSubBufferPixelFormat == nullptr, mR_ArgumentNull);

  size_t subBufferCount = 0;
  mERROR_CHECK(mPixelFormat_GetSubBufferCount(pixelFormat, &subBufferCount));
  mERROR_IF(subBufferCount <= bufferIndex, mR_IndexOutOfBounds);

  switch (pixelFormat)
  {
  case mPF_YUV422:
      *pSubBufferPixelFormat = mPixelFormat::mPF_Monochrome8;
    break;

  case mPF_B8G8R8:
  case mPF_B8G8R8A8:
  case mPF_Monochrome8:
  case mPF_Monochrome16:
    *pSubBufferPixelFormat = pixelFormat;
    break;

  default:
    mRETURN_RESULT(mR_InvalidParameter);
    break;
  }

  mRETURN_SUCCESS();
}
