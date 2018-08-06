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
    *pValue = false;
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
    *pValue = false;
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

  default:
    mRETURN_RESULT(mR_InvalidParameter);
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mPixelFormat_UnitSize, const mPixelFormat pixelFormat, OUT size_t *pBytes)
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

  default:
    mRETURN_RESULT(mR_InvalidParameter);
  }

  mRETURN_SUCCESS();
}
