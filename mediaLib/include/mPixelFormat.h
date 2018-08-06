// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef mPixelFormat_h__
#define mPixelFormat_h__

#include "default.h"

enum mPixelFormat
{
  mPF_B8G8R8,
  mPF_B8G8R8A8,

  mPixelFormat_Count,
};

mFUNCTION(mPixelFormat_HasSubBuffers, const mPixelFormat pixelFormat, OUT bool *pValue);
mFUNCTION(mPixelFormat_IsChromaSubsampled, const mPixelFormat pixelFormat, OUT bool *pValue);
mFUNCTION(mPixelFormat_GetSize, const mPixelFormat pixelFormat, const mVec2s &size, OUT size_t *pBytes);
mFUNCTION(mPixelFormat_UnitSize, const mPixelFormat pixelFormat, OUT size_t *pBytes);

#endif // mPixelFormat_h__
