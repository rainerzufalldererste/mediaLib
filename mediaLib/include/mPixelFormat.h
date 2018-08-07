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

struct mImageBuffer;

enum mPixelFormat
{
  mPF_B8G8R8,
  mPF_B8G8R8A8,
  mPF_YUV422,
  mPF_YUV411,
  mPF_Monochrome8,
  mPF_Monochrome16,

  mPixelFormat_Count,
};

mFUNCTION(mPixelFormat_HasSubBuffers, const mPixelFormat pixelFormat, OUT bool *pValue);
mFUNCTION(mPixelFormat_IsChromaSubsampled, const mPixelFormat pixelFormat, OUT bool *pValue);
mFUNCTION(mPixelFormat_GetSize, const mPixelFormat pixelFormat, const mVec2s &size, OUT size_t *pBytes);
mFUNCTION(mPixelFormat_GetUnitSize, const mPixelFormat pixelFormat, OUT size_t *pBytes);
mFUNCTION(mPixelFormat_GetSubBufferCount, const mPixelFormat pixelFormat, OUT size_t *pBufferCount);
mFUNCTION(mPixelFormat_GetSubBufferOffset, const mPixelFormat pixelFormat, const size_t bufferIndex, const mVec2s &size, OUT size_t *pOffset);
mFUNCTION(mPixelFormat_GetSubBufferSize, const mPixelFormat pixelFormat, const size_t bufferIndex, const mVec2s &size, OUT mVec2s *pSubBufferSize);
mFUNCTION(mPixelFormat_GetSubBufferPixelFormat, const mPixelFormat pixelFormat, const size_t bufferIndex, OUT mPixelFormat *pSubBufferPixelFormat);
mFUNCTION(mPixelFormat_GetSubBufferStride, const mPixelFormat pixelFormat, const size_t bufferIndex, const size_t originalLineStride, OUT size_t *pSubBufferLineStride);
mFUNCTION(mPixelFormat_TransformBuffer, mPtr<mImageBuffer> &source, mPtr<mImageBuffer> &target);

#endif // mPixelFormat_h__
