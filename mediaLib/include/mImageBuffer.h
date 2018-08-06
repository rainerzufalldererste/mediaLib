// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef mImageBuffer_h__
#define mImageBuffer_h__

#include "default.h"
#include "mPixelFormat.h"

struct mImageBuffer
{
  uint8_t *pPixels;
  size_t allocatedSize;
  bool ownedResource;
  mPixelFormat pixelFormat;
  mVec2s currentSize;
  size_t lineStride;
};

mFUNCTION(mImageBuffer_Create, OUT mPtr<mImageBuffer> *pImageBuffer);
mFUNCTION(mImageBuffer_Create, OUT mPtr<mImageBuffer> *pImageBuffer, const mVec2s &size, const mPixelFormat pixelFormat = mPF_B8G8R8A8);
mFUNCTION(mImageBuffer_Create, OUT mPtr<mImageBuffer> *pImageBuffer, IN void *pData, const mVec2s &size, const mPixelFormat pixelFormat = mPF_B8G8R8A8);
mFUNCTION(mImageBuffer_Create, OUT mPtr<mImageBuffer> *pImageBuffer, IN void *pData, const mVec2s &size, const size_t stride, const mPixelFormat pixelFormat = mPF_B8G8R8A8);
mFUNCTION(mImageBuffer_Create, OUT mPtr<mImageBuffer> *pImageBuffer, IN void *pData, const mVec2s &size, const mRectangle2D<size_t> &rect, const mPixelFormat pixelFormat = mPF_B8G8R8A8);
mFUNCTION(mImageBuffer_Destroy, OUT mPtr<mImageBuffer> *pImageBuffer);

#endif // mImageBuffer_h__
