// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef mMediaFileWriter_h__
#define mMediaFileWriter_h__

#include "default.h"
#include "mImageBuffer.h"

enum mMediaFileVideoFormat
{
  mMFVF_H264,
  mMFVF_H264_ES,
  mMFVF_H263,
  mMFVF_WMV1,
  mMFVF_WMV2,
  mMFVF_WMV3,
  mMFVF_RGB32,
  mMFVF_RGB24,
  mMFVF_YUV420,
  mMFVF_HEVC,
  mMFVF_HEVC_ES,
};

struct mMediaFileInformation
{
  size_t averageBitrate = 25 * 1024;
  mVec2s frameSize = mVec2s(1920, 1080);
  mVec2s frameRateFraction = mVec2s(30, 1);
  mVec2s pixelAspectRatioFraction = mVec2s(1, 1);
  mMediaFileVideoFormat videoFormat = mMFVF_H264;
};

struct mMediaFileWriter;

mFUNCTION(mMediaFileWriter_Create, OUT mPtr<mMediaFileWriter> *pMediaFileWriter, IN mAllocator *pAllocator, const std::wstring &filename, IN mMediaFileInformation *pMediaFileInformation);
mFUNCTION(mMediaFileWriter_Destroy, IN_OUT mPtr<mMediaFileWriter> *pMediaFileWriter);

mFUNCTION(mMediaFileWriter_AppendVideoFrame, mPtr<mMediaFileWriter> &mediaFileWriter, mPtr<mImageBuffer> &imageBuffer);
mFUNCTION(mMediaFileWriter_Finalize, mPtr<mMediaFileWriter> &mediaFileWriter);

#endif // mMediaFileWriter_h__
