// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef mMediaFileInputHandler_h__
#define mMediaFileInputHandler_h__

#include "default.h"
#include <string>
#include "mPixelFormat.h"
#include "mImageBuffer.h"
#include "mTimeStamp.h"

struct mMediaFileInputHandler;

enum mMediaMajorType
{
  mMMT_Undefined,
  mMMT_Video,
  mMMT_Audio,

  mMediaMajorType_Count
};

struct mMediaType
{
  size_t streamIndex;
  mMediaMajorType mediaType;
  size_t wmf_streamIndex;
  mTimeStamp timePoint;
};

struct mVideoStreamType : mMediaType
{
  mVec2s resolution;
  size_t stride;
  mPixelFormat pixelFormat;
  double_t frameRate;
};

struct mAudioStreamType : mMediaType
{
  size_t bufferSize;
  size_t samplesPerSecond;
  size_t bitsPerSample;
  size_t channelCount;
};

enum mMediaFileInputHandler_CreateFlags
{
  mMMFIH_CF_VideoEnabled = 1 << 0,
  mMMFIH_CF_AudioEnabled = 1 << 1,

  mMMFIH_CF_AllMediaTypesEnabled = mMMFIH_CF_VideoEnabled | mMMFIH_CF_AudioEnabled,
};

typedef mFUNCTION(ProcessVideoBufferFunction, mPtr<mImageBuffer> &, const mVideoStreamType &);
typedef mFUNCTION(ProcessAudioBufferFunction, IN uint8_t *, const mAudioStreamType &);

mFUNCTION(mMediaFileInputHandler_Create, OUT mPtr<mMediaFileInputHandler> *pPtr, IN OPTIONAL mAllocator *pAllocator, const std::wstring &fileName, const mMediaFileInputHandler_CreateFlags createFlags);
mFUNCTION(mMediaFileInputHandler_Destroy, IN_OUT mPtr<mMediaFileInputHandler> *pPtr);
mFUNCTION(mMediaFileInputHandler_Play, mPtr<mMediaFileInputHandler> &ptr);
mFUNCTION(mMediaFileInputHandler_GetVideoStreamResolution, mPtr<mMediaFileInputHandler> &ptr, OUT mVec2s *pResolution, const size_t videoStreamIndex = 0);
mFUNCTION(mMediaFileInputHandler_SetVideoCallback, mPtr<mMediaFileInputHandler> &ptr, IN ProcessVideoBufferFunction *pCallback);
mFUNCTION(mMediaFileInputHandler_SetAudioCallback, mPtr<mMediaFileInputHandler> &ptr, IN ProcessAudioBufferFunction *pCallback);

mFUNCTION(mMediaFileInputHandler_GetVideoStreamCount, mPtr<mMediaFileInputHandler> &ptr, OUT size_t *pCount);
mFUNCTION(mMediaFileInputHandler_GetAudioStreamCount, mPtr<mMediaFileInputHandler> &ptr, OUT size_t *pCount);
mFUNCTION(mMediaFileInputHandler_GetVideoStreamType, mPtr<mMediaFileInputHandler> &ptr, const size_t index, OUT mVideoStreamType *pStreamType);
mFUNCTION(mMediaFileInputHandler_GetAudioStreamType, mPtr<mMediaFileInputHandler> &ptr, const size_t index, OUT mAudioStreamType *pStreamType);

struct mMediaFileInputIterator;

mFUNCTION(mMediaFileInputHandler_GetIterator, mPtr<mMediaFileInputHandler> &ptr, OUT mPtr<mMediaFileInputIterator> *pIterator, const mMediaMajorType mediaType, const size_t streamIndex = 0);

mFUNCTION(mMediaFileInputIterator_Destroy, IN_OUT mPtr<mMediaFileInputIterator> *pIterator);

// Returns 'mR_EndOfStream' on end of stream.
mFUNCTION(mMediaFileInputIterator_GetNextVideoFrame, mPtr<mMediaFileInputIterator> &iterator, OUT mPtr<mImageBuffer> *pImageBuffer, OUT OPTIONAL mVideoStreamType *pVideoStreamType);

// Returns 'mR_EndOfStream' on end of stream.
mFUNCTION(mMediaFileInputIterator_GetNextAudioFrame, mPtr<mMediaFileInputIterator> &iterator, OUT mPtr<uint8_t> *pData, OUT OPTIONAL mAudioStreamType *pAudioStreamType);

mFUNCTION(mMediaFileInputIterator_SeekTo, mPtr<mMediaFileInputIterator> &iterator, const mTimeStamp &timeStamp);

// Returns 'mR_EndOfStream' on end of stream.
mFUNCTION(mMediaFileInputIterator_SkipFrame, mPtr<mMediaFileInputIterator> &iterator);

#endif // mMediaFileInputHandler_h__
