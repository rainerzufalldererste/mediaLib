// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "default.h"
#include <string>

#pragma comment(lib, "dxva2.lib")
#pragma comment(lib, "evr.lib")
#pragma comment(lib, "mf.lib")
#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfplay.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")

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
};

enum mPixelFormat
{
  mPF_B8G8R8,
  mPF_B8G8R8A8,

  mPixelFormat_Count
};

struct mVideoStreamType : mMediaType
{
  mVec2s resolution;
  size_t stride;
  mPixelFormat pixelFormat;
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

typedef mFUNCTION(ProcessVideoBufferFunction, IN uint8_t *, const mVideoStreamType &);
typedef mFUNCTION(ProcessAudioBufferFunction, IN uint8_t *, const mAudioStreamType &);

mFUNCTION(mMediaFileInputHandler_Create, OUT mPtr<mMediaFileInputHandler> *pPtr, const std::wstring &fileName, const mMediaFileInputHandler_CreateFlags createFlags);
mFUNCTION(mMediaFileInputHandler_Destroy, IN_OUT mPtr<mMediaFileInputHandler> *pPtr);
mFUNCTION(mMediaFileInputHandler_Play, mPtr<mMediaFileInputHandler> &ptr);
mFUNCTION(mMediaFileInputHandler_GetVideoStreamResolution, mPtr<mMediaFileInputHandler> &ptr, OUT mVec2s *pResolution, const size_t videoStreamIndex = 0);
mFUNCTION(mMediaFileInputHandler_SetVideoCallback, mPtr<mMediaFileInputHandler> &ptr, IN ProcessVideoBufferFunction *pCallback);
mFUNCTION(mMediaFileInputHandler_SetAudioCallback, mPtr<mMediaFileInputHandler> &ptr, IN ProcessAudioBufferFunction *pCallback);
