#ifndef mMediaFileInputHandler_h__
#define mMediaFileInputHandler_h__

#include "mediaLib.h"

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

mFUNCTION(mMediaFileInputHandler_Create, OUT mPtr<mMediaFileInputHandler> *pInputHandler, IN OPTIONAL mAllocator *pAllocator, const mString &filename, const mMediaFileInputHandler_CreateFlags createFlags);
mFUNCTION(mMediaFileInputHandler_Destroy, IN_OUT mPtr<mMediaFileInputHandler> *pInputHandler);
mFUNCTION(mMediaFileInputHandler_Play, mPtr<mMediaFileInputHandler> &inputHandler);
mFUNCTION(mMediaFileInputHandler_GetVideoStreamResolution, mPtr<mMediaFileInputHandler> &inputHandler, OUT mVec2s *pResolution, const size_t videoStreamIndex = 0);
mFUNCTION(mMediaFileInputHandler_SetVideoCallback, mPtr<mMediaFileInputHandler> &inputHandler, IN ProcessVideoBufferFunction *pCallback);
mFUNCTION(mMediaFileInputHandler_SetAudioCallback, mPtr<mMediaFileInputHandler> &inputHandler, IN ProcessAudioBufferFunction *pCallback);

mFUNCTION(mMediaFileInputHandler_GetVideoStreamCount, mPtr<mMediaFileInputHandler> &inputHandler, OUT size_t *pCount);
mFUNCTION(mMediaFileInputHandler_GetAudioStreamCount, mPtr<mMediaFileInputHandler> &inputHandler, OUT size_t *pCount);
mFUNCTION(mMediaFileInputHandler_GetVideoStreamType, mPtr<mMediaFileInputHandler> &inputHandler, const size_t index, OUT mVideoStreamType *pStreamType);
mFUNCTION(mMediaFileInputHandler_GetAudioStreamType, mPtr<mMediaFileInputHandler> &inputHandler, const size_t index, OUT mAudioStreamType *pStreamType);

struct mMediaFileInputIterator;

mFUNCTION(mMediaFileInputHandler_GetIterator, mPtr<mMediaFileInputHandler> &inputHandler, OUT mPtr<mMediaFileInputIterator> *pIterator, const mMediaMajorType mediaType, const size_t streamIndex = 0);

mFUNCTION(mMediaFileInputIterator_Destroy, IN_OUT mPtr<mMediaFileInputIterator> *pIterator);

// Returns 'mR_EndOfStream' on end of stream.
mFUNCTION(mMediaFileInputIterator_GetNextVideoFrame, mPtr<mMediaFileInputIterator> &iterator, OUT mPtr<mImageBuffer> *pImageBuffer, OUT OPTIONAL mVideoStreamType *pVideoStreamType);

// Returns 'mR_EndOfStream' on end of stream.
mFUNCTION(mMediaFileInputIterator_GetNextAudioFrame, mPtr<mMediaFileInputIterator> &iterator, OUT mPtr<uint8_t> *pData, OUT OPTIONAL mAudioStreamType *pAudioStreamType);

mFUNCTION(mMediaFileInputIterator_SeekTo, mPtr<mMediaFileInputIterator> &iterator, const mTimeStamp &timeStamp);

// Returns 'mR_EndOfStream' on end of stream.
mFUNCTION(mMediaFileInputIterator_SkipFrame, mPtr<mMediaFileInputIterator> &iterator);

struct mAudioSource;

mFUNCTION(mMediaFileInputHandler_GetAudioSource, mPtr<mMediaFileInputHandler> &inputHandler, OUT mPtr<mAudioSource> *pAudioSource, const size_t streamIndex = 0);

#endif // mMediaFileInputHandler_h__
