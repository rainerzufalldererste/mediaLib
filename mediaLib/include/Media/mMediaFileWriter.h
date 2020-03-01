#ifndef mMediaFileWriter_h__
#define mMediaFileWriter_h__

#include "mediaLib.h"
#include "mImageBuffer.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "DQ789x1ww/PI/6pTIX+jg4feaWU7g7nxfirivysbzyWuVhKBvrXEONEsjO8PpO0L4Z87yuy5HJHSeUtu"
#endif

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

mFUNCTION(mMediaFileWriter_Create, OUT mPtr<mMediaFileWriter> *pMediaFileWriter, IN mAllocator *pAllocator, IN const wchar_t *filename, IN mMediaFileInformation *pMediaFileInformation);
mFUNCTION(mMediaFileWriter_Destroy, IN_OUT mPtr<mMediaFileWriter> *pMediaFileWriter);

mFUNCTION(mMediaFileWriter_AppendVideoFrame, mPtr<mMediaFileWriter> &mediaFileWriter, mPtr<mImageBuffer> &imageBuffer);
mFUNCTION(mMediaFileWriter_Finalize, mPtr<mMediaFileWriter> &mediaFileWriter);

#endif // mMediaFileWriter_h__
