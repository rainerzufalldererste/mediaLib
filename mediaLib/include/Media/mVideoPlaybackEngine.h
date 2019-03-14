#ifndef mVideoPlaybackEngine_h__
#define mVideoPlaybackEngine_h__

#include "mediaLib.h"
#include "mImageBuffer.h"

struct mVideoPlaybackEngine;

enum mVideoPlaybackEngine_PlaybackFlags : size_t
{
  mVPE_PF_None = 0,
  mVPE_PF_SeekingInVideoAllowed = 1 << 0,
  mVPE_PF_DroppingFramesOnProducerOverloadEnabled = 1 << 1,
};

mFUNCTION(mVideoPlaybackEngine_Create, OUT mPtr<mVideoPlaybackEngine> *pPlaybackEngine, IN mAllocator *pAllocator, IN const wchar_t *fileName, mPtr<mThreadPool> &threadPool, const size_t videoStreamIndex = 0, const mPixelFormat outputPixelFormat = mPF_B8G8R8A8, const size_t playbackFlags = mVideoPlaybackEngine_PlaybackFlags::mVPE_PF_None);
mFUNCTION(mVideoPlaybackEngine_Destroy, IN_OUT mPtr<mVideoPlaybackEngine> *pPlaybackEngine);

mFUNCTION(mVideoPlaybackEngine_GetCurrentFrame, mPtr<mVideoPlaybackEngine> &playbackEngine, OUT mPtr<mImageBuffer> *pImageBuffer, OUT OPTIONAL bool *pIsNewFrame = nullptr);

#endif // mVideoPlaybackEngine_h__
