// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "mTimedVideoPlaybackEngine.h"
#include "mQueue.h"
#include "mMediaFileInputHandler.h"
#include "mTimeStamp.h"

struct mTimedVideoPlaybackEngine
{
  mThread *pPlaybackThread;
  mPtr<mThreadPool> threadPool;
  mPtr<mQueue<mPtr<mImageBuffer>>> imageQueue;
  mPtr<mQueue<mPtr<mImageBuffer>>> freeImageBuffers;
  mMutex *pQueueMutex;
  mPtr<mMediaFileInputHandler> mediaFileInputHandler;
  mTimeStamp startTimeStamp;
  mTimeStamp updateTimeStamp;
  mTimeStamp frameTime;
  mTimeStamp displayTime;
  mAllocator *pAllocator;
  std::atomic<bool> isRunning;
  size_t videoStreamIndex;
  mPixelFormat targetPixelFormat;
};

mFUNCTION(mTimedVideoPlaybackEngine_Create_Internal, IN mTimedVideoPlaybackEngine *pPlaybackEngine, IN mAllocator *pAllocator, const std::wstring &fileName, mPtr<mThreadPool> &threadPool, const size_t videoStreamIndex, const mPixelFormat outputPixelFormat);
mFUNCTION(mTimedVideoPlaybackEngine_Destroy_Internal, IN_OUT mTimedVideoPlaybackEngine *pPlaybackEngine);

mFUNCTION(mTimedVideoPlaybackEngine_PlaybackThread, mTimedVideoPlaybackEngine *pPlaybackEngine)
{
  mFUNCTION_SETUP();

  while (pPlaybackEngine->isRunning)
  {

  }

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mTimedVideoPlaybackEngine_Create, OUT mPtr<mTimedVideoPlaybackEngine> *pPlaybackEngine, IN mAllocator *pAllocator, const std::wstring & fileName, mPtr<mThreadPool> &threadPool, const size_t videoStreamIndex /* = 0 */, const mPixelFormat outputPixelFormat /* = mPF_B8G8R8A8 */)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mSharedPointer_Allocate(pPlaybackEngine, pAllocator, (std::function<void(mTimedVideoPlaybackEngine *)>)[](mTimedVideoPlaybackEngine *pData) { mTimedVideoPlaybackEngine_Destroy_Internal(pData); }, 1));

  mERROR_CHECK(mTimedVideoPlaybackEngine_Create_Internal(pPlaybackEngine->GetPointer(), pAllocator, fileName, threadPool, videoStreamIndex, outputPixelFormat));

  mRETURN_SUCCESS();
}

mFUNCTION(mTimedVideoPlaybackEngine_Destroy, IN_OUT mPtr<mTimedVideoPlaybackEngine> *pPlaybackEngine)
{
  mFUNCTION_SETUP();

  mUnused(pPlaybackEngine);

  mRETURN_SUCCESS();
}

mFUNCTION(mTimedVideoPlaybackEngine_GetCurrentFrame, mPtr<mTimedVideoPlaybackEngine> &playbackEngine, OUT mPtr<mImageBuffer> *pImageBuffer)
{
  mFUNCTION_SETUP();

  if (playbackEngine->pPlaybackThread == nullptr)
  {
    mERROR_CHECK(mTimeStamp_Now(&playbackEngine->startTimeStamp));
    mERROR_CHECK(mTimeStamp_Now(&playbackEngine->updateTimeStamp));
    mERROR_CHECK(mTimeStamp_Now(&playbackEngine->displayTime));

    playbackEngine->isRunning = true;
    mDEFER_ON_ERROR(playbackEngine->isRunning = false);

    mDEFER_DESTRUCTION_ON_ERROR(&playbackEngine->pPlaybackThread, mSetToNullptr);
    mERROR_CHECK(mThread_Create(&playbackEngine->pPlaybackThread, playbackEngine->pAllocator, mTimedVideoPlaybackEngine_PlaybackThread, playbackEngine.GetPointer()));
    
    mVec2s resolution;
    mERROR_CHECK(mMediaFileInputHandler_GetVideoStreamResolution(playbackEngine->mediaFileInputHandler, &resolution, playbackEngine->videoStreamIndex));
  }

  if (playbackEngine->pPlaybackThread->threadState == mThread_ThreadState::mT_TS_Stopped)
  {
    mERROR_IF(mFAILED(playbackEngine->pPlaybackThread->result), playbackEngine->pPlaybackThread->result);
  }

  size_t count = 0;

  while (count == 0)
  {
    {
      mERROR_CHECK(mMutex_Lock(playbackEngine->pQueueMutex));
      mDEFER_DESTRUCTION(playbackEngine->pQueueMutex, mMutex_Unlock);

      mERROR_CHECK(mQueue_GetCount(playbackEngine->imageQueue, &count));

      if (count == 0)
        goto sleep; // unlocks the mutex.

      mTimeStamp now;
      mERROR_CHECK(mTimeStamp_Now(&now));
      mPtr<mImageBuffer> imageBuffer;

      while (count > 1)
      {
        if (now - playbackEngine->startTimeStamp > playbackEngine->displayTime)
        {
          mERROR_CHECK(mQueue_PopFront(playbackEngine->imageQueue, &imageBuffer));
          mERROR_CHECK(mQueue_PushBack(playbackEngine->freeImageBuffers, &imageBuffer));
          playbackEngine->displayTime += playbackEngine->frameTime;
        }
      }

      mUnused(pImageBuffer);
    }

  sleep:
    mERROR_CHECK(mSleep(1));
    continue;
  }

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mTimedVideoPlaybackEngine_Create_Internal, IN mTimedVideoPlaybackEngine *pPlaybackEngine, IN mAllocator *pAllocator, const std::wstring &fileName, mPtr<mThreadPool> &threadPool, const size_t videoStreamIndex, const mPixelFormat outputPixelFormat)
{
  mFUNCTION_SETUP();

  pPlaybackEngine->pAllocator = pAllocator;
  pPlaybackEngine->threadPool = threadPool;
  pPlaybackEngine->videoStreamIndex = videoStreamIndex;
  pPlaybackEngine->targetPixelFormat = outputPixelFormat;
  new (&pPlaybackEngine->isRunning) std::atomic<bool>(false);

  mERROR_CHECK(mMediaFileInputHandler_Create(&pPlaybackEngine->mediaFileInputHandler, pAllocator, fileName, mMediaFileInputHandler_CreateFlags::mMMFIH_CF_VideoEnabled));

  mVideoStreamType videoStreamType;
  mERROR_CHECK(mMediaFileInputHandler_GetVideoStreamType(pPlaybackEngine->mediaFileInputHandler, videoStreamIndex, &videoStreamType));
  mERROR_CHECK(mTimeStamp_FromSeconds(&pPlaybackEngine->frameTime, videoStreamType.frameRate));

  mERROR_CHECK(mMutex_Create(&pPlaybackEngine->pQueueMutex, pAllocator));
  mERROR_CHECK(mQueue_Create(&pPlaybackEngine->imageQueue, pAllocator));
  mERROR_CHECK(mQueue_Create(&pPlaybackEngine->freeImageBuffers, pAllocator));

  mRETURN_SUCCESS();
}
