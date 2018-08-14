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
  mPtr<mMediaFileInputIterator> iterator;
  mTimeStamp startTimeStamp;
  mTimeStamp updateTimeStamp;
  mTimeStamp frameTime;
  mTimeStamp displayTime;
  mAllocator *pAllocator;
  std::atomic<bool> isRunning;
  size_t videoStreamIndex;
  mPixelFormat targetPixelFormat;
  size_t maxQueuedFrames;
};

mFUNCTION(mTimedVideoPlaybackEngine_Create_Internal, IN mTimedVideoPlaybackEngine *pPlaybackEngine, IN mAllocator *pAllocator, const std::wstring &fileName, mPtr<mThreadPool> &threadPool, const size_t videoStreamIndex, const mPixelFormat outputPixelFormat);
mFUNCTION(mTimedVideoPlaybackEngine_Destroy_Internal, IN_OUT mTimedVideoPlaybackEngine *pPlaybackEngine);

mFUNCTION(mTimedVideoPlaybackEngine_PlaybackThread, mTimedVideoPlaybackEngine *pPlaybackEngine)
{
  mFUNCTION_SETUP();

  mTimeStamp playbackTime;
  mERROR_CHECK(mTimeStamp_FromSeconds(&playbackTime, 0));

  while (pPlaybackEngine->isRunning)
  {
    mTimeStamp now;
    mERROR_CHECK(mTimeStamp_Now(&now));

    size_t count = 0;
    mPtr<mImageBuffer> currentImageBuffer;

    // Get ImageBuffer.
    {
      {
        mERROR_CHECK(mMutex_Lock(pPlaybackEngine->pQueueMutex));
        mDEFER_DESTRUCTION(pPlaybackEngine->pQueueMutex, mMutex_Unlock);

        mERROR_CHECK(mQueue_GetCount(pPlaybackEngine->freeImageBuffers, &count));

        if (count > pPlaybackEngine->maxQueuedFrames)
        {
          mERROR_CHECK(mSleep(1));
          continue;
        }

        if (count > 0)
          mERROR_CHECK(mQueue_PopFront(pPlaybackEngine->freeImageBuffers, &currentImageBuffer));
      }

      if (currentImageBuffer == nullptr)
        mERROR_CHECK(mImageBuffer_Create(&currentImageBuffer, pPlaybackEngine->pAllocator));
    }

    mPtr<mImageBuffer> rawImageBuffer;
    mERROR_CHECK(mMediaFileInputIterator_GetNextVideoFrame(pPlaybackEngine->iterator, &rawImageBuffer, nullptr));

    mERROR_CHECK(mImageBuffer_AllocateBuffer(currentImageBuffer, rawImageBuffer->currentSize, pPlaybackEngine->targetPixelFormat));
    mERROR_CHECK(mPixelFormat_TransformBuffer(rawImageBuffer, currentImageBuffer, pPlaybackEngine->threadPool));

    // Push ImageBuffer to queue.
    {
      mERROR_CHECK(mMutex_Lock(pPlaybackEngine->pQueueMutex));
      mDEFER_DESTRUCTION(pPlaybackEngine->pQueueMutex, mMutex_Unlock);

      mERROR_CHECK(mQueue_PushBack(pPlaybackEngine->imageQueue, &currentImageBuffer));
    }
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

  mERROR_IF(pPlaybackEngine == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Destroy(pPlaybackEngine));
  *pPlaybackEngine = nullptr;

  mRETURN_SUCCESS();
}

mFUNCTION(mTimedVideoPlaybackEngine_GetCurrentFrame, mPtr<mTimedVideoPlaybackEngine> &playbackEngine, OUT mPtr<mImageBuffer> *pImageBuffer)
{
  mFUNCTION_SETUP();

  mERROR_IF(playbackEngine == nullptr || pImageBuffer == nullptr, mR_ArgumentNull);

  if (playbackEngine->pPlaybackThread == nullptr)
  {
    mERROR_CHECK(mTimeStamp_Now(&playbackEngine->startTimeStamp));
    mERROR_CHECK(mTimeStamp_Now(&playbackEngine->updateTimeStamp));
    mERROR_CHECK(mTimeStamp_FromSeconds(&playbackEngine->displayTime, 0));

    playbackEngine->isRunning = true;
    mDEFER_ON_ERROR(playbackEngine->isRunning = false);

    mDEFER_DESTRUCTION_ON_ERROR(&playbackEngine->pPlaybackThread, mSetToNullptr);
    mERROR_CHECK(mThread_Create(&playbackEngine->pPlaybackThread, playbackEngine->pAllocator, mTimedVideoPlaybackEngine_PlaybackThread, playbackEngine.GetPointer()));
    
    mVec2s resolution;
    mERROR_CHECK(mMediaFileInputHandler_GetVideoStreamResolution(playbackEngine->mediaFileInputHandler, &resolution, playbackEngine->videoStreamIndex));
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
      mDEFER_DESTRUCTION(&imageBuffer, mImageBuffer_Destroy);

      while (count > 1)
      {
        if (now - playbackEngine->startTimeStamp > playbackEngine->displayTime)
        {
          mERROR_CHECK(mQueue_PopFront(playbackEngine->imageQueue, &imageBuffer));
          mERROR_CHECK(mQueue_PushBack(playbackEngine->freeImageBuffers, &imageBuffer));
          playbackEngine->displayTime += playbackEngine->frameTime;
        }
        else
        {
          *pImageBuffer = imageBuffer;
          break;
        }
      }

      mERROR_CHECK(mQueue_PeekFront(playbackEngine->imageQueue, pImageBuffer));
    }

  sleep:
    mERROR_CHECK(mSleep(1));
    continue;
  }

  if (playbackEngine->pPlaybackThread->threadState == mThread_ThreadState::mT_TS_Stopped)
  {
    mERROR_IF(mFAILED(playbackEngine->pPlaybackThread->result), playbackEngine->pPlaybackThread->result);
    mRETURN_RESULT(mR_EndOfStream);
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
  pPlaybackEngine->maxQueuedFrames = 8;
  new (&pPlaybackEngine->isRunning) std::atomic<bool>(false);

  mERROR_CHECK(mMediaFileInputHandler_Create(&pPlaybackEngine->mediaFileInputHandler, pAllocator, fileName, mMediaFileInputHandler_CreateFlags::mMMFIH_CF_VideoEnabled));
  mERROR_CHECK(mMediaFileInputHandler_GetIterator(pPlaybackEngine->mediaFileInputHandler, &pPlaybackEngine->iterator, mMediaMajorType::mMMT_Video, videoStreamIndex));

  mVideoStreamType videoStreamType;
  mERROR_CHECK(mMediaFileInputHandler_GetVideoStreamType(pPlaybackEngine->mediaFileInputHandler, videoStreamIndex, &videoStreamType));
  mERROR_CHECK(mTimeStamp_FromSeconds(&pPlaybackEngine->frameTime, videoStreamType.frameRate));

  mERROR_CHECK(mMutex_Create(&pPlaybackEngine->pQueueMutex, pAllocator));
  mERROR_CHECK(mQueue_Create(&pPlaybackEngine->imageQueue, pAllocator));
  mERROR_CHECK(mQueue_Create(&pPlaybackEngine->freeImageBuffers, pAllocator));

  mRETURN_SUCCESS();
}

mFUNCTION(mTimedVideoPlaybackEngine_Destroy_Internal, IN_OUT mTimedVideoPlaybackEngine * pPlaybackEngine)
{
  mFUNCTION_SETUP();

  pPlaybackEngine->isRunning = false;

  mERROR_CHECK(mThread_Join(pPlaybackEngine->pPlaybackThread));
  mERROR_CHECK(mThread_Destroy(&pPlaybackEngine->pPlaybackThread));

  mERROR_CHECK(mThreadPool_Destroy(&pPlaybackEngine->threadPool));
  mERROR_CHECK(mMutex_Destroy(&pPlaybackEngine->pQueueMutex));

  mPtr<mImageBuffer> imageBuffer;
  size_t count = 0;
  mERROR_CHECK(mQueue_GetCount(pPlaybackEngine->imageQueue, &count));
  
  for (size_t i = 0; i < count; ++i)
  {
    mERROR_CHECK(mQueue_PopFront(pPlaybackEngine->imageQueue, &imageBuffer));
    mERROR_CHECK(mImageBuffer_Destroy(&imageBuffer));
  }

  count = 0;
  mERROR_CHECK(mQueue_GetCount(pPlaybackEngine->freeImageBuffers, &count));

  for (size_t i = 0; i < count; ++i)
  {
    mERROR_CHECK(mQueue_PopFront(pPlaybackEngine->freeImageBuffers, &imageBuffer));
    mERROR_CHECK(mImageBuffer_Destroy(&imageBuffer));
  }

  mERROR_CHECK(mMediaFileInputIterator_Destroy(&pPlaybackEngine->iterator));
  mERROR_CHECK(mMediaFileInputHandler_Destroy(&pPlaybackEngine->mediaFileInputHandler));

  pPlaybackEngine->isRunning.~atomic();

  mRETURN_SUCCESS();
}
