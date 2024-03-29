#include "mAudioScene.h"

#include "mQueue.h"
#include "mMutex.h"
#include "mProfiler.h"

#pragma warning(push)
#pragma warning(disable: 4200)

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "w1LT4GzeLk8uc+uQKJUQBZlJ7LugcQ2TT8HDS7OAefTje1amkpHKvkvvAYgf7cR4KYjeCeLx5a6WJ62P"
#endif

constexpr size_t mAudioScene_MaxAudioDelay = 1024;
constexpr float_t mAudioScene_SpeedOfSoundInAirAtRoomTemperatureInMetersPerSecond = 343.f;
constexpr size_t mAudioScene_AudioResampleFadeSamples = 8;

struct mSpacialAudioSourceContainer : mSpacialAudioSource
{
  mVec3f position;
  float_t *pSamples; // This is already resampled to the sample rate of the associated mAudioScene.
  mAllocator *pAllocator;
  size_t sampleCount, sampleCapacity;
  bool retrievedNewSamples;
  mAudioSource_PerformanceInfo performanceInfo; // may or may not be unused by consumer.
};

struct mVirtualMicrophoneContainer : mVirtualMicrophone
{
  float_t *pSamples;
  mAllocator *pAllocator;
  size_t sampleCount, sampleCapacity;
  mVec3f position;
  mVec3f forward;
};

struct mAudioScene final : mAudioSource // Can't be inherited from because it has a trailing flex array.
{
  mMutex *pMutex;
  mUniqueContainer<mQueue<mPtr<mSpacialAudioSourceContainer>>> monoInputs;
  mUniqueContainer<mQueue<size_t>> deadAudioSourceIndices;
  mAllocator *pAllocator;
  bool hasBeenUpdatedThisFrame;
  size_t lastConsumedSamples;
  mAudio_ResampleQuality resampleQuality, stretchQuality;
  float_t resampleTimeMs, stretchTimeMs;
  size_t outputIndex;
  mPtr<mVirtualMicrophoneContainer> outputs[];
};

#pragma warning(pop)

static mFUNCTION(mAudioScene_Destroy_Internal, mAudioScene *pAudioSource);
static mFUNCTION(mAudioScene_GetBuffer_Internal, mPtr<mAudioSource> &audioSource, OUT float_t *pBuffer, const size_t bufferLength, const size_t channelIndex, OUT size_t *pBufferCount);
static mFUNCTION(mAudioScene_ReadBuffers_Internal, mAudioScene *pAudioScene, const size_t sampleCount);
static mFUNCTION(mAudioScene_MoveToNextBuffer_Internal, mPtr<mAudioSource> &audioSource, const size_t samples);
static mFUNCTION(mAudioScene_BroadcastDelay_Internal, mPtr<mAudioSource> &audioSource, const size_t samples);
static mFUNCTION(mAudioScene_BroadcastBottleneck_Internal, mPtr<mAudioSource> &audioSource, const float_t referenceTimeMs);
static mFUNCTION(mSpacialAudioSourceContainer_Destroy_Internal, mSpacialAudioSourceContainer *pAudioSource);
static mFUNCTION(mVirtualMicrophoneContainer_Destroy_Internal, mVirtualMicrophoneContainer *pMicrophone);

inline float_t mAudioScene_GetVolumeFromMicFactor(const float_t volumeFactor, mVirtualMicrophoneContainer *pMic)
{
  if (isnan(volumeFactor))
    return 1.f;

  // Old formula produced some weird behaviour with things being behind the microphone.
  //return mPow((volumeFactor * .5f + .5f) * (1.f - pMic->polarPatternRaise) + pMic->polarPatternRaise, pMic->polarPatternWidth);

  return pMic->polarPatternRaise + (.5f - mCos(mPow((volumeFactor * .5f + .5f), pMic->polarPatternWidth) * mPIf) * .5f) * (1.f - pMic->polarPatternRaise);
}

inline float_t mAudioScene_GetOffAxisFilterFactorFromMicFactor(const float_t directionFactor, mVirtualMicrophoneContainer *pMic)
{
  if (isnan(directionFactor))
    return 1.f;

  return (1.f - (directionFactor * .5f + .5f)) * pMic->behindFilterStrength;
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mAudioScene_Create, OUT mPtr<mAudioSource> *pAudioScene, IN mAllocator *pAllocator, const size_t outputCount, const size_t sampleRate /* = 48000 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAudioScene == nullptr, mR_ArgumentNull);

  mAudioScene *pScene = nullptr;
  mDEFER_CALL_ON_ERROR(pAudioScene, mSharedPointer_Destroy);
  mERROR_CHECK((mSharedPointer_AllocateInheritedWithFlexArray<mAudioSource, mAudioScene, decltype(mAudioScene::outputs[0])>(pAudioScene, pAllocator, outputCount, [](mAudioScene *pData) { mAudioScene_Destroy_Internal(pData); }, &pScene)));

  pScene->channelCount = outputCount;
  pScene->sampleRate = sampleRate;
  pScene->volume = 1.f;
  pScene->pAllocator = pAllocator;
  pScene->seekable = false;
  pScene->resampleQuality = mA_RQ_BestQuality;

  pScene->pGetBufferFunc = mAudioScene_GetBuffer_Internal;
  pScene->pMoveToNextBufferFunc = mAudioScene_MoveToNextBuffer_Internal;
  pScene->pBroadcastDelayFunc = mAudioScene_BroadcastDelay_Internal;
  pScene->pBroadcastBottleneckFunc = mAudioScene_BroadcastBottleneck_Internal;

  mERROR_CHECK(mMutex_Create(&pScene->pMutex, pAllocator));
  mERROR_CHECK(mQueue_Create(&pScene->monoInputs, pAllocator));
  mERROR_CHECK(mQueue_Create(&pScene->deadAudioSourceIndices, pAllocator));

  mRETURN_SUCCESS();
}

mFUNCTION(mAudioScene_AddVirtualMicrophone,
  OUT mPtr<mAudioSource> &audioScene,
  OUT OPTIONAL mPtr<mVirtualMicrophone> *pMicrophone,
  const mVec3f position,
  const mVec3f forward,
  OPTIONAL const std::function<mResult(mVirtualMicrophone *pMicrophone, const size_t samples)> &updateFunc,
  const float_t polarPatternWidth,
  const float_t polarPatternRaise,
  const float_t behindFilterStrength)
{
  mFUNCTION_SETUP();

  mERROR_IF(audioScene == nullptr, mR_ArgumentNull);
  mERROR_IF(forward.LengthSquared() < mSmallest<float_t>(), mR_InvalidParameter);
  mERROR_IF(audioScene->pGetBufferFunc != mAudioScene_GetBuffer_Internal, mR_ResourceIncompatible);

  mAudioScene *pAudioScene = static_cast<mAudioScene *>(audioScene.GetPointer());

  mERROR_IF(pAudioScene->outputIndex == pAudioScene->channelCount, mR_ResourceStateInvalid);

  mPtr<mVirtualMicrophoneContainer> virtualMicrophone = nullptr;
  mERROR_CHECK((mSharedPointer_Allocate<mVirtualMicrophoneContainer>(&virtualMicrophone, pAudioScene->pAllocator, [](mVirtualMicrophoneContainer *pData) { mVirtualMicrophoneContainer_Destroy_Internal(pData); }, 1)));

  new (&virtualMicrophone->updateCallback) std::function<mResult(mVirtualMicrophone *pMicrophone, const size_t samples)>(updateFunc);

  virtualMicrophone->pAllocator = pAudioScene->pAllocator;
  virtualMicrophone->behindFilterStrength = behindFilterStrength;
  virtualMicrophone->polarPatternWidth = polarPatternWidth;
  virtualMicrophone->polarPatternRaise = polarPatternRaise;
  virtualMicrophone->position = *const_cast<mVec3f *>(&virtualMicrophone->nextPosition) = position;
  virtualMicrophone->forward = *const_cast<mVec3f *>(&virtualMicrophone->nextForward) = mVec3f(forward).Normalize();

  pAudioScene->outputs[pAudioScene->outputIndex] = virtualMicrophone;
  ++pAudioScene->outputIndex;

  if (pMicrophone != nullptr)
    *pMicrophone = virtualMicrophone;

  mRETURN_SUCCESS();
}

mFUNCTION(mAudioScene_AddSpacialMonoAudioSource,
  OUT mPtr<mAudioSource> &audioScene,
  OUT OPTIONAL mPtr<mSpacialAudioSource> *pSpacialAudioSource,
  IN mAllocator *pAllocator,
  mPtr<mAudioSource> &audioSource,
  const size_t channelIndex,
  const mVec3f position,
  OPTIONAL const std::function<mResult(mSpacialAudioSource *pAudioSource, const size_t samples)> &updateFunc)
{
  mFUNCTION_SETUP();

  mERROR_IF(audioScene == nullptr || audioSource == nullptr, mR_ArgumentNull);
  mERROR_IF(audioSource->channelCount <= channelIndex, mR_InvalidParameter);
  mERROR_IF(audioScene->pGetBufferFunc != mAudioScene_GetBuffer_Internal, mR_ResourceIncompatible);

  mAudioScene *pAudioScene = static_cast<mAudioScene *>(audioScene.GetPointer());

  mPtr<mSpacialAudioSourceContainer> spacialAudioSource = nullptr;
  mERROR_CHECK((mSharedPointer_Allocate<mSpacialAudioSourceContainer>(&spacialAudioSource, pAllocator, [](mSpacialAudioSourceContainer *pData) { mSpacialAudioSourceContainer_Destroy_Internal(pData); }, 1)));

  new (&spacialAudioSource->updateCallback) std::function<mResult(mSpacialAudioSource *pAudioSource, const size_t samples)>(updateFunc);

  spacialAudioSource->pAllocator = pAllocator;
  spacialAudioSource->position = *const_cast<mVec3f *>(&spacialAudioSource->nextPosition) = position;
  spacialAudioSource->channelIndex = channelIndex;
  spacialAudioSource->monoAudioSource = audioSource;

  {
    mERROR_CHECK(mMutex_Lock(pAudioScene->pMutex));
    mDEFER_CALL(pAudioScene->pMutex, mMutex_Unlock);

    mERROR_CHECK(mQueue_PushBack(pAudioScene->monoInputs, &spacialAudioSource));
  }

  spacialAudioSource->monoAudioSource->isBeingConsumed = true;

  if (pSpacialAudioSource != nullptr)
    *pSpacialAudioSource = spacialAudioSource;

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

static mFUNCTION(mAudioScene_Destroy_Internal, mAudioScene *pAudioSource)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAudioSource == nullptr, mR_ArgumentNull);

  for (size_t i = 0; i < pAudioSource->outputIndex; i++)
    mERROR_CHECK(mSharedPointer_Destroy(&pAudioSource->outputs[i]));

  mERROR_CHECK(mQueue_Destroy(&pAudioSource->monoInputs));
  mERROR_CHECK(mQueue_Destroy(&pAudioSource->deadAudioSourceIndices));

  mERROR_CHECK(mMutex_Destroy(&pAudioSource->pMutex));

  mRETURN_SUCCESS();
}

static mFUNCTION(mAudioScene_GetBuffer_Internal, mPtr<mAudioSource> &audioSource, OUT float_t *pBuffer, const size_t bufferLength, const size_t channelIndex, OUT size_t *pBufferCount)
{
  mFUNCTION_SETUP();

  mPROFILE_SCOPED("mAudioScene_GetBuffer_Internal");

  mERROR_IF(audioSource == nullptr || pBuffer == nullptr || pBufferCount == nullptr, mR_ArgumentNull);
  mERROR_IF(channelIndex >= audioSource->channelCount, mR_ArgumentNull);
  mERROR_IF(audioSource->pGetBufferFunc != mAudioScene_GetBuffer_Internal, mR_ResourceIncompatible);

  mAudioScene *pAudioScene = static_cast<mAudioScene *>(audioSource.GetPointer());

  mERROR_CHECK(mMutex_Lock(pAudioScene->pMutex));
  mDEFER_CALL(pAudioScene->pMutex, mMutex_Unlock);
  
  mERROR_IF(pAudioScene->channelCount != pAudioScene->outputIndex, mR_NotInitialized);
  
  if (!pAudioScene->hasBeenUpdatedThisFrame)
  {
    pAudioScene->hasBeenUpdatedThisFrame = true;

    for (auto &_source : pAudioScene->monoInputs->Iterate())
      _source->performanceInfo.processingTimeMs = 0;

    pAudioScene->resampleTimeMs = 0;
    pAudioScene->stretchTimeMs = 0;

    // Update Buffers.
    mERROR_CHECK(mAudioScene_ReadBuffers_Internal(pAudioScene, bufferLength));

    // Allocate Output Buffers.
    for (size_t i = 0; i < pAudioScene->channelCount; i++)
    {
      if (pAudioScene->outputs[i]->sampleCapacity < bufferLength)
      {
        const size_t newCapacity = bufferLength;
        mERROR_CHECK(mAllocator_Reallocate(pAudioScene->pAllocator, &pAudioScene->outputs[i]->pSamples, newCapacity));
        pAudioScene->outputs[i]->sampleCapacity = newCapacity;
      }

      mERROR_CHECK(mZeroMemory(pAudioScene->outputs[i]->pSamples, bufferLength));
    }

    // Generate Output Buffers.
    for (auto &_source : pAudioScene->monoInputs->Iterate())
    {
      for (size_t i = 0; i < pAudioScene->channelCount; i++)
      {
        const mVec3f startDirection = pAudioScene->outputs[i]->position - _source->position;
        const mVec3f endDirection = *const_cast<mVec3f *>(&pAudioScene->outputs[i]->nextPosition) - *const_cast<mVec3f *>(&_source->nextPosition);
        
        const float_t startDist = startDirection.Length();
        const float_t endDist = endDirection.Length();
        
        const float_t samplesPerMeterPerSecond = pAudioScene->sampleRate / mAudioScene_SpeedOfSoundInAirAtRoomTemperatureInMetersPerSecond;
        const size_t startOffset = mMin((size_t)roundf((startDist) * samplesPerMeterPerSecond), mAudioScene_MaxAudioDelay);
        const size_t endOffset = mMin((size_t)roundf((endDist) * samplesPerMeterPerSecond), mAudioScene_MaxAudioDelay);
        
        const float_t startMicFactor = -mVec3f::Dot(mVec3f(startDirection).Normalize(), mVec3f(pAudioScene->outputs[i]->forward).Normalize());
        const float_t endMicFactor = -mVec3f::Dot(mVec3f(endDirection).Normalize(), mVec3f(*const_cast<mVec3f *>(&pAudioScene->outputs[i]->nextForward)).Normalize());
        
        const float_t avgMicFactor = ((startMicFactor + endMicFactor) * .5f);
        const float_t directionalVolume = mAudioScene_GetVolumeFromMicFactor(avgMicFactor, pAudioScene->outputs[i].GetPointer());
        const float_t offAxisFactor = mAudioScene_GetOffAxisFilterFactorFromMicFactor(avgMicFactor, pAudioScene->outputs[i].GetPointer());
        
        const float_t volume = directionalVolume * _source->monoAudioSource->volume / (1.f + ((startDist + endDist) * .5f));
        
        if (_source->sampleCount > startOffset && _source->sampleCount > endOffset)
        {
          if (pAudioScene->outputs[i]->behindFilterStrength < 0.005f || offAxisFactor < 0.01f)
          {
            if (startOffset == endOffset)
            {
              mERROR_CHECK(mAudio_AddWithVolumeFloat(pAudioScene->outputs[i]->pSamples, _source->pSamples + mAudioScene_MaxAudioDelay - startOffset, volume, mMin(bufferLength, _source->sampleCount - startOffset)));
            }
            else
            {
              size_t currentSampleCount = bufferLength + startOffset - endOffset;
              size_t resampledSampleCount = bufferLength;
        
              currentSampleCount = mMin(currentSampleCount, _source->sampleCount - mMax(startOffset, endOffset));
              resampledSampleCount = mMin(resampledSampleCount, _source->sampleCount - mMax(startOffset, endOffset));

              const int64_t stretchTimeMs = mGetCurrentTimeNs();

              mERROR_CHECK(mAudio_AddResampleMonoToMonoWithVolume(pAudioScene->outputs[i]->pSamples, _source->pSamples + mAudioScene_MaxAudioDelay - startOffset, currentSampleCount, resampledSampleCount, volume, pAudioScene->stretchQuality));

              pAudioScene->stretchTimeMs += (mGetCurrentTimeNs() - stretchTimeMs) * 1e-6f;
            }
          }
          else
          {
            if (startOffset == endOffset)
            {
              // This should use a proper HRTF implementation.
              mERROR_CHECK(mAudio_AddWithVolumeFloatVariableLowpass(pAudioScene->outputs[i]->pSamples, _source->pSamples + mAudioScene_MaxAudioDelay - startOffset, volume, mMin(bufferLength, _source->sampleCount - startOffset), offAxisFactor));
            }
            else
            {
              size_t currentSampleCount = bufferLength + startOffset - endOffset;
              size_t resampledSampleCount = bufferLength;
          
              currentSampleCount = mMin(currentSampleCount, _source->sampleCount - mMax(startOffset, endOffset));
              resampledSampleCount = mMin(resampledSampleCount, _source->sampleCount - mMax(startOffset, endOffset));

              const int64_t stretchTimeMs = mGetCurrentTimeNs();

              // This should use a proper HRTF implementation.
              mERROR_CHECK(mAudio_AddResampleMonoToMonoWithVolumeVariableLowpass(pAudioScene->outputs[i]->pSamples, _source->pSamples + mAudioScene_MaxAudioDelay - startOffset, currentSampleCount, resampledSampleCount, volume, offAxisFactor, pAudioScene->stretchQuality));

              pAudioScene->stretchTimeMs += (mGetCurrentTimeNs() - stretchTimeMs) * 1e-6f;
            }
          }
        }
      }
    }

    for (auto &source : pAudioScene->monoInputs->Iterate())
    {
      source->retrievedNewSamples = false;
      source->position = *const_cast<mVec3f *>(&source->nextPosition);
    }

    for (size_t i = 0; i < pAudioScene->channelCount; i++)
    {
      pAudioScene->outputs[i]->forward = *const_cast<mVec3f *>(&pAudioScene->outputs[i]->nextForward);
      pAudioScene->outputs[i]->position = *const_cast<mVec3f *>(&pAudioScene->outputs[i]->nextPosition);
    }

    pAudioScene->lastConsumedSamples = bufferLength;
  }

  mERROR_CHECK(mMemmove(pBuffer, pAudioScene->outputs[channelIndex]->pSamples, bufferLength));
  *pBufferCount = bufferLength;

  mRETURN_SUCCESS();
}

static mFUNCTION(mAudioScene_MoveToNextBuffer_Internal, mPtr<mAudioSource> &audioSource, const size_t /* samples */)
{
  mFUNCTION_SETUP();

  mPROFILE_SCOPED("mAudioScene_MoveToNextBuffer_Internal");

  mERROR_IF(audioSource == nullptr, mR_ArgumentNull);
  mERROR_IF(audioSource->pMoveToNextBufferFunc != mAudioScene_MoveToNextBuffer_Internal, mR_ResourceIncompatible);

  mAudioScene *pAudioScene = static_cast<mAudioScene *>(audioSource.GetPointer());

  mERROR_CHECK(mMutex_Lock(pAudioScene->pMutex));
  mDEFER_CALL(pAudioScene->pMutex, mMutex_Unlock);

  mERROR_IF(pAudioScene->channelCount != pAudioScene->outputIndex, mR_NotInitialized);
  
  pAudioScene->hasBeenUpdatedThisFrame = false;

  // Consume the samples requested last frame.
  for (auto &source : pAudioScene->monoInputs->Iterate())
  {
    if (source->sampleCount < pAudioScene->lastConsumedSamples)
    {
      source->sampleCount = 0;
    }
    else
    {
      mERROR_CHECK(mMemmove(source->pSamples, source->pSamples + pAudioScene->lastConsumedSamples, source->sampleCount - pAudioScene->lastConsumedSamples));
      source->sampleCount -= pAudioScene->lastConsumedSamples;
    }
  }

  mRETURN_SUCCESS();
}

static mFUNCTION(mAudioScene_BroadcastDelay_Internal, mPtr<mAudioSource> &audioSource, const size_t samples)
{
  mFUNCTION_SETUP();

  mERROR_IF(audioSource == nullptr, mR_ArgumentNull);
  mERROR_IF(audioSource->pBroadcastDelayFunc != mAudioScene_BroadcastDelay_Internal, mR_ResourceIncompatible);

  mAudioScene *pAudioScene = static_cast<mAudioScene *>(audioSource.GetPointer());

  mERROR_CHECK(mMutex_Lock(pAudioScene->pMutex));
  mDEFER_CALL(pAudioScene->pMutex, mMutex_Unlock);

  mERROR_IF(pAudioScene->channelCount != pAudioScene->outputIndex, mR_NotInitialized);

  size_t index = (size_t)-1;

  for (auto &_source : pAudioScene->monoInputs->Iterate())
  {
    ++index;

    if (_source->monoAudioSource->pBroadcastDelayFunc == nullptr)
      continue;

    size_t innerIndex = (size_t)-1;
    bool found = false;

    for (auto &_innerSource : pAudioScene->monoInputs->Iterate())
    {
      ++innerIndex;

      if (innerIndex >= index)
        break;

      if (_source->monoAudioSource == _innerSource->monoAudioSource)
      {
        found = true;
        break;
      }
    }

    if (!found)
      mERROR_CHECK(_source->monoAudioSource->pBroadcastDelayFunc(_source->monoAudioSource, samples + _source->sampleCount));
  }

  mRETURN_SUCCESS();
}

static mFUNCTION(mAudioScene_BroadcastBottleneck_Internal, mPtr<mAudioSource> &audioSource, const float_t referenceTimeMs)
{
  mFUNCTION_SETUP();

  mERROR_IF(audioSource == nullptr, mR_ArgumentNull);
  mERROR_IF(audioSource->pBroadcastBottleneckFunc != mAudioScene_BroadcastBottleneck_Internal, mR_ResourceIncompatible);

  mAudioScene *pAudioScene = static_cast<mAudioScene *>(audioSource.GetPointer());

  if (pAudioScene->resampleTimeMs / referenceTimeMs > 0.3f && pAudioScene->resampleQuality < mA_RQ_Linear)
  {
    pAudioScene->resampleQuality = (mAudio_ResampleQuality)(pAudioScene->resampleQuality + 1);

    mPRINT("AudioScene default resampling quality degraded (0x", mFX()(pAudioScene->resampleQuality), ").\n");
  }

  if (pAudioScene->stretchTimeMs / referenceTimeMs > 0.3f && pAudioScene->stretchQuality < mA_RQ_Linear)
  {
    pAudioScene->stretchQuality = (mAudio_ResampleQuality)(pAudioScene->stretchQuality + 1);

    mPRINT("AudioScene default stretch quality degraded (0x", mFX()(pAudioScene->stretchQuality), ").\n");
  }

  for (auto &_item : pAudioScene->monoInputs->Iterate())
  {
    if (_item->performanceInfo.processingTimeMs / referenceTimeMs >= 0.3 && _item->monoAudioSource->pBroadcastBottleneckFunc != nullptr)
    {
      const mResult result = mSILENCE_ERROR(_item->monoAudioSource->pBroadcastBottleneckFunc(_item->monoAudioSource, _item->performanceInfo.processingTimeMs));

      if (mFAILED(result))
        mPRINT_ERROR("Audio Source failed in AudioScene with error code 0x", mFUInt<mFHex>(result), " in pBroadcastBottleneckFunc. (", g_mResult_lastErrorFile, ": ", g_mResult_lastErrorLine, ")");
    }
  }

  mRETURN_SUCCESS();
}

static mFUNCTION(mAudioScene_ReadBuffers_Internal, mAudioScene *pAudioScene, const size_t sampleCount)
{
  mFUNCTION_SETUP();

  mPROFILE_SCOPED("mAudioScene_ReadBuffers_Internal");

  // Remove dead audio sources.
  {
    mERROR_CHECK(mQueue_OrderBy(pAudioScene->deadAudioSourceIndices));

    for (const size_t indexToRemove : pAudioScene->deadAudioSourceIndices->IterateReverse())
    {
      mPtr<mSpacialAudioSourceContainer> removedAudioSource;
      mDEFER_CALL(&removedAudioSource, mSharedPointer_Destroy);

      mERROR_CHECK(mQueue_PopAt(pAudioScene->monoInputs, indexToRemove, &removedAudioSource));
      removedAudioSource->monoAudioSource->hasBeenConsumed = true;
    }

    mERROR_CHECK(mQueue_Clear(pAudioScene->deadAudioSourceIndices));
  }

  size_t index = (size_t)-1;

  for (auto &audioSource : pAudioScene->monoInputs->Iterate())
  {
    ++index;

    if (audioSource == nullptr || audioSource->monoAudioSource->stopPlayback || audioSource->monoAudioSource->hasBeenConsumed)
    {
      bool contained = true;
      mERROR_CHECK(mQueue_Contains(pAudioScene->deadAudioSourceIndices, index, &contained));

      if (!contained)
        mERROR_CHECK(mQueue_PushBack(pAudioScene->deadAudioSourceIndices, index));
      
      continue;
    }

    const size_t requiredSampleCount = (sampleCount + mAudioScene_MaxAudioDelay) * audioSource->monoAudioSource->sampleRate / pAudioScene->sampleRate;

    if (audioSource->sampleCount < requiredSampleCount)
    {
      audioSource->retrievedNewSamples = true;

      if (audioSource->sampleCapacity < audioSource->sampleCount + requiredSampleCount)
      {
        // Create Sufficient Space for both: the resampled and original signal + mAudioScene_MaxAudioDelay.
        const size_t newCapacity = mMax(audioSource->sampleCapacity * 2, mMax(audioSource->sampleCapacity + ((audioSource->sampleCount + requiredSampleCount) - audioSource->sampleCapacity) * 2, audioSource->sampleCapacity + ((audioSource->sampleCount + sampleCount + mAudioScene_MaxAudioDelay) - audioSource->sampleCapacity) * 2));
        mERROR_CHECK(mAllocator_Reallocate(audioSource->pAllocator, &audioSource->pSamples, newCapacity));
        audioSource->sampleCapacity = newCapacity;
      }

      size_t samplesRetrieved = requiredSampleCount;
      const int64_t processingStartNs = mGetCurrentTimeNs();

      const mResult result = mSILENCE_ERROR(audioSource->monoAudioSource->pGetBufferFunc(audioSource->monoAudioSource, audioSource->pSamples + audioSource->sampleCount, requiredSampleCount, audioSource->channelIndex, &samplesRetrieved));

      audioSource->performanceInfo.processingTimeMs += (mGetCurrentTimeNs() - processingStartNs) * 1e-6f;

      if (mFAILED(result))
      {
        if (result != mR_EndOfStream)
          mPRINT_ERROR("Audio Source failed in Audio Scene in pGetBufferFunc with error code 0x", mFUInt<mFHex>(result), ". (", g_mResult_lastErrorFile, ": ", g_mResult_lastErrorLine, ")");

        bool contained = true;
        mERROR_CHECK(mQueue_Contains(pAudioScene->deadAudioSourceIndices, index, &contained));

        if (!contained)
          mERROR_CHECK(mQueue_PushBack(pAudioScene->deadAudioSourceIndices, index));

        continue;
      }

      if (samplesRetrieved != 0)
      {
        if (requiredSampleCount != sampleCount + mAudioScene_MaxAudioDelay)
        {
          const size_t resultingSampleCount = samplesRetrieved * (sampleCount + mAudioScene_MaxAudioDelay) / requiredSampleCount;
          const int64_t resamplingStartTimeNs = mGetCurrentTimeNs();

          mERROR_CHECK(mAudio_InplaceResampleMonoWithFade(audioSource->pSamples + audioSource->sampleCount, samplesRetrieved, resultingSampleCount, mAudioScene_AudioResampleFadeSamples, pAudioScene->resampleQuality));

          pAudioScene->resampleTimeMs += (mGetCurrentTimeNs() - resamplingStartTimeNs) * 1e-6f;
          audioSource->sampleCount += resultingSampleCount;
        }
        else
        {
          audioSource->sampleCount += samplesRetrieved;
        }
      }
    }
  }

  index = (size_t)-1;

  for (auto &audioSource : pAudioScene->monoInputs->Iterate())
  {
    ++index;

    bool removed = true;
    mERROR_CHECK(mQueue_Contains(pAudioScene->deadAudioSourceIndices, index, &removed));

    if (removed)
      continue;

    bool askForNewBuffer = audioSource->retrievedNewSamples;

    if (index > 0 && askForNewBuffer)
    {
      size_t innerIndex = (size_t)-1;

      for (auto &innerAudioSource : pAudioScene->monoInputs->Iterate())
      {
        ++innerIndex;

        if (innerIndex == index)
          break;

        if (audioSource->monoAudioSource == innerAudioSource->monoAudioSource)
        {
          askForNewBuffer = false;
          break;
        }
      }
    }

    const size_t requiredSampleCount = (sampleCount + mAudioScene_MaxAudioDelay) * audioSource->monoAudioSource->sampleRate / pAudioScene->sampleRate;

    if (askForNewBuffer && audioSource->monoAudioSource->pMoveToNextBufferFunc != nullptr)
    {
      const int64_t processingStartNs = mGetCurrentTimeNs();
      const mResult result = mSILENCE_ERROR(audioSource->monoAudioSource->pMoveToNextBufferFunc(audioSource->monoAudioSource, requiredSampleCount));

      audioSource->performanceInfo.processingTimeMs += (mGetCurrentTimeNs() - processingStartNs) * 1e-6f;

      if (mFAILED(result))
      {
        if (result != mR_EndOfStream)
          mPRINT_ERROR("Audio Source failed in Audio Scene in pMoveToNextBufferFunc with error code 0x", mFUInt<mFHex>(result), ". (", g_mResult_lastErrorFile, ": ", g_mResult_lastErrorLine, ")");

        mERROR_CHECK(mQueue_PushBack(pAudioScene->deadAudioSourceIndices, index));
        continue;
      }
    }

    if (audioSource->updateCallback != nullptr && mFAILED(audioSource->updateCallback(audioSource.GetPointer(), requiredSampleCount)))
    {
      mERROR_CHECK(mQueue_PushBack(pAudioScene->deadAudioSourceIndices, index));
      continue;
    }
  }

  mRETURN_SUCCESS();
}

static mFUNCTION(mSpacialAudioSourceContainer_Destroy_Internal, mSpacialAudioSourceContainer *pAudioSource)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAudioSource == nullptr, mR_ArgumentNull);

  pAudioSource->updateCallback.~function();

  mERROR_CHECK(mSharedPointer_Destroy(&pAudioSource->monoAudioSource));

  if (pAudioSource->pSamples != nullptr)
    mERROR_CHECK(mAllocator_FreePtr(pAudioSource->pAllocator, &pAudioSource->pSamples));

  mRETURN_SUCCESS();
}

static mFUNCTION(mVirtualMicrophoneContainer_Destroy_Internal, mVirtualMicrophoneContainer *pMicrophone)
{
  mFUNCTION_SETUP();

  mERROR_IF(pMicrophone == nullptr, mR_ArgumentNull);

  pMicrophone->updateCallback.~function();

  if (pMicrophone->pSamples != nullptr)
    mERROR_CHECK(mAllocator_FreePtr(pMicrophone->pAllocator, &pMicrophone->pSamples));

  mRETURN_SUCCESS();
}
