#include "mAudioEngine.h"

#include "SDL.h"
#include "SDL_audio.h"

mFUNCTION(mAudioEngine_Destroy_Internal, IN_OUT mAudioEngine *pAudioEngine);
void SDLCALL mAudioEngine_AudioCallback_Internal(IN void *pUserData, OUT uint8_t *pStream, const int32_t length);
mFUNCTION(mAudioEngine_ManagedAudioCallback_Internal, IN mAudioEngine *pAudioEngine, OUT float_t *pStream, const size_t length);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mAudioEngine_Create, OUT mPtr<mAudioEngine> *pAudioEngine, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAudioEngine == nullptr, mR_ArgumentNull);

  mDEFER_CALL_ON_ERROR(pAudioEngine, mAudioEngine_Destroy);

  mERROR_CHECK(mSharedPointer_Allocate(pAudioEngine, pAllocator, (std::function<void (mAudioEngine *)>)[](mAudioEngine *pData) {mAudioEngine_Destroy_Internal(pData);}, 1));

  mERROR_IF(0 != SDL_Init(SDL_INIT_AUDIO), mR_NotSupported);

  SDL_AudioSpec want = { 0 };
  want.freq = mAudioEngine_PreferredSampleRate;
  want.format = AUDIO_F32;
  want.channels = mAudioEngine_MaxSupportedChannelCount;
  want.samples = mAudioEngine_BufferSize;
  want.callback = mAudioEngine_AudioCallback_Internal;
  want.userdata = pAudioEngine->GetPointer();

  SDL_AudioSpec have;

  const SDL_AudioDeviceID deviceId = SDL_OpenAudioDevice(nullptr, SDL_FALSE, &want, &have, SDL_AUDIO_ALLOW_CHANNELS_CHANGE);

  mERROR_IF(deviceId == 0, mR_NotSupported);
  (*pAudioEngine)->deviceId = deviceId;

  mERROR_CHECK(mPool_Create(&(*pAudioEngine)->audioSources, pAllocator));
  mERROR_CHECK(mMutex_Create(&(*pAudioEngine)->pMutex, pAllocator));
  mERROR_CHECK(mQueue_Create(&(*pAudioEngine)->unusedAudioSources, pAllocator));

  (*pAudioEngine)->bufferSize = have.samples;
  (*pAudioEngine)->channelCount = have.channels;
  (*pAudioEngine)->sampleRate = have.freq;
  (*pAudioEngine)->pAllocator = pAllocator;

  mERROR_CHECK(mAudioEngine_SetPaused(*pAudioEngine, false));

  mRETURN_SUCCESS();
}

mFUNCTION(mAudioEngine_Destroy, IN_OUT mPtr<mAudioEngine> *pAudioEngine)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAudioEngine == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Destroy(pAudioEngine));

  mRETURN_SUCCESS();
}

mFUNCTION(mAudioEngine_SetPaused, mPtr<mAudioEngine> &audioEngine, const bool paused)
{
  mFUNCTION_SETUP();

  mERROR_IF(audioEngine == nullptr, mR_ArgumentNull);

  SDL_PauseAudioDevice(audioEngine->deviceId, paused ? SDL_TRUE : SDL_FALSE);

  mRETURN_SUCCESS();
}

mFUNCTION(mAudioEngine_AddAudioSource, mPtr<mAudioEngine> &audioEngine, mPtr<mAudioSource> &audioSource)
{
  mFUNCTION_SETUP();

  mERROR_IF(audioEngine == nullptr || audioSource == nullptr, mR_ArgumentNull);
  mERROR_IF(audioSource->isBeingConsumed == true, mR_ResourceStateInvalid);

  mPtr<mAudioSource> audioSourceToAdd;

  if (audioSource->sampleRate != audioEngine->sampleRate)
    mERROR_CHECK(mAudioSourceResampler_Create(&audioSourceToAdd, audioEngine->pAllocator, audioSource, audioEngine->sampleRate));
  else
    audioSourceToAdd = audioSource;

  mERROR_CHECK(mMutex_Lock(audioEngine->pMutex));
  mDEFER_CALL(audioEngine->pMutex, mMutex_Unlock);

  audioSource->isBeingConsumed = true;

  size_t unusedIndex;
  mERROR_CHECK(mPool_Add(audioEngine->audioSources, std::move(audioSourceToAdd), &unusedIndex));

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

void SDLCALL mAudioEngine_AudioCallback_Internal(IN void *pUserData, OUT uint8_t *pStream, const int32_t length)
{
  mAudioEngine *pAudioEngine = (mAudioEngine *)pUserData;

  const mResult result = mAudioEngine_ManagedAudioCallback_Internal(pAudioEngine, (float_t *)pStream, length * sizeof(uint8_t) / sizeof(float_t));

  mASSERT_DEBUG(mSUCCEEDED(result), "Audio Engine Callback failed with error code %" PRIx64 ".", (uint64_t)result);
}

mFUNCTION(mAudioEngine_ManagedAudioCallback_Internal, IN mAudioEngine *pAudioEngine, OUT float_t *pStream, const size_t length)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAudioEngine == nullptr || pStream == nullptr, mR_ArgumentNull);

  const size_t perChannelLength = length / pAudioEngine->channelCount;

  mERROR_CHECK(mMutex_Lock(pAudioEngine->pMutex));
  mDEFER_CALL(pAudioEngine->pMutex, mMutex_Unlock);

  mDEFER(mQueue_Clear(pAudioEngine->unusedAudioSources));

  mERROR_CHECK(mMemset(pStream, length));

  for (auto &&_item : pAudioEngine->audioSources->Iterate())
  {
    if ((*_item)->stopPlayback)
    {
      mERROR_CHECK(mQueue_PushBack(pAudioEngine->unusedAudioSources, _item.index));
      continue;
    }

    const size_t bufferLength = pAudioEngine->bufferSize * (*_item)->sampleRate / pAudioEngine->sampleRate;

    for (size_t channel = 0; channel < mMin(pAudioEngine->channelCount, (*_item)->channelCount); channel++)
    {
      size_t bufferCount;

      const mResult result = (*(*_item)->pGetBufferFunc)(*_item, pAudioEngine->audioCallbackBuffer + bufferLength * channel, bufferLength, channel, &bufferCount);

      if (mFAILED(result))
      {
        if (result != mR_EndOfStream)
          mPRINT_ERROR("Audio Source failed with error code 0x%" PRIx64 " in pGetBufferFunc.\n", (uint64_t)result);

        (*_item)->hasBeenConsumed = true;
        mERROR_CHECK(mQueue_PushBack(pAudioEngine->unusedAudioSources, _item.index));
        continue;
      }
    }

    if (!(*_item)->stopPlayback && (*_item)->pMoveToNextBufferFunc != nullptr)
    {
      const mResult result = (*(*_item)->pMoveToNextBufferFunc)(*_item, bufferLength);

      if (mFAILED(result))
      {
        if (result != mR_EndOfStream)
          mPRINT_ERROR("Audio Source failed with error code 0x%" PRIx64 " in pMoveToNextBufferFunc.\n", (uint64_t)result);

        (*_item)->hasBeenConsumed = true;
        mERROR_CHECK(mQueue_PushBack(pAudioEngine->unusedAudioSources, _item.index));
        continue;
      }
    }

    const float_t volume = (*_item)->volume;

    if ((*_item)->sampleRate == pAudioEngine->sampleRate)
    {
      if ((*_item)->channelCount >= pAudioEngine->channelCount)
      {
        for (size_t channel = 0; channel < pAudioEngine->channelCount; channel++)
          mERROR_CHECK(mAudio_AddToInterleavedFromChannelWithVolumeFloat(pStream, &pAudioEngine->audioCallbackBuffer[bufferLength * channel], channel, pAudioEngine->channelCount, bufferLength, volume));
      }
      else if ((*_item)->channelCount == 1)
      {
        mERROR_CHECK(mAudio_AddToInterleavedBufferFromMonoWithVolumeFloat(pStream, pAudioEngine->audioCallbackBuffer, pAudioEngine->channelCount, bufferLength, volume));
      }
      else
      {
        mFAIL_DEBUG("This configuration is not supported yet.");
      }
    }
    else if ((*_item)->sampleRate != pAudioEngine->sampleRate)
    {
      if ((*_item)->channelCount >= pAudioEngine->channelCount)
      {
        for (size_t channel = 0; channel < pAudioEngine->channelCount; channel++)
          mERROR_CHECK(mAudio_AddResampleToInterleavedFromChannelWithVolume(pStream, pAudioEngine->audioCallbackBuffer + channel * bufferLength, channel, pAudioEngine->channelCount, perChannelLength, bufferLength, volume));
      }
      else if ((*_item)->channelCount == 1)
      {
        mERROR_CHECK(mAudio_AddResampleMonoToInterleavedFromChannelWithVolume(pStream, pAudioEngine->audioCallbackBuffer, pAudioEngine->channelCount, perChannelLength, bufferLength, volume));
      }
      else
      {
        mFAIL_DEBUG("This configuration is not supported yet.");
      }
    }
  }

  for (size_t index : pAudioEngine->unusedAudioSources->Iterate())
  {
    mPtr<mAudioSource> source;
    mDEFER_CALL(&source, mSharedPointer_Destroy);
    mERROR_CHECK(mPool_RemoveAt(pAudioEngine->audioSources, index, &source));
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mAudioEngine_Destroy_Internal, IN_OUT mAudioEngine *pAudioEngine)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAudioEngine == nullptr, mR_ArgumentNull);

  SDL_CloseAudioDevice(pAudioEngine->deviceId);

  mERROR_CHECK(mPool_Destroy(&pAudioEngine->audioSources));
  mERROR_CHECK(mMutex_Destroy(&pAudioEngine->pMutex));

  mRETURN_SUCCESS();
}
