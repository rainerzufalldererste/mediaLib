#include "mAudioEngine.h"

#include "mThreading.h"
#include "mPool.h"
#include "mQueue.h"
#include "mProfiler.h"

#define DECLSPEC
#include "SDL.h"
#include "SDL_audio.h"
#undef DECLSPEC

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "aGIVr++X05W55RIB3uR2j2dEzsYnRANw7tYm89aUN1Yn0255o0csUlY1oBBwutQhaBbIv2NgHP1eHUQ2"
#endif

constexpr size_t mAudioEngine_BufferNotReadyTries = 3;

struct mAudioEngine
{
  mMutex *pMutex;
  size_t sampleRate;
  size_t channelCount;
  size_t bufferSize;
  mPtr<mPool<mPtr<mAudioSource>>> audioSources;
  mPtr<mQueue<size_t>> unusedAudioSources;
  uint32_t deviceId;
  float_t audioCallbackBuffer[(size_t)mAudioEngine_MaxSupportedChannelCount * (size_t)mAudioEngine_MaxSupportedAudioSourceSamepleRate * (size_t)mAudioEngine_MaxSupportedAudioSourceSamepleRate / (size_t)mAudioEngine_PreferredSampleRate];
  float_t buffer[mAudioEngine_BufferSize * mAudioEngine_MaxSupportedChannelCount];
  mAllocator *pAllocator;
  mThread *pUpdateThread;
  volatile bool bufferReady;
  volatile bool keepRunning;
  volatile float_t masterVolume;
};

static mFUNCTION(mAudioEngine_Destroy_Internal, IN_OUT mAudioEngine *pAudioEngine);
static void SDLCALL mAudioEngine_AudioCallback_Internal(IN void *pUserData, OUT uint8_t *pStream, const int32_t length);
static mFUNCTION(mAudioEngine_ManagedAudioCallback_Internal, IN mAudioEngine *pAudioEngine, OUT float_t *pStream, const size_t length);
static mFUNCTION(mAudioEngine_PrepareNextAudioBuffer_Internal, IN mAudioEngine *pAudioEngine);

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
  want.format = AUDIO_S16;
  want.channels = mAudioEngine_MaxSupportedChannelCount;
  want.samples = mAudioEngine_BufferSize;
  want.callback = mAudioEngine_AudioCallback_Internal;
  want.userdata = pAudioEngine->GetPointer();

  SDL_AudioSpec have;

  const SDL_AudioDeviceID deviceId = SDL_OpenAudioDevice(nullptr, SDL_FALSE, &want, &have, SDL_AUDIO_ALLOW_CHANNELS_CHANGE);

  mERROR_IF(deviceId == 0, mR_NotSupported);
  mERROR_IF(have.samples != mAudioEngine_BufferSize, mR_ResourceIncompatible);
  (*pAudioEngine)->deviceId = deviceId;

  mERROR_CHECK(mPool_Create(&(*pAudioEngine)->audioSources, pAllocator));
  mERROR_CHECK(mMutex_Create(&(*pAudioEngine)->pMutex, pAllocator));
  mERROR_CHECK(mQueue_Create(&(*pAudioEngine)->unusedAudioSources, pAllocator));

  (*pAudioEngine)->bufferSize = have.samples;
  (*pAudioEngine)->channelCount = have.channels;
  (*pAudioEngine)->sampleRate = have.freq;
  (*pAudioEngine)->pAllocator = pAllocator;
  (*pAudioEngine)->keepRunning = true;
  (*pAudioEngine)->masterVolume = 1.f;

  mERROR_CHECK(mThread_Create(&(*pAudioEngine)->pUpdateThread, pAllocator, mAudioEngine_PrepareNextAudioBuffer_Internal, pAudioEngine->GetPointer()));
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

mFUNCTION(mAudioEngine_SetVolume, mPtr<mAudioEngine> &audioEngine, const float_t volume)
{
  mFUNCTION_SETUP();

  mERROR_IF(audioEngine == nullptr, mR_ArgumentNull);

  audioEngine->masterVolume = volume;

  mRETURN_SUCCESS();
}

mFUNCTION(mAudioEngine_GetVolume, mPtr<mAudioEngine> &audioEngine, OUT float_t *pVolume)
{
  mFUNCTION_SETUP();

  mERROR_IF(audioEngine == nullptr, mR_ArgumentNull);

  *pVolume = audioEngine->masterVolume;

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

static void SDLCALL mAudioEngine_AudioCallback_Internal(IN void *pUserData, OUT uint8_t *pStream, const int32_t length)
{
  mAudioEngine *pAudioEngine = (mAudioEngine *)pUserData;

  mFUNCTION_SETUP();

  mPROFILE_SCOPED("mAudioEngine_AudioCallback");

  mERROR_IF_GOTO(length / sizeof(int16_t) != pAudioEngine->bufferSize * pAudioEngine->channelCount, mR_ResourceIncompatible, mSTDRESULT, epilogue);
  
  size_t i = 0;

  for (; i < mAudioEngine_BufferNotReadyTries; i++)
  {
    {
      mERROR_CHECK_GOTO(mMutex_Lock(pAudioEngine->pMutex), mSTDRESULT, epilogue);
      mDEFER_CALL(pAudioEngine->pMutex, mMutex_Unlock);

      if (pAudioEngine->bufferReady)
      {
        mERROR_CHECK_GOTO(mAudio_ConvertFloatToInt16WithDitheringAndFactor(reinterpret_cast<int16_t *>(pStream), pAudioEngine->buffer, pAudioEngine->bufferSize * pAudioEngine->channelCount, pAudioEngine->masterVolume), mSTDRESULT, epilogue);
        pAudioEngine->bufferReady = false;
        break;
      }
    }

    mSleep(1); // Give the PreparationThread sufficient time to acquire the mutex.
  }

#if !defined(GIT_BUILD)
  if (i >= mAudioEngine_BufferNotReadyTries)
    mDebugOut("! [AUDIO_ERROR]  AudioEngine callback could not be retrieved in time. (%" PRIu64 " tries)\n", i);
#endif

epilogue:
  mASSERT_DEBUG(mSUCCEEDED(mSTDRESULT), mFormat("Audio Engine Callback failed with error code ", mFUInt<mFHex>(mSTDRESULT), "."));
}

static mFUNCTION(mAudioEngine_PrepareNextAudioBuffer_Internal, IN mAudioEngine *pAudioEngine)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAudioEngine == nullptr, mR_ArgumentNull);

  while (pAudioEngine->keepRunning)
  {
    bool ready;

    {
      mERROR_CHECK(mMutex_Lock(pAudioEngine->pMutex));
      mDEFER_CALL(pAudioEngine->pMutex, mMutex_Unlock);

      ready = pAudioEngine->bufferReady;

      if (!ready)
      {
        const mResult result = mAudioEngine_ManagedAudioCallback_Internal(pAudioEngine, reinterpret_cast<float_t *>(pAudioEngine->buffer), pAudioEngine->bufferSize * pAudioEngine->channelCount);

        mASSERT_DEBUG(mSUCCEEDED(result), mFormat("Audio Engine Prepare Next Audio Buffer failed with error code ", mFUInt<mFHex>(result), "."));

        pAudioEngine->bufferReady = true;
        continue;
      }
    }
    
    // Broadcast delay.
    {
      mERROR_CHECK(mMutex_Lock(pAudioEngine->pMutex));
      mDEFER_CALL(pAudioEngine->pMutex, mMutex_Unlock);

      for (auto &&_item : pAudioEngine->audioSources->Iterate())
      {
        if ((*_item)->pBroadcastDelayFunc != nullptr)
        {
          const mResult result = (*_item)->pBroadcastDelayFunc((*_item), mAudioEngine_BufferSize);

          mASSERT_DEBUG(mSUCCEEDED(result), mFormat("Failed to broadcast audio delay to audio source with error code ", mFUInt<mFHex>(result), "."));
        }
      }
    }
    
    mSleep(1);
  }

  mRETURN_SUCCESS();
}

static mFUNCTION(mAudioEngine_ManagedAudioCallback_Internal, IN mAudioEngine *pAudioEngine, OUT float_t *pStream, const size_t length)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAudioEngine == nullptr || pStream == nullptr, mR_ArgumentNull);

  mPROFILE_SCOPED("mAudioEngine_ManagedAudioCallback");

  const size_t perChannelLength = length / pAudioEngine->channelCount;

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
    bool continueOuter = false;

    for (size_t channel = 0; channel < mMin(pAudioEngine->channelCount, (*_item)->channelCount); channel++)
    {
      size_t bufferCount;

      const mResult result = mSILENCE_ERROR((*(*_item)->pGetBufferFunc)(*_item, pAudioEngine->audioCallbackBuffer + bufferLength * channel, bufferLength, channel, &bufferCount));

      if (mFAILED(result))
      {
        if (result != mR_EndOfStream)
          mPRINT_ERROR("Audio Source failed with error code 0x", mFUInt<mFHex>(result) ," in pGetBufferFunc. (", g_mResult_lastErrorFile, ": ", g_mResult_lastErrorLine, ")");

        (*_item)->hasBeenConsumed = true;
        mERROR_CHECK(mQueue_PushBack(pAudioEngine->unusedAudioSources, _item.index));
        continueOuter = true;
        continue;
      }
    }

    if (continueOuter)
      continue;

    if (!(*_item)->stopPlayback && (*_item)->pMoveToNextBufferFunc != nullptr)
    {
      const mResult result = mSILENCE_ERROR((*(*_item)->pMoveToNextBufferFunc)(*_item, bufferLength));

      if (mFAILED(result))
      {
        if (result != mR_EndOfStream)
          mPRINT_ERROR("Audio Source failed with error code 0x", mFUInt<mFHex>(result), " in pMoveToNextBufferFunc. (", g_mResult_lastErrorFile, ": ", g_mResult_lastErrorLine, ")");

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

static mFUNCTION(mAudioEngine_Destroy_Internal, IN_OUT mAudioEngine *pAudioEngine)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAudioEngine == nullptr, mR_ArgumentNull);

  pAudioEngine->keepRunning = false;
  mERROR_CHECK(mThread_Join(pAudioEngine->pUpdateThread));
  mERROR_CHECK(mThread_Destroy(&pAudioEngine->pUpdateThread));

  SDL_CloseAudioDevice(pAudioEngine->deviceId);

  mERROR_CHECK(mPool_Destroy(&pAudioEngine->audioSources));
  mERROR_CHECK(mMutex_Destroy(&pAudioEngine->pMutex));

  mERROR_CHECK(mQueue_Destroy(&pAudioEngine->unusedAudioSources));

  mRETURN_SUCCESS();
}
