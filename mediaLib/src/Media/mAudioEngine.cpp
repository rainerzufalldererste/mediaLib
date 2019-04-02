#include "mAudioEngine.h"
#include "mCachedFileReader.h"

struct mAudioSourceWav : mAudioSource
{
  mPtr<mCachedFileReader> fileReader;

  struct
  {
    uint8_t riffHeader[4];
    uint32_t chunkSize; // RIFF Chunk Size.
    uint8_t waveHeader[4];

    // fmt sub chunk.
    uint8_t fmtSubchunkID[4]; // "fmt ".
    uint32_t fmtSubchunkSize; // Size of the fmt chunk.
    uint16_t fmtAudioFormat; // Audio format 1=PCM, 6=mulaw, 7=alaw, 257=IBM Mu-Law, 258=IBM A-Law, 259=ADPCM.
    uint16_t fmtChannelCount;
    uint32_t fmtSampleRate;
    uint32_t fmtBytesPerSec;
    uint16_t fmtBlockAlign; // 2=16-bit mono, 4=16-bit stereo.
    uint16_t fmtBitsPerSample;

    // data sub chunk.
    uint8_t dataSubchunkID[4]; // "data".
    uint32_t dataSubchunkSize; // Data length.
  } riffWaveHeader;

  size_t readPosition;
  size_t startOffset;
};

mFUNCTION(mAudioSourceWav_Destroy_Internal, IN_OUT mAudioSourceWav *pAudioSource);
mFUNCTION(mAudioSourceWav_GetBuffer_Internal, mPtr<mAudioSource> &audioSource, OUT float_t *pBuffer, const size_t bufferLength, const size_t channelIndex);
mFUNCTION(mAudioSourceWav_MoveToNextBuffer_Internal, mPtr<mAudioSource> &audioSource, const size_t samples);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mAudioSourceWav_Create, OUT mPtr<mAudioSource> *pAudioSource, IN mAllocator *pAllocator, const mString &filename)
{
  mFUNCTION_SETUP();

  mAudioSourceWav *pAudioSourceWav = nullptr;
  mERROR_CHECK(mSharedPointer_AllocateInherited(pAudioSource, pAllocator, (std::function<void(mAudioSourceWav *)>)[](mAudioSourceWav *pData){mAudioSourceWav_Destroy_Internal(pData);}, &pAudioSourceWav));

  mDEFER_CALL_ON_ERROR(pAudioSource, mSharedPointer_Destroy);

  pAudioSourceWav->volume = 1.0f;

  static_assert(sizeof(decltype(mAudioSourceWav::riffWaveHeader)) == (sizeof(uint32_t) * 5 + sizeof(uint16_t) * 4 + sizeof(uint8_t) * 4 * 4), "struct packing invalid");

  mERROR_CHECK(mCachedFileReader_Create(&pAudioSourceWav->fileReader, pAllocator, filename, 1024 * 1024 * 4));
  mERROR_CHECK(mCachedFileReader_ReadAt(pAudioSourceWav->fileReader, 0, sizeof(decltype(mAudioSourceWav::riffWaveHeader)), (uint8_t *)&pAudioSourceWav->riffWaveHeader));

  mERROR_IF(pAudioSourceWav->riffWaveHeader.riffHeader[0] != 'R' || pAudioSourceWav->riffWaveHeader.riffHeader[1] != 'I' || pAudioSourceWav->riffWaveHeader.riffHeader[2] != 'F' || pAudioSourceWav->riffWaveHeader.riffHeader[3] != 'F', mR_ResourceInvalid);
  mERROR_IF(pAudioSourceWav->riffWaveHeader.waveHeader[0] != 'W' || pAudioSourceWav->riffWaveHeader.waveHeader[1] != 'A' || pAudioSourceWav->riffWaveHeader.waveHeader[2] != 'V' || pAudioSourceWav->riffWaveHeader.waveHeader[3] != 'E', mR_ResourceInvalid);
  mERROR_IF(pAudioSourceWav->riffWaveHeader.fmtSubchunkID[0] != 'f' || pAudioSourceWav->riffWaveHeader.fmtSubchunkID[1] != 'm' || pAudioSourceWav->riffWaveHeader.fmtSubchunkID[2] != 't' || pAudioSourceWav->riffWaveHeader.fmtSubchunkID[3] != ' ', mR_ResourceInvalid);
  mERROR_IF(pAudioSourceWav->riffWaveHeader.dataSubchunkID[0] != 'd' || pAudioSourceWav->riffWaveHeader.dataSubchunkID[1] != 'a' || pAudioSourceWav->riffWaveHeader.dataSubchunkID[2] != 't' || pAudioSourceWav->riffWaveHeader.dataSubchunkID[3] != 'a', mR_ResourceInvalid);
  mERROR_IF(pAudioSourceWav->riffWaveHeader.fmtBitsPerSample != 16, mR_ResourceIncompatible);
  mERROR_IF(pAudioSourceWav->riffWaveHeader.fmtBlockAlign != 2 && pAudioSourceWav->riffWaveHeader.fmtBlockAlign != 4, mR_ResourceIncompatible);
  mERROR_IF(pAudioSourceWav->riffWaveHeader.fmtChannelCount != 1 && pAudioSourceWav->riffWaveHeader.fmtChannelCount != 2, mR_ResourceIncompatible);

  pAudioSourceWav->readPosition = pAudioSourceWav->startOffset = sizeof(decltype(mAudioSourceWav::riffWaveHeader));
  pAudioSourceWav->sampleRate = (size_t)pAudioSourceWav->riffWaveHeader.fmtSampleRate;
  pAudioSourceWav->channelCount = (size_t)pAudioSourceWav->riffWaveHeader.fmtChannelCount;

  pAudioSourceWav->pGetBufferFunc = mAudioSourceWav_GetBuffer_Internal;
  pAudioSourceWav->pMoveToNextBufferFunc = mAudioSourceWav_MoveToNextBuffer_Internal;

  mRETURN_SUCCESS();
}

mFUNCTION(mAudioSourceWav_Destroy, IN_OUT mPtr<mAudioSource> *pAudioSource)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAudioSource == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Destroy(pAudioSource));

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mAudioSourceWav_Destroy_Internal, IN_OUT mAudioSourceWav *pAudioSource)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAudioSource == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mCachedFileReader_Destroy(&pAudioSource->fileReader));

  mRETURN_SUCCESS();
}

mFUNCTION(mAudioSourceWav_GetBuffer_Internal, mPtr<mAudioSource> &audioSource, OUT float_t *pBuffer, const size_t bufferLength, const size_t channelIndex)
{
  mFUNCTION_SETUP();

  mERROR_IF(audioSource == nullptr || pBuffer == nullptr, mR_ArgumentNull);
  mERROR_IF(channelIndex >= audioSource->channelCount, mR_IndexOutOfBounds);

  mAudioSourceWav *pAudioSourceWav = static_cast<mAudioSourceWav *>(audioSource.GetPointer());

  const size_t readSize = mMin((bufferLength + pAudioSourceWav->riffWaveHeader.fmtBlockAlign) * sizeof(int16_t) * pAudioSourceWav->channelCount, pAudioSourceWav->riffWaveHeader.dataSubchunkSize - (pAudioSourceWav->readPosition - pAudioSourceWav->startOffset));
  const size_t readItems = mMin(bufferLength * sizeof(int16_t) * pAudioSourceWav->channelCount, pAudioSourceWav->riffWaveHeader.dataSubchunkSize - (pAudioSourceWav->readPosition - pAudioSourceWav->startOffset)) / (sizeof(int16_t) * pAudioSourceWav->channelCount);

  int16_t *pData = nullptr;
  mERROR_CHECK(mCachedFileReader_PointerAt(pAudioSourceWav->fileReader, pAudioSourceWav->readPosition, readSize, (uint8_t **)&pData));

  for (size_t i = 0; i < readItems; i++)
    pBuffer[i] = (float_t)pData[i * pAudioSourceWav->channelCount + channelIndex] / (float_t)(INT16_MAX);

  if (readItems < bufferLength)
  {
    mERROR_CHECK(mMemset(pBuffer + readItems, bufferLength - readItems));
    pAudioSourceWav->stopPlayback = true;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mAudioSourceWav_MoveToNextBuffer_Internal, mPtr<mAudioSource> &audioSource, const size_t samples)
{
  mFUNCTION_SETUP();

  mERROR_IF(audioSource == nullptr, mR_ArgumentNull);

  mAudioSourceWav *pAudioSourceWav = static_cast<mAudioSourceWav *>(audioSource.GetPointer());

  pAudioSourceWav->readPosition += ((samples * pAudioSourceWav->channelCount) / pAudioSourceWav->riffWaveHeader.fmtBlockAlign) * pAudioSourceWav->riffWaveHeader.fmtBlockAlign * sizeof(int16_t);

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

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

  mERROR_CHECK(mMutex_Lock(audioEngine->pMutex));
  mDEFER_CALL(audioEngine->pMutex, mMutex_Unlock);

  size_t unusedIndex;
  mERROR_CHECK(mPool_Add(audioEngine->audioSources, std::move(audioSource), &unusedIndex));

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
      const mResult result = (*(*_item)->pGetBufferFunc)(*_item, pAudioEngine->audioCallbackBuffer + bufferLength * channel, bufferLength, channel);

      if (mFAILED(result))
      {
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
        for (size_t i = 0; i < perChannelLength; i++)
          for (size_t channel = 0; channel < pAudioEngine->channelCount; channel++)
            pStream[i * pAudioEngine->channelCount + channel] += pAudioEngine->audioCallbackBuffer[i + bufferLength * channel] * volume;
      }
      else if ((*_item)->channelCount == 1)
      {
        for (size_t channel = 0; channel < pAudioEngine->channelCount; channel++)
          for (size_t i = 0; i < perChannelLength; i++)
            pStream[i * pAudioEngine->channelCount + channel] += pAudioEngine->audioCallbackBuffer[i] * volume;
      }
      else
      {
        mFAIL_DEBUG("This configuration is not supported yet.");
      }
    }
    else if ((*_item)->sampleRate != pAudioEngine->sampleRate)
    {
      // TODO: This is terrible but works for now. -> Implement resampling.

      const float_t sampleRateFactor = (float_t)(*_item)->sampleRate / (float_t)pAudioEngine->sampleRate;

      if ((*_item)->channelCount >= pAudioEngine->channelCount)
      {
        for (size_t channel = 0; channel < pAudioEngine->channelCount; channel++)
          for (size_t i = 0; i < perChannelLength; i++)
            pStream[i * pAudioEngine->channelCount + channel] += pAudioEngine->audioCallbackBuffer[mClamp((size_t)roundf(i * sampleRateFactor), 0ULL, bufferLength) + bufferLength * channel] * volume;
      }
      else if ((*_item)->channelCount == 1)
      {
        for (size_t channel = 0; channel < pAudioEngine->channelCount; channel++)
          for (size_t i = 0; i < perChannelLength; i++)
            pStream[i * pAudioEngine->channelCount + channel] += pAudioEngine->audioCallbackBuffer[mClamp((size_t)roundf(i * sampleRateFactor), 0ULL, bufferLength)] * volume;
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
