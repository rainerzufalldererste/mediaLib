#include "mOpusAudio.h"

#include "ogg/ogg.h"
#include "opusfile.h"
#include "opusenc.h"

#include "mMutex.h"
#include "mProfiler.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "po9hyjmrZ7aFoZtL/D0SAE4T1Sau+hbwufurBll6y3ykX/jK54K8291mJ1C6skm0wzTbWIqM/9oyVSIX"
#endif

constexpr size_t mOpusEncoder_MaxChannelCount = 2;
constexpr size_t mOpusEncoder_ChannelBufferSize = 1024;

static_assert((mOpusEncoder_ChannelBufferSize & (32 - 1)) == 0, "Invalid Channel Buffer Size.");

struct mOpusEncoder
{
  OggOpusComments *pComments;
  OggOpusEnc *pEncoder;
  mPtr<mAudioSource> audioSource;
  bool addedSamples;
};

static mFUNCTION(mOpusEncoder_Destroy_Internal, IN_OUT mOpusEncoder *pEncoder);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mOpusEncoder_Create, OUT mPtr<mOpusEncoder> *pEncoder, IN mAllocator *pAllocator, const mString &filename, mPtr<mAudioSource> &audioSource, const size_t bitrate /* = 320 * 1024 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pEncoder == nullptr || audioSource == nullptr, mR_ArgumentNull);
  mERROR_IF(audioSource->pGetBufferFunc == nullptr || audioSource->pMoveToNextBufferFunc == nullptr, mR_NotInitialized);
  mERROR_IF(filename.bytes <= 1 || filename.hasFailed, mR_InvalidParameter);
  mERROR_IF(audioSource->channelCount > mOpusEncoder_MaxChannelCount || audioSource->channelCount == 0, mR_NotSupported);
  mERROR_IF(audioSource->isBeingConsumed, mR_ResourceStateInvalid);

  mDEFER_CALL_ON_ERROR(pEncoder, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Allocate<mOpusEncoder>(pEncoder, pAllocator, [](mOpusEncoder *pData) { mOpusEncoder_Destroy_Internal(pData); }, 1));

  (*pEncoder)->audioSource = audioSource;

  (*pEncoder)->pComments = ope_comments_create();
  mERROR_IF((*pEncoder)->pComments == nullptr, mR_InternalError);

  int32_t error = 0;

  (*pEncoder)->pEncoder = ope_encoder_create_file(filename.c_str(), (*pEncoder)->pComments, (opus_int32)((*pEncoder)->audioSource->sampleRate), (int32_t)((*pEncoder)->audioSource->channelCount), 0, &error);
  mERROR_IF((*pEncoder)->pEncoder == nullptr || error != OPE_OK, mR_InternalError);

  (*pEncoder)->audioSource->isBeingConsumed = true;

  error = ope_encoder_ctl((*pEncoder)->pEncoder, OPUS_SET_BITRATE((int32_t)bitrate));
  mERROR_IF(error != OPE_OK, mR_InvalidParameter);

  mRETURN_SUCCESS();
}

mFUNCTION(mOpusEncoder_Run, mPtr<mOpusEncoder> &encoder)
{
  mFUNCTION_SETUP();

  mERROR_IF(encoder == nullptr, mR_ArgumentNull);

  bool endOfStream = false;
  size_t bufferCount = mOpusEncoder_ChannelBufferSize;
  mALIGN(32) float_t buffer[mOpusEncoder_ChannelBufferSize * mOpusEncoder_MaxChannelCount];

  while (!encoder->audioSource->stopPlayback && !endOfStream)
  {
    mERROR_CHECK(mZeroMemory(buffer, mARRAYSIZE(buffer)));

    for (size_t channel = 0; channel < mMin(encoder->audioSource->channelCount, mOpusEncoder_MaxChannelCount); channel++)
    {
      float_t channelBuffer[mOpusEncoder_ChannelBufferSize];
      mResult result;

      if (mFAILED(mSILENCE_ERROR(result = encoder->audioSource->pGetBufferFunc(encoder->audioSource, channelBuffer, mARRAYSIZE(channelBuffer), channel, &bufferCount))))
      {
        mERROR_IF(result != mR_EndOfStream, result);
        endOfStream = true;
        break;
      }

      mERROR_CHECK(mAudio_SetInterleavedChannelFloat(buffer, channelBuffer, channel, mMin(encoder->audioSource->channelCount, mOpusEncoder_MaxChannelCount), bufferCount));
    }

    mERROR_CHECK(mAudio_ApplyVolumeFloat(buffer, encoder->audioSource->volume, bufferCount));

    const int error = ope_encoder_write_float(encoder->pEncoder, buffer, (int32_t)bufferCount);

    mERROR_IF(error != OPE_OK, mR_InternalError);

    encoder->addedSamples = true;

    if (endOfStream)
      break;

    if (encoder->audioSource->pMoveToNextBufferFunc != nullptr)
    {
      mResult result;

      if (mFAILED(mSILENCE_ERROR(result = encoder->audioSource->pMoveToNextBufferFunc(encoder->audioSource, mOpusEncoder_ChannelBufferSize))))
      {
        mERROR_IF(result != mR_EndOfStream, result);
        break;
      }
    }
  }

  if (encoder->addedSamples)
  {
    const int error = ope_encoder_drain(encoder->pEncoder);
    mERROR_IF(error != OPE_OK, mR_InternalError);
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mOpusEncoder_Destroy, IN_OUT mPtr<mOpusEncoder> *pEncoder)
{
  mFUNCTION_SETUP();

  mERROR_IF(pEncoder == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Destroy(pEncoder));

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

static mFUNCTION(mOpusEncoder_Destroy_Internal, IN_OUT mOpusEncoder *pEncoder)
{
  mFUNCTION_SETUP();

  mERROR_IF(pEncoder == nullptr, mR_ArgumentNull);

  if (pEncoder->pEncoder != nullptr)
  {
    if (pEncoder->addedSamples)
      /* int error = */ope_encoder_drain(pEncoder->pEncoder);

    ope_encoder_destroy(pEncoder->pEncoder);
    pEncoder->pEncoder = nullptr;
  }

  if (pEncoder->pComments != nullptr)
  {
    ope_comments_destroy(pEncoder->pComments);
    pEncoder->pComments = nullptr;
  }

  mERROR_CHECK(mSharedPointer_Destroy(&pEncoder->audioSource));

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

struct mOpusEncoderPassive
{
  OggOpusComments *pComments;
  OggOpusEnc *pEncoder;
  bool addedSamples;
};

static mFUNCTION(mOpusEncoderPassive_Destroy_Internal, IN_OUT mOpusEncoderPassive *pEncoder);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mOpusEncoderPassive_Create, OUT mPtr<mOpusEncoderPassive> *pEncoder, IN mAllocator *pAllocator, const mString &filename, const size_t channelCount, const size_t sampleRate, const size_t bitrate /* = 320 * 1024 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pEncoder == nullptr, mR_ArgumentNull);
  mERROR_IF(filename.bytes <= 1 || filename.hasFailed, mR_InvalidParameter);

  mDEFER_CALL_ON_ERROR(pEncoder, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Allocate<mOpusEncoderPassive>(pEncoder, pAllocator, [](mOpusEncoderPassive *pData) { mOpusEncoderPassive_Destroy_Internal(pData); }, 1));

  (*pEncoder)->pComments = ope_comments_create();
  mERROR_IF((*pEncoder)->pComments == nullptr, mR_InternalError);

  int32_t error = 0;

  (*pEncoder)->pEncoder = ope_encoder_create_file(filename.c_str(), (*pEncoder)->pComments, (int32_t)sampleRate, (int32_t)channelCount, 0, &error);
  mERROR_IF((*pEncoder)->pEncoder == nullptr || error != OPE_OK, mR_InternalError);

  error = ope_encoder_ctl((*pEncoder)->pEncoder, OPUS_SET_BITRATE((int32_t)bitrate));
  mERROR_IF(error != OPE_OK, mR_InvalidParameter);

  mRETURN_SUCCESS();
}

mFUNCTION(mOpusEncoderPassive_AddSamples, mPtr<mOpusEncoderPassive> &encoder, IN const float_t *pChannelInterleavedData, const size_t sampleCountPerChannel)
{
  mFUNCTION_SETUP();

  mERROR_IF(encoder == nullptr || pChannelInterleavedData == nullptr, mR_ArgumentNull);
  mERROR_IF(sampleCountPerChannel == 0, mR_InvalidParameter);

  size_t samplesRemaining = sampleCountPerChannel;
  size_t samplesConsumed = 0;

  while (samplesRemaining)
  {
    const size_t samplesThisTime = mMin(samplesRemaining, (size_t)1024);

    const int error = ope_encoder_write_float(encoder->pEncoder, pChannelInterleavedData + samplesConsumed, (int32_t)samplesThisTime);
    mERROR_IF(error != OPE_OK, mR_InternalError);

    samplesRemaining -= samplesThisTime;
    samplesConsumed += samplesThisTime;

    encoder->addedSamples = true;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mOpusEncoderPassive_Destroy, IN_OUT mPtr<mOpusEncoderPassive> *pEncoder)
{
  mFUNCTION_SETUP();

  mERROR_IF(pEncoder == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Destroy(pEncoder));

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

static mFUNCTION(mOpusEncoderPassive_Destroy_Internal, IN_OUT mOpusEncoderPassive *pEncoder)
{
  mFUNCTION_SETUP();

  mERROR_IF(pEncoder == nullptr, mR_ArgumentNull);

  if (pEncoder->pEncoder != nullptr)
  {
    if (pEncoder->addedSamples)
      /* int error = */ope_encoder_drain(pEncoder->pEncoder);
    
    ope_encoder_destroy(pEncoder->pEncoder);
    pEncoder->pEncoder = nullptr;
  }

  if (pEncoder->pComments != nullptr)
  {
    ope_comments_destroy(pEncoder->pComments);
    pEncoder->pComments = nullptr;
  }

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

struct mOpusFileAudioSource : mAudioSource
{
  OggOpusFile *pFile;
  mAllocator *pAllocator;
  float_t *pData;
  size_t dataSize, dataCapacity, lastConsumedSamples;
  bool endReached;
  mMutex *pMutex;
  volatile bool paused, nextPaused, startedPlaying;
};

static mFUNCTION(mOpusFileAudioSource_Destroy_Internal, mOpusFileAudioSource *pAudioSource);
static mFUNCTION(mOpusFileAudioSource_GetBuffer_Internal, mPtr<mAudioSource> &audioSource, OUT float_t *pBuffer, const size_t bufferLength, const size_t channelIndex, OUT size_t *pBufferCount);
static mFUNCTION(mOpusFileAudioSource_MoveToNextBuffer_Internal, mPtr<mAudioSource> &audioSource, const size_t samples);
static mFUNCTION(mOpusFileAudioSource_SeekSample_Internal, mPtr<mAudioSource> &audioSource, const size_t sample);
static mFUNCTION(mOpusFileAudioSource_ConsumeBuffer_Internal, IN mOpusFileAudioSource *pAudioSource, const size_t samples);
static mFUNCTION(mOpusFileAudioSource_ReadBuffer_Internal, IN mOpusFileAudioSource *pAudioSource, const size_t samplesToLoad);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mOpusFileAudioSource_Create, OUT mPtr<mAudioSource> *pAudioSource, IN mAllocator *pAllocator, const mString &filename)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAudioSource == nullptr, mR_ArgumentNull);
  mERROR_IF(filename.hasFailed || filename.count <= 1, mR_InvalidParameter);

  mOpusFileAudioSource *pInstance = nullptr;
  mDEFER_CALL_ON_ERROR(pAudioSource, mSharedPointer_Destroy);
  mERROR_CHECK((mSharedPointer_AllocateInherited<mAudioSource, mOpusFileAudioSource>(pAudioSource, pAllocator, [](mOpusFileAudioSource *pData) { mOpusFileAudioSource_Destroy_Internal(pData); }, &pInstance)));

  pInstance->pAllocator = pAllocator;

  int32_t error = 0;

  pInstance->pFile = op_open_file(filename.c_str(), &error);

  mERROR_IF(error != 0 || pInstance->pFile == nullptr, mR_ResourceIncompatible);

  pInstance->seekable = op_seekable(pInstance->pFile) != 0;
  pInstance->channelCount = op_channel_count(pInstance->pFile, 0);
  pInstance->sampleRate = 48000;
  pInstance->volume = 1.f;

  mERROR_CHECK(mMutex_Create(&pInstance->pMutex, pAllocator));

  pInstance->pGetBufferFunc = mOpusFileAudioSource_GetBuffer_Internal;
  pInstance->pMoveToNextBufferFunc = mOpusFileAudioSource_MoveToNextBuffer_Internal;
  
  if (pInstance->seekable)
    pInstance->pSeekSampleFunc = mOpusFileAudioSource_SeekSample_Internal;

  mRETURN_SUCCESS();
}

mFUNCTION(mOpusFileAudioSource_Destroy, OUT mPtr<mAudioSource> *pAudioSource)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAudioSource == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Destroy(pAudioSource));
  
  mRETURN_SUCCESS();
}

mFUNCTION(mOpusFileAudioSource_GetSampleCount, OUT mPtr<mAudioSource> &audioSource, OUT size_t *pSampleCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(audioSource == nullptr || pSampleCount == nullptr, mR_ArgumentNull);

  mOpusFileAudioSource *pAudioSource = static_cast<mOpusFileAudioSource *>(audioSource.GetPointer());

  mERROR_CHECK(mMutex_Lock(pAudioSource->pMutex));
  mDEFER_CALL(pAudioSource->pMutex, mMutex_Unlock);

  const int64_t samples = op_pcm_total(pAudioSource->pFile, 0);
  mERROR_IF(samples < 0, mR_InternalError);

  *pSampleCount = (size_t)samples;

  mRETURN_SUCCESS();
}

mFUNCTION(mOpusFileAudioSource_GetSamplePosition, OUT mPtr<mAudioSource> &audioSource, OUT size_t *pSamplePosition)
{
  mFUNCTION_SETUP();

  mERROR_IF(audioSource == nullptr || pSamplePosition == nullptr, mR_ArgumentNull);

  mOpusFileAudioSource *pAudioSource = static_cast<mOpusFileAudioSource *>(audioSource.GetPointer());

  mERROR_CHECK(mMutex_Lock(pAudioSource->pMutex));
  mDEFER_CALL(pAudioSource->pMutex, mMutex_Unlock);

  const int64_t sampleIndex = op_pcm_tell(pAudioSource->pFile);
  mERROR_IF(sampleIndex < 0, mR_InternalError);

  *pSamplePosition = (size_t)mMax((int64_t)sampleIndex - (int64_t)pAudioSource->dataSize / (int64_t)audioSource->channelCount, (int64_t)0);

  mRETURN_SUCCESS();
}

mFUNCTION(mOpusFileAudioSource_SetPaused, OUT mPtr<mAudioSource> &audioSource, OUT bool paused)
{
  mFUNCTION_SETUP();

  mERROR_IF(audioSource == nullptr, mR_ArgumentNull);

  mOpusFileAudioSource *pAudioSource = static_cast<mOpusFileAudioSource *>(audioSource.GetPointer());

  pAudioSource->nextPaused = paused;

  if (!pAudioSource->startedPlaying)
    pAudioSource->paused = paused;

  mRETURN_SUCCESS();
}

mFUNCTION(mOpusFileAudioSource_IsPaused, OUT mPtr<mAudioSource> &audioSource, OUT bool *pIsPaused)
{
  mFUNCTION_SETUP();

  mERROR_IF(audioSource == nullptr || pIsPaused == nullptr, mR_ArgumentNull);

  mOpusFileAudioSource *pAudioSource = static_cast<mOpusFileAudioSource *>(audioSource.GetPointer());

  *pIsPaused = pAudioSource->paused;

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

static mFUNCTION(mOpusFileAudioSource_Destroy_Internal, mOpusFileAudioSource *pAudioSource)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAudioSource == nullptr, mR_ArgumentNull);

  if (pAudioSource->pMutex != nullptr)
    mERROR_CHECK(mMutex_Lock(pAudioSource->pMutex));

  if (pAudioSource->pFile != nullptr)
  {
    op_free(pAudioSource->pFile);
    pAudioSource->pFile = nullptr;
  }

  if (pAudioSource->pMutex != nullptr)
    mERROR_CHECK(mMutex_Unlock(pAudioSource->pMutex));

  if (pAudioSource->pData != nullptr)
  {
    mERROR_CHECK(mAllocator_FreePtr(pAudioSource->pAllocator, &pAudioSource->pData));
    pAudioSource->dataSize = 0;
  }

  mERROR_CHECK(mMutex_Destroy(&pAudioSource->pMutex));

  mRETURN_SUCCESS();
}

static mFUNCTION(mOpusFileAudioSource_GetBuffer_Internal, mPtr<mAudioSource> &audioSource, OUT float_t *pBuffer, const size_t bufferLength, const size_t channelIndex, OUT size_t *pBufferCount)
{
  mFUNCTION_SETUP();

  mPROFILE_SCOPED("mOpusFileAudioSource_GetBuffer_Internal");

  mERROR_IF(audioSource == nullptr || pBuffer == nullptr || pBufferCount == nullptr, mR_ArgumentNull);
  mERROR_IF(channelIndex >= audioSource->channelCount, mR_InvalidParameter);
  mERROR_IF(audioSource->pGetBufferFunc != mOpusFileAudioSource_GetBuffer_Internal, mR_ResourceIncompatible);

  mOpusFileAudioSource *pAudioSource = static_cast<mOpusFileAudioSource *>(audioSource.GetPointer());

  pAudioSource->startedPlaying = true;

  if (pAudioSource->paused)
  {
    mERROR_CHECK(mZeroMemory(pBuffer, bufferLength));
    *pBufferCount = bufferLength;

    mRETURN_SUCCESS();
  }

  mERROR_CHECK(mMutex_Lock(pAudioSource->pMutex));
  mDEFER_CALL(pAudioSource->pMutex, mMutex_Unlock);

  if (pAudioSource->dataSize < bufferLength * pAudioSource->channelCount)
  {
    const mResult result = mSILENCE_ERROR(mOpusFileAudioSource_ReadBuffer_Internal(pAudioSource, bufferLength * pAudioSource->channelCount - pAudioSource->dataSize));

    if (mFAILED(result))
    {
      mERROR_IF(result != mR_EndOfStream, result);
      pAudioSource->endReached = true;
    }
  }

  *pBufferCount = mMin(bufferLength, pAudioSource->dataSize / pAudioSource->channelCount);

  mERROR_CHECK(mAudio_ExtractFloatChannelFromInterleavedFloat(pBuffer, channelIndex, pAudioSource->pData, pAudioSource->channelCount, *pBufferCount));

  pAudioSource->lastConsumedSamples = *pBufferCount;

  mRETURN_SUCCESS();
}

static mFUNCTION(mOpusFileAudioSource_MoveToNextBuffer_Internal, mPtr<mAudioSource> &audioSource, const size_t samples)
{
  mFUNCTION_SETUP();

  mPROFILE_SCOPED("mOpusFileAudioSource_MoveToNextBuffer_Internal");

  mERROR_IF(audioSource == nullptr, mR_ArgumentNull);
  mERROR_IF(audioSource->pMoveToNextBufferFunc != mOpusFileAudioSource_MoveToNextBuffer_Internal, mR_ResourceIncompatible);

  mOpusFileAudioSource *pAudioSource = static_cast<mOpusFileAudioSource *>(audioSource.GetPointer());

  mERROR_CHECK(mMutex_Lock(pAudioSource->pMutex));
  mDEFER_CALL(pAudioSource->pMutex, mMutex_Unlock);

  pAudioSource->paused = pAudioSource->nextPaused;

  mERROR_IF(pAudioSource->endReached, mR_EndOfStream);

  mERROR_CHECK(mOpusFileAudioSource_ConsumeBuffer_Internal(pAudioSource, pAudioSource->lastConsumedSamples * audioSource->channelCount));

  pAudioSource->lastConsumedSamples = 0;

  if (samples * pAudioSource->channelCount > pAudioSource->dataSize)
  {
    const mResult result = mSILENCE_ERROR(mOpusFileAudioSource_ReadBuffer_Internal(pAudioSource, samples * pAudioSource->channelCount));

    if (mFAILED(result))
    {
      mERROR_IF(result != mR_EndOfStream, result);
      pAudioSource->endReached = true;
    }
  }

  mRETURN_SUCCESS();
}

static mFUNCTION(mOpusFileAudioSource_SeekSample_Internal, mPtr<mAudioSource> &audioSource, const size_t sample)
{
  mFUNCTION_SETUP();

  mPROFILE_SCOPED("mOpusFileAudioSource_SeekSample_Internal");

  mERROR_IF(audioSource == nullptr, mR_ArgumentNull);
  mERROR_IF(audioSource->pSeekSampleFunc != mOpusFileAudioSource_SeekSample_Internal, mR_ResourceIncompatible);

  mOpusFileAudioSource *pAudioSource = static_cast<mOpusFileAudioSource *>(audioSource.GetPointer());

  mERROR_CHECK(mMutex_Lock(pAudioSource->pMutex));
  mDEFER_CALL(pAudioSource->pMutex, mMutex_Unlock);

  mERROR_IF(0 != op_pcm_seek(pAudioSource->pFile, (int64_t)sample), mR_InternalError);
  pAudioSource->dataSize = 0;

  mRETURN_SUCCESS();
}

static mFUNCTION(mOpusFileAudioSource_ConsumeBuffer_Internal, IN mOpusFileAudioSource *pAudioSource, const size_t samples)
{
  mFUNCTION_SETUP();

  mERROR_IF(samples == 0, mR_Success);

  if (pAudioSource->dataSize >= samples)
  {
    pAudioSource->dataSize -= samples;
    mERROR_CHECK(mMemmove(pAudioSource->pData, pAudioSource->pData + samples, pAudioSource->dataSize));
  }
  else
  {
    pAudioSource->dataSize = 0;
  }

  mRETURN_SUCCESS();
}

static mFUNCTION(mOpusFileAudioSource_ReadBuffer_Internal, IN mOpusFileAudioSource *pAudioSource, const size_t samplesToLoad)
{
  mFUNCTION_SETUP();

  mPROFILE_SCOPED("mOpusFileAudioSource_ReadBuffer_Internal");

  mERROR_IF(samplesToLoad == 0, mR_Success);

  if (pAudioSource->dataSize + samplesToLoad > pAudioSource->dataCapacity)
  {
    const size_t newCapacity = mMax(pAudioSource->dataSize * 2, pAudioSource->dataSize + samplesToLoad);
    mERROR_CHECK(mAllocator_Reallocate(pAudioSource->pAllocator, &pAudioSource->pData, newCapacity));
    pAudioSource->dataCapacity = newCapacity;
  }

  size_t samplesRemaining = samplesToLoad;

  while (samplesRemaining > 0)
  {
    int32_t streamIndex = -1;
    int32_t samplesLoaded = -1;

    while (streamIndex != 0)
    {
      samplesLoaded = op_read_float(pAudioSource->pFile, pAudioSource->pData + pAudioSource->dataSize, (int32_t)samplesRemaining, &streamIndex);

      mERROR_IF(streamIndex == -1, mR_InternalError);
    }

    mERROR_IF(samplesLoaded == 0, mR_EndOfStream);
    mERROR_IF(samplesLoaded < 0, mR_InternalError);

    const size_t sampleCount = (size_t)samplesLoaded * pAudioSource->channelCount;

    pAudioSource->dataSize += sampleCount;
    
    if (samplesLoaded >= samplesRemaining)
      break;
    else
      samplesRemaining -= sampleCount;
  }

  mRETURN_SUCCESS();
}
