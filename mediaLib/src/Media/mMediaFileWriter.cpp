#include "mMediaFileWriter.h"
#include <mfapi.h>
#include <mfidl.h>
#include <Mferror.h>
#include <mfreadwrite.h>
#include <intsafe.h>
#include "mMediaFoundation.h"

struct mMediaFileWriter
{
  IMFSinkWriter *pSinkWriter;
  DWORD videoStreamIndex;
  mMediaFileInformation mediaFileInformation;
  LONGLONG lastFrameTimestamp;
  bool finalizeCalled;
};

template <typename T>
static mINLINE void _ReleaseReference(T **pData)
{
  if (pData && *pData)
  {
    (*pData)->Release();
    *pData = nullptr;
  }
}

mFUNCTION(mMediaFileWriter_Create_Internal, OUT mMediaFileWriter *pMediaFileWriter, IN mAllocator *pAllocator, IN const wchar_t *filename, IN mMediaFileInformation *pMediaFileInformation);
mFUNCTION(mMediaFileWriter_Destroy_Internal, IN_OUT mMediaFileWriter *pMediaFileWriter);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mMediaFileWriter_Create, OUT mPtr<mMediaFileWriter> *pMediaFileWriter, IN mAllocator *pAllocator, IN const wchar_t *filename, IN mMediaFileInformation *pMediaFileInformation)
{
  mFUNCTION_SETUP();

  mERROR_IF(pMediaFileWriter == nullptr, mR_ArgumentNull);

  if (*pMediaFileWriter != nullptr)
  {
    mERROR_CHECK(mSharedPointer_Destroy(pMediaFileWriter));
    *pMediaFileWriter = nullptr;
  }

  mMediaFileWriter *pMediaFileWriterRaw = nullptr;
  mDEFER_ON_ERROR(mAllocator_FreePtr(pAllocator, &pMediaFileWriterRaw));
  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &pMediaFileWriterRaw, 1));

  mDEFER_CALL_ON_ERROR(pMediaFileWriter, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Create<mMediaFileWriter>(pMediaFileWriter, pMediaFileWriterRaw, [](mMediaFileWriter *pData) { mMediaFileWriter_Destroy_Internal(pData); }, pAllocator));
  pMediaFileWriterRaw = nullptr; // to not be destroyed on error.

  mERROR_CHECK(mMediaFileWriter_Create_Internal(pMediaFileWriter->GetPointer(), pAllocator, filename, pMediaFileInformation));

  mRETURN_SUCCESS();
}

mFUNCTION(mMediaFileWriter_Destroy, IN_OUT mPtr<mMediaFileWriter> *pMediaFileWriter)
{
  mFUNCTION_SETUP();

  mERROR_IF(pMediaFileWriter == nullptr || *pMediaFileWriter == nullptr, mR_ArgumentNull);
  mERROR_CHECK(mSharedPointer_Destroy(pMediaFileWriter));

  mRETURN_SUCCESS();
}

mFUNCTION(mMediaFileWriter_AppendVideoFrame, mPtr<mMediaFileWriter> &mediaFileWriter, mPtr<mImageBuffer> &imageBuffer)
{
  mFUNCTION_SETUP();

  mERROR_IF(mediaFileWriter == nullptr || imageBuffer == nullptr, mR_ArgumentNull);
  mERROR_IF(imageBuffer->currentSize.x != mediaFileWriter->mediaFileInformation.frameSize.x || imageBuffer->currentSize.y != mediaFileWriter->mediaFileInformation.frameSize.y || imageBuffer->pixelFormat != mPF_B8G8R8A8, mR_InvalidParameter);

  HRESULT hr = S_OK;
  mUnused(hr);

  size_t bufferPixels;
  mERROR_CHECK(mPixelFormat_GetSize(imageBuffer->pixelFormat, imageBuffer->currentSize, &bufferPixels));

  IMFMediaBuffer *pBuffer = nullptr;
  mDEFER_CALL(&pBuffer, _ReleaseReference);
  mERROR_IF(FAILED(hr = MFCreateMemoryBuffer((DWORD)bufferPixels, &pBuffer)), mR_InternalError);

  BYTE *pData;
  mERROR_IF(FAILED(hr = pBuffer->Lock(&pData, nullptr, nullptr)), mR_InternalError);

  size_t pixelFormatUnitSize;
  mERROR_CHECK(mPixelFormat_GetUnitSize(imageBuffer->pixelFormat, &pixelFormatUnitSize));

  mERROR_IF(FAILED(hr = MFCopyImage(pData, (DWORD)(imageBuffer->currentSize.x * pixelFormatUnitSize), imageBuffer->pPixels, (DWORD)(imageBuffer->lineStride * pixelFormatUnitSize), (DWORD)(imageBuffer->currentSize.x * pixelFormatUnitSize), (DWORD)imageBuffer->currentSize.y)), mR_InternalError);

  const size_t targetStride = imageBuffer->currentSize.x;
  const size_t sourceStride = imageBuffer->lineStride;

  for (size_t y = 0; y < imageBuffer->currentSize.y; y++)
  {
    if (mediaFileWriter->mediaFileInformation.videoFormat == mMediaFileVideoFormat::mMFVF_H264 || mediaFileWriter->mediaFileInformation.videoFormat == mMediaFileVideoFormat::mMFVF_H264_ES || mediaFileWriter->mediaFileInformation.videoFormat == mMediaFileVideoFormat::mMFVF_H263 || mediaFileWriter->mediaFileInformation.videoFormat == mMediaFileVideoFormat::mMFVF_HEVC || mediaFileWriter->mediaFileInformation.videoFormat == mMediaFileVideoFormat::mMFVF_HEVC_ES) // Just h.264 is tested to be upside down.
    mERROR_CHECK(mAllocator_Move(imageBuffer->pAllocator, &((uint32_t *)pData)[y * targetStride], &((uint32_t *)imageBuffer->pPixels)[(imageBuffer->currentSize.y - y - 1) * sourceStride], imageBuffer->currentSize.x));
    else
      mERROR_CHECK(mAllocator_Move(imageBuffer->pAllocator, &((uint32_t *)pData)[y * targetStride], &((uint32_t *)imageBuffer->pPixels)[y * sourceStride], imageBuffer->currentSize.x));
  }

  mERROR_IF(FAILED(hr = pBuffer->Unlock()), mR_InternalError);
  mERROR_IF(FAILED(hr = pBuffer->SetCurrentLength((DWORD)bufferPixels)), mR_InternalError);

  IMFSample *pSample = nullptr;
  mDEFER_CALL(&pSample, _ReleaseReference);
  mERROR_IF(FAILED(hr = MFCreateSample(&pSample)), mR_InternalError);
  mERROR_IF(FAILED(hr = pSample->AddBuffer(pBuffer)), mR_InternalError);
  mERROR_IF(FAILED(hr = pSample->SetSampleTime(mediaFileWriter->lastFrameTimestamp)), mR_InternalError);

  const size_t duration = (10 * 1000 * mediaFileWriter->mediaFileInformation.frameRateFraction.x) / mediaFileWriter->mediaFileInformation.frameRateFraction.y;
  mediaFileWriter->lastFrameTimestamp += duration;
  mERROR_IF(mediaFileWriter->lastFrameTimestamp > LONGLONG_MAX, mR_ArgumentOutOfBounds);

  mERROR_IF(duration > LONGLONG_MAX, mR_ArgumentOutOfBounds);
  mERROR_IF(FAILED(hr = pSample->SetSampleDuration((LONGLONG)duration)), mR_InternalError);

  mERROR_IF(FAILED(hr = mediaFileWriter->pSinkWriter->WriteSample(mediaFileWriter->videoStreamIndex, pSample)), mR_InternalError);

  mRETURN_SUCCESS();
}

mFUNCTION(mMediaFileWriter_Finalize, mPtr<mMediaFileWriter> &mediaFileWriter)
{
  mFUNCTION_SETUP();

  mERROR_IF(mediaFileWriter == nullptr, mR_ArgumentNull);

  if (mediaFileWriter->finalizeCalled == false)
  {
    mediaFileWriter->finalizeCalled = true;
    mERROR_IF(FAILED(mediaFileWriter->pSinkWriter->Finalize()), mR_InternalError);
  }
  else
  {
    mRETURN_RESULT(mR_ResourceStateInvalid);
  }

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mMediaFileWriter_Create_Internal, OUT mMediaFileWriter *pMediaFileWriter, IN mAllocator *pAllocator, IN const wchar_t *filename, IN mMediaFileInformation *pMediaFileInformation)
{
  mFUNCTION_SETUP();

  mERROR_IF(pMediaFileWriter == nullptr || pMediaFileInformation == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mMediaFoundation_AddReference());

  mERROR_IF(pMediaFileInformation->frameSize.x > UINT32_MAX || pMediaFileInformation->frameSize.y > UINT32_MAX || pMediaFileInformation->averageBitrate > UINT32_MAX || pMediaFileInformation->frameRateFraction.x > UINT32_MAX || pMediaFileInformation->frameRateFraction.y > UINT32_MAX || pMediaFileInformation->pixelAspectRatioFraction.x > UINT32_MAX || pMediaFileInformation->pixelAspectRatioFraction.y > UINT32_MAX, mR_ArgumentOutOfBounds);

  pMediaFileWriter->mediaFileInformation = *pMediaFileInformation;
  pMediaFileWriter->lastFrameTimestamp = 0;
  pMediaFileWriter->finalizeCalled = false;
  mUnused(pAllocator);

  HRESULT hr = S_OK;
  mUnused(hr);

  mERROR_IF(FAILED(hr = MFCreateSinkWriterFromURL(filename, nullptr, nullptr, &pMediaFileWriter->pSinkWriter)), mR_InternalError);

  IMFMediaType *pVideoOutputMediaType;
  mDEFER_CALL(&pVideoOutputMediaType, _ReleaseReference);
  mERROR_IF(FAILED(hr = MFCreateMediaType(&pVideoOutputMediaType)), mR_InternalError);

  GUID subtype = MFVideoFormat_H264;

  switch (pMediaFileInformation->videoFormat)
  {
  case mMFVF_H264:
    subtype = MFVideoFormat_H264;
    break;

  case mMFVF_H264_ES:
    subtype = MFVideoFormat_H264_ES;
    break;

  case mMFVF_H263:
    subtype = MFVideoFormat_H263;
    break;

  case mMFVF_WMV1:
    subtype = MFVideoFormat_WMV1;
    break;

  case mMFVF_WMV2:
    subtype = MFVideoFormat_WMV2;
    break;

  case mMFVF_WMV3:
    subtype = MFVideoFormat_WMV3;
    break;

  case mMFVF_RGB32:
    subtype = MFVideoFormat_RGB32;
    break;

  case mMFVF_RGB24:
    subtype = MFVideoFormat_RGB24;
    break;

  case mMFVF_YUV420:
    subtype = MFVideoFormat_I420;
    break;

  case mMFVF_HEVC:
    subtype = MFVideoFormat_HEVC;
    break;

  case mMFVF_HEVC_ES:
    subtype = MFVideoFormat_HEVC_ES;
    break;
  }

  mERROR_IF(FAILED(hr = pVideoOutputMediaType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video)), mR_InternalError);
  mERROR_IF(FAILED(hr = pVideoOutputMediaType->SetGUID(MF_MT_SUBTYPE, subtype)), mR_InternalError);
  mERROR_IF(FAILED(hr = pVideoOutputMediaType->SetUINT32(MF_MT_AVG_BITRATE, (uint32_t)pMediaFileInformation->averageBitrate)), mR_InternalError);
  mERROR_IF(FAILED(hr = pVideoOutputMediaType->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive)), mR_InvalidParameter);
  mERROR_IF(FAILED(hr = MFSetAttributeSize(pVideoOutputMediaType, MF_MT_FRAME_SIZE, (uint32_t)pMediaFileInformation->frameSize.x, (uint32_t)pMediaFileInformation->frameSize.y)), mR_InternalError);
  mERROR_IF(FAILED(hr = MFSetAttributeRatio(pVideoOutputMediaType, MF_MT_FRAME_RATE, (uint32_t)pMediaFileInformation->frameRateFraction.x, (uint32_t)pMediaFileInformation->frameRateFraction.y)), mR_InternalError);
  mERROR_IF(FAILED(hr = MFSetAttributeRatio(pVideoOutputMediaType, MF_MT_PIXEL_ASPECT_RATIO, (uint32_t)pMediaFileInformation->pixelAspectRatioFraction.x, (uint32_t)pMediaFileInformation->pixelAspectRatioFraction.y)), mR_InternalError);

  mERROR_IF(FAILED(hr = pMediaFileWriter->pSinkWriter->AddStream(pVideoOutputMediaType, &pMediaFileWriter->videoStreamIndex)), mR_InternalError);

  IMFMediaType *pVideoInputMediaType;
  mDEFER_CALL(&pVideoInputMediaType, _ReleaseReference);
  mERROR_IF(FAILED(hr = MFCreateMediaType(&pVideoInputMediaType)), mR_InternalError);

  mERROR_IF(FAILED(hr = pVideoInputMediaType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video)), mR_InternalError);
  mERROR_IF(FAILED(hr = pVideoInputMediaType->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32)), mR_InternalError);
  mERROR_IF(FAILED(hr = pVideoInputMediaType->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive)), mR_InternalError);
  mERROR_IF(FAILED(hr = MFSetAttributeSize(pVideoInputMediaType, MF_MT_FRAME_SIZE, (uint32_t)pMediaFileInformation->frameSize.x, (uint32_t)pMediaFileInformation->frameSize.y)), mR_InternalError);
  mERROR_IF(FAILED(hr = MFSetAttributeRatio(pVideoInputMediaType, MF_MT_FRAME_RATE, (uint32_t)pMediaFileInformation->frameRateFraction.x, (uint32_t)pMediaFileInformation->frameRateFraction.y)), mR_InternalError);
  mERROR_IF(FAILED(hr = MFSetAttributeRatio(pVideoInputMediaType, MF_MT_PIXEL_ASPECT_RATIO, (uint32_t)pMediaFileInformation->pixelAspectRatioFraction.x, (uint32_t)pMediaFileInformation->pixelAspectRatioFraction.y)), mR_InternalError);

  mERROR_IF(FAILED(hr = pMediaFileWriter->pSinkWriter->SetInputMediaType(pMediaFileWriter->videoStreamIndex, pVideoInputMediaType, nullptr)), mR_InternalError);

  mERROR_IF(FAILED(hr = pMediaFileWriter->pSinkWriter->BeginWriting()), mR_InternalError);

  mRETURN_SUCCESS();
}

mFUNCTION(mMediaFileWriter_Destroy_Internal, IN_OUT mMediaFileWriter *pMediaFileWriter)
{
  mFUNCTION_SETUP();

  mERROR_IF(pMediaFileWriter == nullptr, mR_ArgumentNull);

  mResult result = mR_Success;

  if (!pMediaFileWriter->finalizeCalled)
  {
    mPtr<mMediaFileWriter> _this;
    result = mSharedPointer_Create(&_this, pMediaFileWriter, mSHARED_POINTER_FOREIGN_RESOURCE);

    if(mSUCCEEDED(result))
      result = mMediaFileWriter_Finalize(_this);

    mSharedPointer_Destroy(&_this);
  }

  _ReleaseReference(&pMediaFileWriter->pSinkWriter);

  mERROR_CHECK(mMediaFoundation_RemoveReference());

  mERROR_IF(mFAILED(result), result);

  mRETURN_SUCCESS();
}
