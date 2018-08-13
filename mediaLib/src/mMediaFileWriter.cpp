// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

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

mFUNCTION(mMediaFileWriter_Create_Internal, OUT mMediaFileWriter *pMediaFileWriter, IN mAllocator *pAllocator, const std::wstring &filename, IN mMediaFileInformation *pMediaFileInformation);
mFUNCTION(mMediaFileWriter_Destroy_Internal, IN_OUT mMediaFileWriter *pMediaFileWriter);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mMediaFileWriter_Create, OUT mPtr<mMediaFileWriter> *pMediaFileWriter, IN mAllocator *pAllocator, const std::wstring &filename, IN mMediaFileInformation *pMediaFileInformation)
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

  mDEFER_DESTRUCTION_ON_ERROR(pMediaFileWriter, mSharedPointer_Destroy);
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

  size_t bufferPixels;
  mERROR_CHECK(mPixelFormat_GetSize(imageBuffer->pixelFormat, imageBuffer->currentSize, &bufferPixels));

  IMFMediaBuffer *pBuffer = nullptr;
  mDEFER_DESTRUCTION(&pBuffer, _ReleaseReference);
  mERROR_IF(FAILED(MFCreateMemoryBuffer((DWORD)bufferPixels, &pBuffer)), mR_InternalError);

  BYTE *pData;
  mERROR_IF(FAILED(pBuffer->Lock(&pData, nullptr, nullptr)), mR_InternalError);

  size_t pixelFormatUnitSize;
  mERROR_CHECK(mPixelFormat_GetUnitSize(imageBuffer->pixelFormat, &pixelFormatUnitSize));

  mERROR_IF(FAILED(MFCopyImage(pData, (DWORD)(imageBuffer->currentSize.x * pixelFormatUnitSize), imageBuffer->pPixels, (DWORD)(imageBuffer->lineStride * pixelFormatUnitSize), (DWORD)(imageBuffer->currentSize.x * pixelFormatUnitSize), (DWORD)imageBuffer->currentSize.y)), mR_InternalError);
  mERROR_IF(FAILED(pBuffer->Unlock()), mR_InternalError);

  IMFSample *pSample = nullptr;
  mDEFER_DESTRUCTION(&pSample, _ReleaseReference);
  mERROR_IF(FAILED(MFCreateSample(&pSample)), mR_InternalError);
  mERROR_IF(FAILED(pSample->AddBuffer(pBuffer)), mR_InternalError);
  mERROR_IF(FAILED(pSample->SetSampleTime(mediaFileWriter->lastFrameTimestamp)), mR_InternalError);

  const size_t duration = (10 * 1000 * 1000 * mediaFileWriter->mediaFileInformation.frameRateFraction.x) / mediaFileWriter->mediaFileInformation.frameRateFraction.y;
  mediaFileWriter->lastFrameTimestamp += duration;
  mERROR_IF(mediaFileWriter->lastFrameTimestamp > LONGLONG_MAX, mR_IndexOutOfBounds);

  mERROR_IF(duration > LONGLONG_MAX, mR_IndexOutOfBounds);
  mERROR_IF(FAILED(pSample->SetSampleDuration((LONGLONG)duration)), mR_InternalError);

  mERROR_IF(FAILED(mediaFileWriter->pSinkWriter->WriteSample(mediaFileWriter->videoStreamIndex, pSample)), mR_InternalError);

  mRETURN_SUCCESS();
}

mFUNCTION(mMediaFileWriter_Finalize, mPtr<mMediaFileWriter> &mediaFileWriter)
{
  mFUNCTION_SETUP();

  mERROR_IF(mediaFileWriter == nullptr, mR_ArgumentNull);
  mERROR_IF(FAILED(mediaFileWriter->pSinkWriter->Finalize()), mR_InternalError);

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mMediaFileWriter_Create_Internal, OUT mMediaFileWriter *pMediaFileWriter, IN mAllocator *pAllocator, const std::wstring &filename, IN mMediaFileInformation *pMediaFileInformation)
{
  mFUNCTION_SETUP();

  mERROR_IF(pMediaFileWriter == nullptr || pMediaFileInformation == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mMediaFoundation_AddReference());

  mERROR_IF(pMediaFileInformation->frameSize.x > UINT32_MAX || pMediaFileInformation->frameSize.y > UINT32_MAX || pMediaFileInformation->averageBitrate > UINT32_MAX || pMediaFileInformation->frameRateFraction.x > UINT32_MAX || pMediaFileInformation->frameRateFraction.y > UINT32_MAX || pMediaFileInformation->pixelAspectRatioFraction.x > UINT32_MAX || pMediaFileInformation->pixelAspectRatioFraction.y > UINT32_MAX, mR_IndexOutOfBounds);

  pMediaFileWriter->mediaFileInformation = *pMediaFileInformation;
  pMediaFileWriter->lastFrameTimestamp = 0;
  mUnused(pAllocator);

  HRESULT hr;
  mUnused(hr);

  mERROR_IF(FAILED(hr = MFCreateSinkWriterFromURL(filename.c_str(), nullptr, nullptr, &pMediaFileWriter->pSinkWriter)), mR_InternalError);

  IMFMediaType *pVideoOutputMediaType;
  mDEFER_DESTRUCTION(&pVideoOutputMediaType, _ReleaseReference);
  mERROR_IF(FAILED(hr = MFCreateMediaType(&pVideoOutputMediaType)), mR_InternalError);

  mERROR_IF(FAILED(hr = pVideoOutputMediaType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video)), mR_InternalError);
  mERROR_IF(FAILED(hr = pVideoOutputMediaType->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_H264)), mR_InternalError);
  mERROR_IF(FAILED(hr = pVideoOutputMediaType->SetUINT32(MF_MT_AVG_BITRATE, (uint32_t)pMediaFileInformation->averageBitrate)), mR_InternalError);
  mERROR_IF(FAILED(hr = pVideoOutputMediaType->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive)), mR_InvalidParameter);
  mERROR_IF(FAILED(hr = MFSetAttributeSize(pVideoOutputMediaType, MF_MT_FRAME_SIZE, (uint32_t)pMediaFileInformation->frameSize.x, (uint32_t)pMediaFileInformation->frameSize.y)), mR_InternalError);
  mERROR_IF(FAILED(hr = MFSetAttributeRatio(pVideoOutputMediaType, MF_MT_FRAME_RATE, (uint32_t)pMediaFileInformation->frameRateFraction.x, (uint32_t)pMediaFileInformation->frameRateFraction.y)), mR_InternalError);
  mERROR_IF(FAILED(hr = MFSetAttributeRatio(pVideoOutputMediaType, MF_MT_PIXEL_ASPECT_RATIO, (uint32_t)pMediaFileInformation->pixelAspectRatioFraction.x, (uint32_t)pMediaFileInformation->pixelAspectRatioFraction.y)), mR_InternalError);

  mERROR_IF(FAILED(hr = pMediaFileWriter->pSinkWriter->AddStream(pVideoOutputMediaType, &pMediaFileWriter->videoStreamIndex)), mR_InternalError);

  IMFMediaType *pVideoInputMediaType;
  mDEFER_DESTRUCTION(&pVideoInputMediaType, _ReleaseReference);
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

  _ReleaseReference(&pMediaFileWriter->pSinkWriter);

  mERROR_CHECK(mMediaFoundation_RemoveReference());

  mRETURN_SUCCESS();
}
