// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <mfapi.h>
#include <mfplay.h>
#include "mVideoFileInputHandler.h"

static volatile size_t _referenceCount = 0;

struct mVideoFileInputHandler
{
  IMFMediaSink *pMediaSink;
  IMFMediaSession *pMediaSession;
  IMFMediaSource *pMediaSource;
  IMFTopology *pTopology;
};

mFUNCTION(mVideoFileInputHandler_Create_Internal, mVideoFileInputHandler *pData, const std::wstring &fileName);
mFUNCTION(mVideoFileInputHandler_Destroy_Internal, mVideoFileInputHandler *pData);
mFUNCTION(mVideoFileInputHandler_InitializeExtenalDependencies);
mFUNCTION(mVideoFileInputHandler_CleanupExtenalDependencies);

template <typename T>
static void _ReleaseReference(T **pData)
{
  if (pData && *pData)
    (*pData)->Release();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mVideoFileInputHandler_Create, OUT mPtr<mVideoFileInputHandler> *pPtr, const std::wstring &fileName)
{
  mFUNCTION_SETUP();

  mERROR_IF(pPtr == nullptr, mR_ArgumentNull);

  if (pPtr != nullptr)
  {
    mERROR_CHECK(mSharedPointer_Destroy(pPtr));
    *pPtr = nullptr;
  }

  mVideoFileInputHandler *pInputHandler = nullptr;
  mDEFER_DESTRUCTION_ON_ERROR(&pInputHandler, mFreePtr);
  mERROR_CHECK(mAllocZero(&pInputHandler, 1));

  mDEFER_DESTRUCTION_ON_ERROR(pPtr, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Create<mVideoFileInputHandler>(pPtr, pInputHandler, [](mVideoFileInputHandler *pData) {mVideoFileInputHandler_Destroy_Internal(pData); }, mAT_mAlloc));
  pInputHandler = nullptr; // to not be destroyed on error.

  mERROR_CHECK(mVideoFileInputHandler_Create_Internal(pPtr->GetPointer(), fileName));
  mRETURN_SUCCESS();
}

mFUNCTION(mVideoFileInputHandler_Destroy, IN_OUT mPtr<mVideoFileInputHandler> *pPtr)
{
  mFUNCTION_SETUP();

  mERROR_IF(pPtr == nullptr || *pPtr == nullptr, mR_ArgumentNull);
  mERROR_CHECK(mSharedPointer_Destroy(pPtr));

  mRETURN_SUCCESS();
}

mFUNCTION(mVideoFileInputHandler_Create_Internal, mVideoFileInputHandler *pInputHandler, const std::wstring &fileName)
{
  mFUNCTION_SETUP();
  mERROR_IF(pInputHandler == nullptr, mR_ArgumentNull);

  HRESULT hr = S_OK;
  mUnused(hr);

  const int64_t referenceCount = ++_referenceCount;

  if (referenceCount == 1)
    mERROR_CHECK(mVideoFileInputHandler_InitializeExtenalDependencies());

  IMFAttributes *pAttributes = nullptr;
  mDEFER_DESTRUCTION(&pAttributes, _ReleaseReference);
  mERROR_IF(FAILED(hr = MFCreateAttributes(&pAttributes, 1)), mR_InternalError);

  mERROR_IF(FAILED(hr = MFCreateMediaSession(pAttributes, &pInputHandler->pMediaSession)), mR_InternalError);

  IMFSourceResolver *pSourceResolver = nullptr;
  mDEFER_DESTRUCTION(&pSourceResolver, _ReleaseReference);
  mERROR_IF(FAILED(hr = MFCreateSourceResolver(&pSourceResolver)), mR_InternalError);

  IUnknown *pSource = nullptr;
  mDEFER_DESTRUCTION(&pSource, _ReleaseReference);

  MF_OBJECT_TYPE objectType = MF_OBJECT_INVALID;
  mERROR_IF(FAILED(hr = pSourceResolver->CreateObjectFromURL(fileName.c_str(), MF_RESOLUTION_MEDIASOURCE, nullptr, &objectType, &pSource)), mR_InternalError);
  mERROR_IF(objectType == MF_OBJECT_INVALID, mR_InternalError);
  mERROR_IF(FAILED(hr = pSource->QueryInterface(__uuidof(IMFMediaSource), (void **)&pInputHandler->pMediaSource)), mR_InternalError);

  mERROR_IF(FAILED(MFCreateTopology(&pInputHandler->pTopology)), mR_InternalError);

  IMFPresentationDescriptor *pPresentationDescriptor = nullptr;
  mDEFER_DESTRUCTION(&pPresentationDescriptor, _ReleaseReference);
  mERROR_IF(FAILED(hr = pInputHandler->pMediaSource->CreatePresentationDescriptor(&pPresentationDescriptor)), mR_InternalError);

  DWORD sourceStreams = 0;
  mERROR_IF(FAILED(hr = pPresentationDescriptor->GetStreamDescriptorCount(&sourceStreams)), mR_InternalError);

  for (DWORD i = 0; i < sourceStreams; ++i)
  {
    IMFStreamDescriptor *pStreamDescriptor = nullptr;
    mDEFER_DESTRUCTION(&pStreamDescriptor, _ReleaseReference);

    BOOL isSelected = false;
    mERROR_IF(FAILED(hr = pPresentationDescriptor->GetStreamDescriptorByIndex(i, &isSelected, &pStreamDescriptor)), mR_InternalError);

    if (isSelected)
    {
      IMFTopologyNode *pSourceNode = nullptr;
      mDEFER_DESTRUCTION(&pSourceNode, _ReleaseReference);
      mERROR_IF(FAILED(hr = MFCreateTopologyNode(MF_TOPOLOGY_SOURCESTREAM_NODE, &pSourceNode)), mR_InternalError);
      mERROR_IF(FAILED(hr = pSourceNode->SetUnknown(MF_TOPONODE_SOURCE, pInputHandler->pMediaSource)), mR_InternalError);
      mERROR_IF(FAILED(hr = pSourceNode->SetUnknown(MF_TOPONODE_PRESENTATION_DESCRIPTOR, pPresentationDescriptor)), mR_InternalError);
      mERROR_IF(FAILED(hr = pSourceNode->SetUnknown(MF_TOPONODE_STREAM_DESCRIPTOR, pStreamDescriptor)), mR_InternalError);

      IMFMediaTypeHandler *pHandler = nullptr;
      mDEFER_DESTRUCTION(&pHandler, _ReleaseReference);
      mERROR_IF(FAILED(hr = pStreamDescriptor->GetMediaTypeHandler(&pHandler)), mR_InternalError);

      GUID guidMajorType = GUID_NULL;
      mERROR_IF(FAILED(hr = pHandler->GetMajorType(&guidMajorType)), mR_InternalError);

      IMFTopologyNode *pOutputNode = nullptr;
      mDEFER_DESTRUCTION(&pOutputNode, _ReleaseReference);
      mERROR_IF(FAILED(hr = MFCreateTopologyNode(MF_TOPOLOGY_OUTPUT_NODE, &pOutputNode)), mR_InternalError);

      if (MFMediaType_Audio == guidMajorType)
      {
        IMFAttributes *pAudioAttributes = nullptr;
        mDEFER_DESTRUCTION(&pAudioAttributes, _ReleaseReference);

        mERROR_IF(FAILED(hr = MFCreateAttributes(&pAudioAttributes, 1)), mR_InternalError);
        mERROR_IF(FAILED(hr = MFCreateAudioRenderer(pAudioAttributes, &pInputHandler->pMediaSink)), mR_InternalError);
      }
      else if (MFMediaType_Video == guidMajorType)
      {
        mERROR_IF(FAILED(hr = MFCreateVideoRenderer(__uuidof(IMFMediaSink), (void **)&pInputHandler->pMediaSink)), mR_InternalError);
      }
      else
      {
        mRETURN_RESULT(mR_InternalError);
      }

      mERROR_IF(FAILED(hr = pOutputNode->SetObject(pInputHandler->pMediaSink)), mR_InternalError);
      mERROR_IF(FAILED(hr = pInputHandler->pTopology->AddNode(pSourceNode)), mR_InternalError);
      mERROR_IF(FAILED(hr = pInputHandler->pTopology->AddNode(pOutputNode)), mR_InternalError);
      mERROR_IF(FAILED(hr = pSourceNode->ConnectOutput(0, pOutputNode, 0)), mR_InternalError);
    }
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mVideoFileInputHandler_Destroy_Internal, mVideoFileInputHandler *pData)
{
  mFUNCTION_SETUP();

  mERROR_IF(pData == nullptr, mR_ArgumentNull);

  if (pData->pMediaSession)
  {
    pData->pMediaSession->Release();
    pData->pMediaSession = nullptr;
  }

  if (pData->pMediaSink)
  {
    pData->pMediaSink->Release();
    pData->pMediaSink = nullptr;
  }

  if (pData->pMediaSource)
  {
    pData->pMediaSource->Release();
    pData->pMediaSource = nullptr;
  }

  if (pData->pTopology)
  {
    pData->pTopology->Release();
    pData->pTopology = nullptr;
  }

  const size_t referenceCount = --_referenceCount;

  if (referenceCount == 0)
    mERROR_CHECK(mVideoFileInputHandler_CleanupExtenalDependencies());

  mRETURN_SUCCESS();
}

mFUNCTION(mVideoFileInputHandler_InitializeExtenalDependencies)
{
  mFUNCTION_SETUP();
  
  HRESULT hr = S_OK;
  mUnused(hr);

  mERROR_IF(FAILED(hr = MFStartup(MF_VERSION, MFSTARTUP_FULL)), mR_InternalError);
  mRETURN_SUCCESS();
}

mFUNCTION(mVideoFileInputHandler_CleanupExtenalDependencies)
{
  mFUNCTION_SETUP();

  HRESULT hr = S_OK;
  mUnused(hr, hr, hr);

  mERROR_IF(FAILED(hr = MFShutdown()), mR_InternalError);
  mRETURN_SUCCESS();
}
