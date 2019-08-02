#include "mImageBuffer.h"
#include "mFile.h"

#define STB_IMAGE_IMPLEMENTATION
#pragma warning(push)
#pragma warning(disable: 4100)
#include "stb_image.h"
#pragma warning(pop)

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "turbojpeg.h"

mFUNCTION(mImageBuffer_Create_Iternal, mPtr<mImageBuffer> *pImageBuffer, IN OPTIONAL mAllocator *pAllocator);
mFUNCTION(mImageBuffer_Destroy_Iternal, mImageBuffer *pImageBuffer);

mFUNCTION(mImageBuffer_Create, OUT mPtr<mImageBuffer> *pImageBuffer, IN OPTIONAL mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pImageBuffer == nullptr, mR_ArgumentNull);
  mERROR_CHECK(mImageBuffer_Create_Iternal(pImageBuffer, pAllocator));

  mRETURN_SUCCESS();
}

mFUNCTION(mImageBuffer_CreateFromFile, OUT mPtr<mImageBuffer> *pImageBuffer, IN OPTIONAL mAllocator *pAllocator, const mString &filename, const mPixelFormat pixelFormat /* = mPF_R8G8B8A8 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pImageBuffer == nullptr, mR_ArgumentNull);
  mERROR_CHECK(mImageBuffer_Create_Iternal(pImageBuffer, pAllocator));

  mERROR_CHECK(mImageBuffer_SetToFile(*pImageBuffer, filename, pixelFormat));

  mRETURN_SUCCESS();
}

mFUNCTION(mImageBuffer_Create, OUT mPtr<mImageBuffer> *pImageBuffer, IN OPTIONAL mAllocator *pAllocator, const mVec2s &size, const mPixelFormat pixelFormat /* = mPF_B8G8R8A8 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pImageBuffer == nullptr, mR_ArgumentNull);
  mERROR_CHECK(mImageBuffer_Create_Iternal(pImageBuffer, pAllocator));
  mERROR_CHECK(mImageBuffer_AllocateBuffer(*pImageBuffer, size, pixelFormat));

  mRETURN_SUCCESS();
}

mFUNCTION(mImageBuffer_Create, OUT mPtr<mImageBuffer> *pImageBuffer, IN OPTIONAL mAllocator *pAllocator, IN const void *pData, const mVec2s &size, const mPixelFormat pixelFormat /* = mPF_B8G8R8A8 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pImageBuffer == nullptr, mR_ArgumentNull);
  mERROR_CHECK(mImageBuffer_Create_Iternal(pImageBuffer, pAllocator));
  mERROR_CHECK(mImageBuffer_SetBuffer(*pImageBuffer, pData, size, pixelFormat));

  mRETURN_SUCCESS();
}

mFUNCTION(mImageBuffer_Create, OUT mPtr<mImageBuffer> *pImageBuffer, IN OPTIONAL mAllocator *pAllocator, IN const void *pData, const mVec2s &size, const size_t stride, const mPixelFormat pixelFormat /* = mPF_B8G8R8A8 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pImageBuffer == nullptr, mR_ArgumentNull);
  mERROR_CHECK(mImageBuffer_Create_Iternal(pImageBuffer, pAllocator));
  mERROR_CHECK(mImageBuffer_SetBuffer(*pImageBuffer, pData, size, stride, pixelFormat));

  mRETURN_SUCCESS();
}

mFUNCTION(mImageBuffer_Create, OUT mPtr<mImageBuffer> *pImageBuffer, IN OPTIONAL mAllocator *pAllocator, IN const void *pData, const mVec2s &size, const mRectangle2D<size_t> &rect, const mPixelFormat pixelFormat /* = mPF_B8G8R8A8 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pImageBuffer == nullptr, mR_ArgumentNull);
  mERROR_CHECK(mImageBuffer_Create_Iternal(pImageBuffer, pAllocator));
  mERROR_CHECK(mImageBuffer_SetBuffer(*pImageBuffer, pData, size, rect, pixelFormat));

  mRETURN_SUCCESS();
}

mFUNCTION(mImageBuffer_Destroy, OUT mPtr<mImageBuffer> *pImageBuffer)
{
  mFUNCTION_SETUP();

  mERROR_IF(pImageBuffer == nullptr, mR_ArgumentNull);
  mERROR_CHECK(mSharedPointer_Destroy(pImageBuffer));
  *pImageBuffer = nullptr;

  mRETURN_SUCCESS();
}

mFUNCTION(mImageBuffer_AllocateBuffer, mPtr<mImageBuffer> &imageBuffer, const mVec2s &size, const mPixelFormat pixelFormat /* = mPF_B8G8R8A8 */)
{
  mFUNCTION_SETUP();

  if (!imageBuffer->ownedResource && imageBuffer->pPixels != nullptr)
    mERROR_CHECK(mImageBuffer_Destroy_Iternal(imageBuffer.GetPointer()));

  size_t allocatedSize;
  mERROR_CHECK(mPixelFormat_GetSize(pixelFormat, size, &allocatedSize));

  if (imageBuffer->pPixels == nullptr)
  {
    uint8_t *pPixels = nullptr;
    mERROR_CHECK(mAllocator_Allocate(imageBuffer->pAllocator, &pPixels, allocatedSize));

    imageBuffer->ownedResource = true;
    imageBuffer->pPixels = pPixels;
    imageBuffer->allocatedSize = allocatedSize;
    imageBuffer->currentSize = size;
    imageBuffer->lineStride = size.x;
    imageBuffer->pixelFormat = pixelFormat;
  }
  else
  {
    if (imageBuffer->allocatedSize < allocatedSize)
    {
      uint8_t *pPixels = nullptr;
      mERROR_CHECK(mAllocator_Reallocate(imageBuffer->pAllocator, &pPixels, allocatedSize));

      imageBuffer->allocatedSize = allocatedSize;
      imageBuffer->ownedResource = true;
      imageBuffer->pPixels = pPixels;
    }

    imageBuffer->currentSize = size;
    imageBuffer->pixelFormat = pixelFormat;
    imageBuffer->lineStride = size.x;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mImageBuffer_SetBuffer, mPtr<mImageBuffer> &imageBuffer, IN const void *pData, const mVec2s &size, const mPixelFormat pixelFormat /* = mPF_B8G8R8A8 */)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mImageBuffer_SetBuffer(imageBuffer, pData, size, size.x, pixelFormat));

  mRETURN_SUCCESS();
}

mFUNCTION(mImageBuffer_SetBuffer, mPtr<mImageBuffer> &imageBuffer, IN const void *pData, const mVec2s &size, const size_t stride, const mPixelFormat pixelFormat /* = mPF_B8G8R8A8 */)
{
  mFUNCTION_SETUP();

  if (imageBuffer->pPixels != nullptr)
    mERROR_CHECK(mImageBuffer_Destroy_Iternal(imageBuffer.GetPointer()));

  mERROR_CHECK(mPixelFormat_GetSize(pixelFormat, size, &imageBuffer->allocatedSize));

  imageBuffer->ownedResource = false;
  imageBuffer->pPixels = (uint8_t *)pData;
  imageBuffer->pixelFormat = pixelFormat;
  imageBuffer->currentSize = size;
  imageBuffer->lineStride = stride;

  mRETURN_SUCCESS();
}

mFUNCTION(mImageBuffer_SetBuffer, mPtr<mImageBuffer> &imageBuffer, IN const void *pData, const mVec2s &size, const mRectangle2D<size_t> &rect, const mPixelFormat pixelFormat)
{
  mFUNCTION_SETUP();

  if (imageBuffer->pPixels != nullptr)
    mERROR_CHECK(mImageBuffer_Destroy_Iternal(imageBuffer.GetPointer()));

  mERROR_IF(rect.x + rect.w >= size.x, mR_InvalidParameter);
  mERROR_IF(rect.y + rect.h >= size.y, mR_InvalidParameter);

  bool hasSubBuffers;
  mERROR_CHECK(mPixelFormat_HasSubBuffers(pixelFormat, &hasSubBuffers));
  mERROR_IF(hasSubBuffers, mR_InvalidParameter);

  size_t pixelFormatUnitSize;
  mERROR_CHECK(mPixelFormat_GetUnitSize(pixelFormat, &pixelFormatUnitSize));

  imageBuffer->ownedResource = false;
  imageBuffer->pPixels = ((uint8_t *)pData) + (size.x * rect.y + rect.x) * pixelFormatUnitSize;
  imageBuffer->lineStride = size.x;
  imageBuffer->currentSize = mVec2s(rect.w, rect.h);
  imageBuffer->allocatedSize = (size.x * size.y - rect.x - rect.y * size.x) * pixelFormatUnitSize;
  imageBuffer->pixelFormat = pixelFormat;

  mRETURN_SUCCESS();
}

mFUNCTION(mImageBuffer_CopyTo, mPtr<mImageBuffer> &source, mPtr<mImageBuffer> &target, const mImageBuffer_CopyFlags copyFlags)
{
  mFUNCTION_SETUP();

  mERROR_IF(source->pPixels == nullptr, mR_NotInitialized);
  
  if (target->pPixels == nullptr)
    mERROR_CHECK(mImageBuffer_AllocateBuffer(target, source->currentSize, source->pixelFormat));

  if ((copyFlags & mImageBuffer_CopyFlags::mIB_CF_ResizeAllowed) && target->ownedResource)
  {
    if (source->currentSize != target->currentSize)
    {
      mERROR_IF(source->pixelFormat != target->pixelFormat && (copyFlags & mImageBuffer_CopyFlags::mIB_CF_PixelFormatChangeAllowed) == 0, mR_InvalidParameter);
      
      mERROR_CHECK(mImageBuffer_AllocateBuffer(target, source->currentSize, source->pixelFormat));
    }
  }
  else
  {
    mERROR_IF(source->currentSize != target->currentSize, mR_InvalidParameter);
  }

  if ((copyFlags & mImageBuffer_CopyFlags::mIB_CF_PixelFormatChangeAllowed) && target->ownedResource)
  {
    mERROR_IF(source->currentSize != target->currentSize && (copyFlags & mImageBuffer_CopyFlags::mIB_CF_ResizeAllowed) == 0, mR_InvalidParameter);

    mERROR_CHECK(mImageBuffer_AllocateBuffer(target, source->currentSize, source->pixelFormat));
  }
  else
  {
    mERROR_IF(source->pixelFormat != target->pixelFormat, mR_InvalidParameter);
  }

  if (source->lineStride == target->lineStride)
  {
    size_t copiedSize;
    mERROR_CHECK(mPixelFormat_GetSize(source->pixelFormat, mVec2s(source->lineStride, source->currentSize.y), &copiedSize));
    mERROR_CHECK(mAllocator_Copy(source->pAllocator, target->pPixels, source->pPixels, copiedSize));
  }
  else
  {
    size_t subBufferCount;
    mERROR_CHECK(mPixelFormat_GetSubBufferCount(source->pixelFormat, &subBufferCount));

    for (size_t i = 0; i < subBufferCount; i++)
    {
      size_t sourceSubBufferOffset;
      mERROR_CHECK(mPixelFormat_GetSubBufferOffset(source->pixelFormat, i, mVec2s(source->lineStride, source->currentSize.y), &sourceSubBufferOffset));

      size_t targetSubBufferOffset;
      mERROR_CHECK(mPixelFormat_GetSubBufferOffset(target->pixelFormat, i, mVec2s(target->lineStride, target->currentSize.y), &targetSubBufferOffset));

      mVec2s subBufferSize;
      mERROR_CHECK(mPixelFormat_GetSubBufferSize(source->pixelFormat, i, source->currentSize, &subBufferSize));

      size_t sourceSubBufferStride;
      mERROR_CHECK(mPixelFormat_GetSubBufferStride(source->pixelFormat, i, source->lineStride, &sourceSubBufferStride));

      size_t targetSubBufferStride;
      mERROR_CHECK(mPixelFormat_GetSubBufferStride(target->pixelFormat, i, target->lineStride, &targetSubBufferStride));

      mPixelFormat subBufferPixelFormat;
      mERROR_CHECK(mPixelFormat_GetSubBufferPixelFormat(source->pixelFormat, i, &subBufferPixelFormat));

      size_t subBufferUnitSize;
      mERROR_CHECK(mPixelFormat_GetUnitSize(subBufferPixelFormat, &subBufferUnitSize));

      uint8_t *pSourceSubBufferPixels = source->pPixels + sourceSubBufferOffset;
      uint8_t *pTargetSubBufferPixels = target->pPixels + targetSubBufferOffset;

      size_t targetLineStride = 0;
      size_t sourceLineStride = 0;

      for (size_t y = 0; y < source->currentSize.y; y++)
      {
        mERROR_CHECK(mAllocator_Copy(source->pAllocator, pTargetSubBufferPixels + targetLineStride, pSourceSubBufferPixels + sourceLineStride, subBufferSize.x));

        targetLineStride += sourceSubBufferStride * subBufferUnitSize;
        sourceLineStride += targetSubBufferStride * subBufferUnitSize;
      }
    }
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mImageBuffer_SaveAsPng, mPtr<mImageBuffer> &imageBuffer, const mString &filename)
{
  mFUNCTION_SETUP();

  mERROR_IF(imageBuffer == nullptr || imageBuffer->pPixels == nullptr, mR_NotInitialized);

  int components;

  switch (imageBuffer->pixelFormat)
  {
  case mPF_R8G8B8A8:
    components = 4;
    break;

  case mPF_R8G8B8:
    components = 3;
    break;

  case mPF_Monochrome8:
    components = 1;
    break;

  default:
    mRETURN_RESULT(mR_OperationNotSupported);
  }

  int result = stbi_write_png(filename.c_str(), (int)imageBuffer->currentSize.x, (int)imageBuffer->currentSize.y, components, imageBuffer->pPixels, (int)(imageBuffer->lineStride * sizeof(uint8_t)) * components);

  mERROR_IF(result == 0, mR_InternalError);

  mRETURN_SUCCESS();
}

mFUNCTION(mImageBuffer_SaveAsJpeg, mPtr<mImageBuffer> &imageBuffer, const mString &filename)
{
  mFUNCTION_SETUP();

  mERROR_IF(imageBuffer == nullptr || imageBuffer->pPixels == nullptr, mR_NotInitialized);
  mERROR_IF(imageBuffer->lineStride != imageBuffer->currentSize.x, mR_InvalidParameter);

  int components;

  switch (imageBuffer->pixelFormat)
  {
  case mPF_R8G8B8A8:
    components = 4;
    break;

  case mPF_R8G8B8:
    components = 3;
    break;

  case mPF_Monochrome8:
    components = 1;
    break;

  default:
    mRETURN_RESULT(mR_OperationNotSupported);
  }

  int result = stbi_write_jpg(filename.c_str(), (int)imageBuffer->currentSize.x, (int)imageBuffer->currentSize.y, components, imageBuffer->pPixels, 85);

  mERROR_IF(result == 0, mR_InternalError);

  mRETURN_SUCCESS();
}

mFUNCTION(mImageBuffer_SaveAsBmp, mPtr<mImageBuffer> &imageBuffer, const mString &filename)
{
  mFUNCTION_SETUP();

  mERROR_IF(imageBuffer == nullptr || imageBuffer->pPixels == nullptr, mR_NotInitialized);
  mERROR_IF(imageBuffer->lineStride != imageBuffer->currentSize.x, mR_InvalidParameter);

  int components;

  switch (imageBuffer->pixelFormat)
  {
  case mPF_R8G8B8A8:
    components = 4;
    break;

  case mPF_R8G8B8:
    components = 3;
    break;

  case mPF_Monochrome8:
    components = 1;
    break;

  default:
    mRETURN_RESULT(mR_OperationNotSupported);
  }

  int result = stbi_write_bmp(filename.c_str(), (int)imageBuffer->currentSize.x, (int)imageBuffer->currentSize.y, components, imageBuffer->pPixels);

  mERROR_IF(result == 0, mR_InternalError);

  mRETURN_SUCCESS();
}

mFUNCTION(mImageBuffer_SaveAsTga, mPtr<mImageBuffer> &imageBuffer, const mString &filename)
{
  mFUNCTION_SETUP();

  mERROR_IF(imageBuffer == nullptr || imageBuffer->pPixels == nullptr, mR_NotInitialized);
  mERROR_IF(imageBuffer->lineStride != imageBuffer->currentSize.x, mR_InvalidParameter);

  int components;

  switch (imageBuffer->pixelFormat)
  {
  case mPF_R8G8B8A8:
    components = 4;
    break;

  case mPF_R8G8B8:
    components = 3;
    break;

  case mPF_Monochrome8:
    components = 1;
    break;

  default:
    mRETURN_RESULT(mR_OperationNotSupported);
  }

  int result = stbi_write_tga(filename.c_str(), (int)imageBuffer->currentSize.x, (int)imageBuffer->currentSize.y, components, imageBuffer->pPixels);

  mERROR_IF(result == 0, mR_InternalError);

  mRETURN_SUCCESS();
}

mFUNCTION(mImageBuffer_SetToFile, mPtr<mImageBuffer> &imageBuffer, const mString &filename, const mPixelFormat pixelFormat /* = mPF_B8G8R8A8 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(imageBuffer == nullptr, mR_ArgumentNull);
  mERROR_IF(filename.hasFailed || filename.text == nullptr, mR_InvalidParameter);

  bool fileExists = false;
  mERROR_CHECK(mFile_Exists(filename, &fileExists));
  mERROR_IF(!fileExists, mR_ResourceNotFound);

  uint8_t *pData = nullptr;
  size_t size = 0;
  mERROR_CHECK(mFile_ReadRaw(filename, &pData, &mDefaultTempAllocator, &size));
  mDEFER(mAllocator_FreePtr(&mDefaultTempAllocator, &pData));

  int components = 4;
  mPixelFormat readPixelFormat = mPF_R8G8B8A8;

  const bool tryJpeg = (size > 3 && pData[0] == 0xFF && pData[1] == 0xD8 && pData[2] == 0xFF);

  if (tryJpeg)
  {
    tjhandle decoder = tjInitDecompress();

    int32_t width, height;

    if (decoder == nullptr)
      goto jpeg_decoder_failed;

    mDEFER(tjDestroy(decoder));

    if (0 != tjDecompressHeader(decoder, pData, (uint32_t)size, &width, &height))
      goto jpeg_decoder_failed;

    if (pixelFormat == mPF_YUV420)
    {
      if (mFAILED(mImageBuffer_AllocateBuffer(imageBuffer, mVec2s((size_t)width, (size_t)height), pixelFormat)))
        goto jpeg_decoder_failed;

      if (0 != tjDecompressToYUV2(decoder, pData, (uint32_t)size, imageBuffer->pPixels, width, 1, height, TJFLAG_FASTDCT))
        goto jpeg_decoder_failed;
    }
    else
    {
      int32_t tjPixelFormat = 0;
      size_t pixelSize = 1;

      switch (pixelFormat)
      {
      case mPF_B8G8R8:
        tjPixelFormat = TJPF_BGR;
        break;

      case mPF_R8G8B8:
        tjPixelFormat = TJPF_RGB;
        break;

      case mPF_B8G8R8A8:
        tjPixelFormat = TJPF_BGRA;
        break;

      case mPF_R8G8B8A8:
        tjPixelFormat = TJPF_RGBA;
        break;

      case mPF_Monochrome8:
        tjPixelFormat = TJPF_GRAY;
        break;

      default:
        goto jpeg_decoder_failed;
        break;
      }

      if (mFAILED(mImageBuffer_AllocateBuffer(imageBuffer, mVec2s((size_t)width, (size_t)height), pixelFormat)) || mFAILED(mPixelFormat_GetUnitSize(pixelFormat, &pixelSize)))
        goto jpeg_decoder_failed;

      if (0 != tjDecompress2(decoder, pData, (uint32_t)size, imageBuffer->pPixels, width, (int)(width * pixelSize), height, tjPixelFormat, TJFLAG_FASTDCT))
        goto jpeg_decoder_failed;
    }

    mRETURN_SUCCESS();
  }

jpeg_decoder_failed:

  switch (pixelFormat)
  {
  case mPF_B8G8R8:
  case mPF_R8G8B8:
    components = 3;
    readPixelFormat = mPF_R8G8B8;
    break;

  case mPF_B8G8R8A8:
  case mPF_R8G8B8A8:
    components = 4;
    readPixelFormat = mPF_R8G8B8A8;
    break;

  case mPF_YUV422:
  case mPF_YUV420:
    components = 3;
    readPixelFormat = mPF_R8G8B8;
    break;

  case mPF_Monochrome8:
  case mPF_Monochrome16:
    components = 1;
    readPixelFormat = mPF_Monochrome8;
    break;

  default:
    mRETURN_RESULT(mR_OperationNotSupported);
    break;
  }

  int x, y, originalChannelCount;
  stbi_uc *pResult = stbi_load_from_memory(pData, (int)size, &x, &y, &originalChannelCount, components);

  if (pResult == nullptr)
    mRETURN_RESULT(mR_InternalError);

  mDEFER_CALL((void *)pResult, stbi_image_free);

  mERROR_CHECK(mImageBuffer_AllocateBuffer(imageBuffer, mVec2s((size_t)x, (size_t)y), pixelFormat));

  size_t imageBytes;
  mERROR_CHECK(mPixelFormat_GetSize(imageBuffer->pixelFormat, imageBuffer->currentSize, &imageBytes));

  if (readPixelFormat != pixelFormat)
  {
    mPtr<mImageBuffer> tempImageBuffer;
    mDEFER_CALL(&tempImageBuffer, mImageBuffer_Destroy);
    mERROR_CHECK(mImageBuffer_Create(&tempImageBuffer, &mDefaultTempAllocator, mVec2s((size_t)x, (size_t)y), readPixelFormat));

    if (tempImageBuffer->pPixels != nullptr) // If someone would corrupt the implementation so that it would allocate an empty image on `mImageBuffer_Create`.
      mERROR_CHECK(mAllocator_FreePtr(tempImageBuffer->pAllocator, &tempImageBuffer->pPixels));

    tempImageBuffer->pAllocator = &mNullAllocator;
    tempImageBuffer->pPixels = (uint8_t *)pResult;

    mERROR_CHECK(mPixelFormat_TransformBuffer(tempImageBuffer, imageBuffer));
  }
  else
  {
    mERROR_CHECK(mAllocator_Copy(imageBuffer->pAllocator, imageBuffer->pPixels, (uint8_t *)pResult, imageBytes));
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mImageBuffer_FlipY, mPtr<mImageBuffer> &imageBuffer)
{
  mFUNCTION_SETUP();

  mERROR_IF(imageBuffer == nullptr, mR_ArgumentNull);

  size_t subBufferCount = 0;
  mERROR_CHECK(mPixelFormat_GetSubBufferCount(imageBuffer->pixelFormat, &subBufferCount));

  uint8_t *pBuffer = nullptr;
  mDEFER(mAllocator_FreePtr(nullptr, &pBuffer));
  size_t bufferSize = 0;

  for (size_t subBuffer = 0; subBuffer < subBufferCount; ++subBuffer)
  {
    mVec2s subBufferSize;
    mERROR_CHECK(mPixelFormat_GetSubBufferSize(imageBuffer->pixelFormat, subBuffer, imageBuffer->currentSize, &subBufferSize));

    mPixelFormat subBufferPixelFormat;
    mERROR_CHECK(mPixelFormat_GetSubBufferPixelFormat(imageBuffer->pixelFormat, subBuffer, &subBufferPixelFormat));

    size_t subBufferPixelFormatUnitSize;
    mERROR_CHECK(mPixelFormat_GetUnitSize(subBufferPixelFormat, &subBufferPixelFormatUnitSize));

    size_t subBufferOffset;
    mERROR_CHECK(mPixelFormat_GetSubBufferOffset(imageBuffer->pixelFormat, subBuffer, imageBuffer->currentSize, &subBufferOffset));

    size_t subBufferStride;
    mERROR_CHECK(mPixelFormat_GetSubBufferStride(imageBuffer->pixelFormat, subBuffer, imageBuffer->lineStride, &subBufferStride));

    size_t strideBytes = subBufferPixelFormatUnitSize * subBufferStride;
    uint8_t *pSubBuffer = imageBuffer->pPixels + subBufferOffset;
    uint8_t *pSubBufferEnd = pSubBuffer + (subBufferSize.y - 1) * strideBytes;

    if (strideBytes > bufferSize)
    {
      mERROR_CHECK(mAllocator_Reallocate(nullptr, &pBuffer, strideBytes));
      bufferSize = strideBytes;
    }

    while (pSubBuffer < pSubBufferEnd)
    {
      mERROR_CHECK(mAllocator_Copy(imageBuffer->pAllocator, pBuffer, pSubBuffer, strideBytes));
      mERROR_CHECK(mAllocator_Copy(imageBuffer->pAllocator, pSubBuffer, pSubBufferEnd, strideBytes));
      mERROR_CHECK(mAllocator_Copy(imageBuffer->pAllocator, pSubBufferEnd, pBuffer, strideBytes));

      pSubBuffer += strideBytes;
      pSubBufferEnd -= strideBytes;
    }
  }

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mImageBuffer_Destroy_Iternal, mImageBuffer *pImageBuffer)
{
  mFUNCTION_SETUP();

  mERROR_IF(pImageBuffer == nullptr, mR_ArgumentNull);

  if (pImageBuffer->ownedResource && pImageBuffer->pPixels != nullptr)
    mSTDRESULT = mAllocator_FreePtr(pImageBuffer->pAllocator, &pImageBuffer->pPixels);

  pImageBuffer->ownedResource = false;
  pImageBuffer->pPixels = nullptr;
  pImageBuffer->allocatedSize = 0;
  pImageBuffer->currentSize = mVec2s(0);
  pImageBuffer->lineStride = 0;

  mERROR_IF(mFAILED(mSTDRESULT), mSTDRESULT);

  mRETURN_SUCCESS();
}

mFUNCTION(mImageBuffer_Create_Iternal, mPtr<mImageBuffer> *pImageBuffer, IN OPTIONAL mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pImageBuffer == nullptr, mR_ArgumentNull);

  if (pImageBuffer != nullptr)
  {
    mERROR_CHECK(mSharedPointer_Destroy(pImageBuffer));
    *pImageBuffer = nullptr;
  }

  mImageBuffer *pImageBufferRaw = nullptr;
  mDEFER_ON_ERROR(mAllocator_FreePtr(pAllocator, &pImageBufferRaw));
  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &pImageBufferRaw, 1));

  mDEFER_CALL_ON_ERROR(pImageBuffer, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Create<mImageBuffer>(pImageBuffer, pImageBufferRaw, [](mImageBuffer *pData) { mImageBuffer_Destroy_Iternal(pData); }, pAllocator));
  pImageBufferRaw = nullptr; // to not be destroyed on error.

  (*pImageBuffer)->pAllocator = pAllocator;

  mRETURN_SUCCESS();
}
