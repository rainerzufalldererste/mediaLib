#include "mImageBuffer.h"

#include "mFile.h"
#include "mProfiler.h"

#define STB_IMAGE_IMPLEMENTATION
#pragma warning(push)
#pragma warning(disable: 4100)
#include "stb_image.h"
#pragma warning(pop)

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "turbojpeg.h"

#define FPNG_RAW_PTRS
#include "fpng.h"

//////////////////////////////////////////////////////////////////////////

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "QgL/buye0i29ugNB9Gb1OObJgZCM5z3w1h8+qkQFX7BE67pbY5EwXZU9SdPR2evaOp1Y1qEZqrZ6b/OT"
#endif

struct WriteFuncData
{
  mUniqueContainer<mFileWriter> file;
  mResult result = mR_Success;

  static void Write(void *pWriteFuncData, void *pData, const int32_t size)
  {
    WriteFuncData *pSelf = static_cast<WriteFuncData *>(pWriteFuncData);

    if (mFAILED(pSelf->result))
      return;

    pSelf->result = mFileWriter_WriteRaw(pSelf->file, reinterpret_cast<uint8_t *>(pData), (size_t)size);
  }
};

//////////////////////////////////////////////////////////////////////////

static mFUNCTION(mImageBuffer_Create_Iternal, mPtr<mImageBuffer> *pImageBuffer, IN OPTIONAL mAllocator *pAllocator);
static mFUNCTION(mImageBuffer_Destroy_Iternal, mImageBuffer *pImageBuffer);

//////////////////////////////////////////////////////////////////////////

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

mFUNCTION(mImageBuffer_CreateFromData, OUT mPtr<mImageBuffer> *pImageBuffer, IN OPTIONAL mAllocator *pAllocator, IN const uint8_t *pData, const size_t size, const mPixelFormat pixelFormat /* = mPF_R8G8B8A8 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pImageBuffer == nullptr, mR_ArgumentNull);
  mERROR_CHECK(mImageBuffer_Create_Iternal(pImageBuffer, pAllocator));

  mERROR_CHECK(mImageBuffer_SetToData(*pImageBuffer, pData, size, pixelFormat));

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
    mERROR_CHECK(mAllocator_AllocateZero(imageBuffer->pAllocator, &pPixels, allocatedSize));

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
    mERROR_CHECK(mMemcpy(target->pPixels, source->pPixels, copiedSize));
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
        mERROR_CHECK(mMemcpy(pTargetSubBufferPixels + targetLineStride, pSourceSubBufferPixels + sourceLineStride, subBufferSize.x));

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
  mERROR_IF(imageBuffer->currentSize.x > INT32_MAX || imageBuffer->currentSize.y > INT32_MAX, mR_ResourceIncompatible);

  int32_t channels = 4;

  switch (imageBuffer->pixelFormat)
  {
  case mPF_R8G8B8A8:
    channels = 4;
    break;

  case mPF_R8G8B8:
    channels = 3;
    break;

  case mPF_Monochrome8:
    channels = 1;
    break;

  default:
    mRETURN_RESULT(mR_OperationNotSupported);
  }

  // Attempt to save using fpng.
  if (channels == 3 && channels == 4)
  {
    uint8_t *pData = nullptr;
    uint64_t dataSize = 0;

    struct _internal 
    {
      static void *realloc(void *pData, size_t size)
      {
        if (mFAILED(mRealloc(reinterpret_cast<uint8_t **>(&pData), size)))
          return nullptr;

        return pData;
      }

      static void free(void *pData)
      {
        mFree(pData);
      }
    };

    mDEFER_CALL(pData, _internal::free);

    if (fpng::fpng_encode_image_to_memory_ptr(imageBuffer->pPixels, (int32_t)imageBuffer->currentSize.x, (int32_t)imageBuffer->currentSize.y, channels, &pData, dataSize, _internal::realloc, _internal::free))
    {
      mERROR_CHECK(mFile_WriteRaw(filename, pData, dataSize));
      mRETURN_SUCCESS();
    }
  }

  // Attempt to save using stbi.
  {
    WriteFuncData fileWriteData;
    mERROR_CHECK(mFileWriter_Create(&fileWriteData.file, filename));

    const int result = stbi_write_png_to_func(WriteFuncData::Write, &fileWriteData, (int32_t)imageBuffer->currentSize.x, (int32_t)imageBuffer->currentSize.y, channels, imageBuffer->pPixels, (int32_t)(imageBuffer->lineStride * sizeof(uint8_t)) * channels);

    mERROR_IF(mFAILED(fileWriteData.result), fileWriteData.result);
    mERROR_IF(result == 0, mR_InternalError);
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mImageBuffer_SaveAsJpeg, mPtr<mImageBuffer> &imageBuffer, const mString &filename)
{
  mFUNCTION_SETUP();

  mERROR_IF(imageBuffer == nullptr || imageBuffer->pPixels == nullptr, mR_NotInitialized);
  mERROR_IF(imageBuffer->lineStride != imageBuffer->currentSize.x, mR_InvalidParameter);
  mERROR_IF(imageBuffer->currentSize.x > INT32_MAX || imageBuffer->currentSize.y > INT32_MAX, mR_ResourceIncompatible);

  // Attempt to use turbo jpeg.
  {
    static tjhandle encoder = tjInitCompress();

    if (encoder == nullptr)
      goto fast_jpeg_encoder_failed;

    switch (imageBuffer->pixelFormat)
    {
    case mPF_B8G8R8:
    case mPF_B8G8R8A8:
    case mPF_R8G8B8:
    case mPF_R8G8B8A8:
    case mPF_Monochrome8:
    {
      size_t componentCount = 0;

      if (mFAILED(mPixelFormat_GetComponentCount(imageBuffer->pixelFormat, &componentCount)))
        goto fast_jpeg_encoder_failed;

      int32_t tjPixelFormat = 0;

      switch (imageBuffer->pixelFormat)
      {
      case mPF_B8G8R8:
        tjPixelFormat = TJPF_BGR;
        break;

      case mPF_B8G8R8A8:
        tjPixelFormat = TJPF_BGRA;
        break;

      case mPF_R8G8B8:
        tjPixelFormat = TJPF_RGB;
        break;

      case mPF_R8G8B8A8:
        tjPixelFormat = TJPF_RGBA;
        break;

      case mPF_Monochrome8:
        tjPixelFormat = TJPF_GRAY;
        break;

      default:
        mFAIL_DEBUG("Invalid State.");
        goto fast_jpeg_encoder_failed;
      }

      uint8_t *pJpegBuffer = nullptr;
      unsigned long jpegBufferSize = 0;

      mDEFER(
        if (pJpegBuffer != nullptr)
          tjFree(pJpegBuffer);
      );

      const int32_t result = tjCompress2(encoder, imageBuffer->pPixels, (int32_t)imageBuffer->currentSize.x, (int32_t)(imageBuffer->lineStride * componentCount), (int32_t)imageBuffer->currentSize.y, tjPixelFormat, &pJpegBuffer, &jpegBufferSize, TJSAMP_420, 85, TJFLAG_FASTDCT);

      if (result != 0)
        goto fast_jpeg_encoder_failed;

      mERROR_CHECK(mFile_WriteRaw(filename, pJpegBuffer, (size_t)jpegBufferSize));
      mRETURN_SUCCESS();

      break;
    }

    case mPF_YUV411:
    case mPF_YUV420:
    case mPF_YUV422:
    case mPF_YUV440:
    case mPF_YUV444:
    {
      int32_t tjSubSampling = 0;

      switch (imageBuffer->pixelFormat)
      {
      case mPF_YUV411:
        tjSubSampling = TJSAMP_411;
        break;

      case mPF_YUV420:
        tjSubSampling = TJSAMP_420;
        break;

      case mPF_YUV422:
        tjSubSampling = TJSAMP_422;
        break;

      case mPF_YUV440:
        tjSubSampling = TJSAMP_440;
        break;

      case mPF_YUV444:
        tjSubSampling = TJSAMP_444;
        break;

      default:
        mFAIL_DEBUG("Invalid State.");
        goto fast_jpeg_encoder_failed;
      }

      uint8_t *pJpegBuffer = nullptr;
      unsigned long jpegBufferSize = 0;

      mDEFER(
        if (pJpegBuffer != nullptr)
          tjFree(pJpegBuffer);
      );

      const int32_t result = tjCompressFromYUV(encoder, imageBuffer->pPixels, (int32_t)imageBuffer->currentSize.x, 1, (int32_t)imageBuffer->currentSize.y, tjSubSampling, &pJpegBuffer, &jpegBufferSize, 85, TJFLAG_FASTDCT);

      if (result != 0)
        goto fast_jpeg_encoder_failed;

      mERROR_CHECK(mFile_WriteRaw(filename, pJpegBuffer, (size_t)jpegBufferSize));
      mRETURN_SUCCESS();

      break;
    }
    }
  }

fast_jpeg_encoder_failed:

  int32_t channels = 4;

  switch (imageBuffer->pixelFormat)
  {
  case mPF_R8G8B8A8:
    channels = 4;
    break;

  case mPF_R8G8B8:
    channels = 3;
    break;

  case mPF_Monochrome8:
    channels = 1;
    break;

  default:
    mRETURN_RESULT(mR_OperationNotSupported);
  }

  WriteFuncData fileWriteData;
  mERROR_CHECK(mFileWriter_Create(&fileWriteData.file, filename));

  const int result = stbi_write_jpg_to_func(WriteFuncData::Write, &fileWriteData, (int32_t)imageBuffer->currentSize.x, (int32_t)imageBuffer->currentSize.y, channels, imageBuffer->pPixels, 85);

  mERROR_IF(mFAILED(fileWriteData.result), fileWriteData.result);
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

  WriteFuncData fileWriteData;
  mERROR_CHECK(mFileWriter_Create(&fileWriteData.file, filename));

  const int result = stbi_write_bmp_to_func(WriteFuncData::Write, &fileWriteData, (int32_t)imageBuffer->currentSize.x, (int32_t)imageBuffer->currentSize.y, components, imageBuffer->pPixels);

  mERROR_IF(mFAILED(fileWriteData.result), fileWriteData.result);
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

  WriteFuncData fileWriteData;
  mERROR_CHECK(mFileWriter_Create(&fileWriteData.file, filename));

  const int result = stbi_write_tga_to_func(WriteFuncData::Write, &fileWriteData, (int32_t)imageBuffer->currentSize.x, (int32_t)imageBuffer->currentSize.y, components, imageBuffer->pPixels);

  mERROR_IF(mFAILED(fileWriteData.result), fileWriteData.result);
  mERROR_IF(result == 0, mR_InternalError);

  mRETURN_SUCCESS();
}

mFUNCTION(mImageBuffer_SaveAsRaw, mPtr<mImageBuffer> &imageBuffer, const mString &filename)
{
  mFUNCTION_SETUP();

  mERROR_IF(imageBuffer == nullptr || imageBuffer->pPixels == nullptr, mR_NotInitialized);

  size_t bytes = 0;
  mERROR_CHECK(mPixelFormat_GetSize(imageBuffer->pixelFormat, imageBuffer->currentSize, &bytes));

  mERROR_CHECK(mFile_WriteRaw(filename, imageBuffer->pPixels, bytes));

  mRETURN_SUCCESS();
}

mFUNCTION(mImageBuffer_SetToFile, mPtr<mImageBuffer> &imageBuffer, const mString &filename, const mPixelFormat pixelFormat /* = mPF_B8G8R8A8 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(imageBuffer == nullptr, mR_ArgumentNull);
  mERROR_IF(filename.hasFailed || filename.text == nullptr, mR_InvalidParameter);

  mPROFILE_SCOPED("mImageBuffer_SetToFile");

  bool fileExists = false;
  mERROR_CHECK(mFile_Exists(filename, &fileExists));
  mERROR_IF(!fileExists, mR_ResourceNotFound);

  uint8_t *pData = nullptr;
  size_t size = 0;
  mERROR_CHECK(mFile_ReadRaw(filename, &pData, &mDefaultTempAllocator, &size));
  mDEFER_CALL_2(mAllocator_FreePtr, &mDefaultTempAllocator, &pData);

  mERROR_CHECK(mImageBuffer_SetToData(imageBuffer, pData, size, pixelFormat));

  mRETURN_SUCCESS();
}

mFUNCTION(mImageBuffer_SetToData, mPtr<mImageBuffer> &imageBuffer, IN const uint8_t *pData, const size_t size, const mPixelFormat pixelFormat /* = mPF_B8G8R8A8 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(imageBuffer == nullptr || pData == nullptr, mR_ArgumentNull);
  mERROR_IF(size == 0, mR_InvalidParameter);

  int32_t components = 4;
  mPixelFormat readPixelFormat = mPF_R8G8B8A8;

  const bool tryJpeg = (size > 2 && pData[0] == 0xFF && pData[1] == 0xD8);

  if (tryJpeg)
  {
    static tjhandle decoder = tjInitDecompress();

    int32_t width, height, subSampling;

    if (decoder == nullptr)
      goto fast_jpeg_decoder_failed;

    if (0 != tjDecompressHeader2(decoder, const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(pData)), (uint32_t)size, &width, &height, &subSampling))
      goto fast_jpeg_decoder_failed;

    mPixelFormat sourcePixelFormat;

    switch (subSampling)
    {
    case TJSAMP_444:
      sourcePixelFormat = mPF_YUV444;
      break;

    case TJSAMP_422:
      sourcePixelFormat = mPF_YUV422;
      break;

    case TJSAMP_420:
      sourcePixelFormat = mPF_YUV420;
      break;

    case TJSAMP_440:
      sourcePixelFormat = mPF_YUV440;
      break;

    case TJSAMP_411:
      sourcePixelFormat = mPF_YUV411;
      break;

    case TJSAMP_GRAY:
      sourcePixelFormat = mPF_Monochrome8;
      break;

    default:
      goto fast_jpeg_decoder_failed;
    }

    if (pixelFormat == sourcePixelFormat)
    {
      if (mFAILED(mImageBuffer_AllocateBuffer(imageBuffer, mVec2s((size_t)width, (size_t)height), pixelFormat)))
        goto fast_jpeg_decoder_failed;

      if (0 != tjDecompressToYUV2(decoder, pData, (uint32_t)size, imageBuffer->pPixels, width, 1, height, TJFLAG_FASTDCT))
        goto fast_jpeg_decoder_failed;
    }
    else
    {
      bool isChromaSubSampled = false;
      mERROR_CHECK(mPixelFormat_IsChromaSubsampled(pixelFormat, &isChromaSubSampled));

      if (isChromaSubSampled)
      {
        bool canInplaceTransform = false;
        mERROR_CHECK(mPixelFormat_CanInplaceTransform(sourcePixelFormat, pixelFormat, &canInplaceTransform));

        if (canInplaceTransform)
        {
          if (mFAILED(mImageBuffer_AllocateBuffer(imageBuffer, mVec2s((size_t)width, (size_t)height), sourcePixelFormat)))
            goto fast_jpeg_decoder_failed;

          if (0 != tjDecompressToYUV2(decoder, pData, (uint32_t)size, imageBuffer->pPixels, width, 1, height, TJFLAG_FASTDCT))
            goto fast_jpeg_decoder_failed;

          mERROR_CHECK(mPixelFormat_InplaceTransformBuffer(imageBuffer, pixelFormat));
        }
        else
        {
          if (mFAILED(mImageBuffer_AllocateBuffer(imageBuffer, mVec2s((size_t)width, (size_t)height), pixelFormat)))
            goto fast_jpeg_decoder_failed;

          mPtr<mImageBuffer> tmp;

          if (mFAILED(mImageBuffer_Create(&tmp, &mDefaultTempAllocator, mVec2s((size_t)width, (size_t)height), sourcePixelFormat)))
            goto fast_jpeg_decoder_failed;

          if (0 != tjDecompressToYUV2(decoder, pData, (uint32_t)size, tmp->pPixels, width, 1, height, TJFLAG_FASTDCT))
            goto fast_jpeg_decoder_failed;

          mERROR_CHECK(mPixelFormat_TransformBuffer(tmp, imageBuffer));
        }
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
          goto fast_jpeg_decoder_failed;
          break;
        }

        if (mFAILED(mImageBuffer_AllocateBuffer(imageBuffer, mVec2s((size_t)width, (size_t)height), pixelFormat)) || mFAILED(mPixelFormat_GetUnitSize(pixelFormat, &pixelSize)))
          goto fast_jpeg_decoder_failed;

        if (0 != tjDecompress2(decoder, pData, (uint32_t)size, imageBuffer->pPixels, width, (int32_t)(width * pixelSize), height, tjPixelFormat, TJFLAG_FASTDCT))
          goto fast_jpeg_decoder_failed;
      }
    }

    mRETURN_SUCCESS();
  }

fast_jpeg_decoder_failed:

  const bool tryFPng = (size > 7 && pData[0] == 0x89 && pData[1] == 0x50 && pData[2] == 0x4E && pData[3] == 0x47 && pData[4] == 0x0D && pData[5] == 0x0A && pData[6] == 0x1A && pData[7] == 0x0A && size <= UINT32_MAX);

  if (tryFPng)
  {
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

    default:
      goto fpng_failed;
    }

    uint64_t requiredCapacity = 0;
    uint32_t width, height, channelsInFile, idatOffset, idatSize;

    int32_t result = fpng::fpng_decode_memory_get_required_capacity(pData, (uint32_t)size, requiredCapacity, width, height, channelsInFile, (uint32_t)components, idatOffset, idatSize);

    if (result != fpng::FPNG_DECODE_SUCCESS)
      goto fpng_failed;

    if (mFAILED(mImageBuffer_AllocateBuffer(imageBuffer, mVec2s((size_t)width, (size_t)height), pixelFormat)))
      goto fpng_failed;

    if (imageBuffer->allocatedSize < requiredCapacity)
      goto fpng_failed;

    result = fpng::fpng_decode_memory_ptr(pData, (uint32_t)size, imageBuffer->pPixels, requiredCapacity, width, height, channelsInFile, (uint32_t)components, idatOffset, idatSize);

    if (result != fpng::FPNG_DECODE_SUCCESS)
      goto fpng_failed;

    mRETURN_SUCCESS();
  }

fpng_failed:

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

  case mPF_YUV444:
  case mPF_YUV422:
  case mPF_YUV440:
  case mPF_YUV420:
  case mPF_YUV411:
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

  int32_t x, y, originalChannelCount;
  stbi_uc *pResult = stbi_load_from_memory(pData, (int32_t)size, &x, &y, &originalChannelCount, components);

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
    mERROR_CHECK(mMemcpy(imageBuffer->pPixels, (uint8_t *)pResult, imageBytes));
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
  mDEFER_CALL_2(mAllocator_FreePtr, nullptr, &pBuffer);
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
      mERROR_CHECK(mMemcpy(pBuffer, pSubBuffer, strideBytes));
      mERROR_CHECK(mMemcpy(pSubBuffer, pSubBufferEnd, strideBytes));
      mERROR_CHECK(mMemcpy(pSubBufferEnd, pBuffer, strideBytes));

      pSubBuffer += strideBytes;
      pSubBufferEnd -= strideBytes;
    }
  }

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

static mFUNCTION(mImageBuffer_Destroy_Iternal, mImageBuffer *pImageBuffer)
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

static mFUNCTION(mImageBuffer_Create_Iternal, mPtr<mImageBuffer> *pImageBuffer, IN OPTIONAL mAllocator *pAllocator)
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
