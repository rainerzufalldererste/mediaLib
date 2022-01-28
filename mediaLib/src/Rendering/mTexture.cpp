#include "mRenderParams.h"

#include "mTexture.h"
#include "mProfiler.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "sCCZHpYtbeA7/BRwiLFUKpvYDoIRwlST01JVXGbKsRYc0GVHhCxSXpPO5CjBdZ5Fc933OCVeG39gHCKI"
#endif

mFUNCTION(mTexture_Create, OUT mTexture *pTexture, mPtr<mImageBuffer> &imageBuffer, const bool upload /* = true */, const size_t textureUnit /* = 0 */, const mTexture2DParams &textureParams /* = mTexture2DParams() */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTexture == nullptr || imageBuffer == nullptr, mR_ArgumentNull);

  pTexture->uploadState = mRP_US_NotInitialized;
  pTexture->resolution = imageBuffer->currentSize;
  pTexture->resolutionF = (mVec2f)pTexture->resolution;
  pTexture->foreignTexture = false;
  pTexture->textureParams = textureParams;

#if defined(mRENDERER_OPENGL)
  pTexture->sampleCount = 0;

  mERROR_IF(textureUnit >= 32, mR_IndexOutOfBounds);
  pTexture->textureUnit = (GLuint)textureUnit;

  glActiveTexture(GL_TEXTURE0 + (GLuint)pTexture->textureUnit);
  glGenTextures(1, &pTexture->textureId);
  mDEFER_ON_ERROR(glDeleteTextures(1, &pTexture->textureId));

  glBindTexture(pTexture->sampleCount > 0 ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, pTexture->textureId);
  mERROR_CHECK(mTexture2DParams_ApplyToBoundTexture(textureParams, pTexture->sampleCount > 0));

  if (textureParams.minFilter >= mRP_TMagFM_NearestNeighborNearestMipMap && textureParams.minFilter <= mRP_TMagFM_BilinearInterpolationBlendMipMap)
    pTexture->isMipMapTexture = true;

#else
  mRETURN_RESULT(mR_NotInitialized);
#endif

  pTexture->imageBuffer = imageBuffer;
  pTexture->uploadState = mRP_US_NotUploaded;

  if (upload)
    mERROR_CHECK(mTexture_Upload(*pTexture));

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}

mFUNCTION(mTexture_Create, OUT mTexture *pTexture, const mString &filename, const bool upload /* = true */, const size_t textureUnit /* = 0 */, const mTexture2DParams &textureParams /* = mTexture2DParams() */)
{
  mFUNCTION_SETUP();

  mPtr<mImageBuffer> imageBuffer;
  mDEFER_CALL(&imageBuffer, mImageBuffer_Destroy);
  mERROR_CHECK(mImageBuffer_CreateFromFile(&imageBuffer, nullptr, filename, mPF_R8G8B8A8));

  mERROR_CHECK(mTexture_Create(pTexture, imageBuffer, upload, textureUnit, textureParams));

  mRETURN_SUCCESS();
}

mFUNCTION(mTexture_Create, OUT mTexture *pTexture, const uint8_t *pData, const mVec2s &size, const mPixelFormat pixelFormat /* = mPF_B8G8R8A8 */, const bool upload /* = true */, const size_t textureUnit /* = 0 */, const mTexture2DParams &textureParams /* = mTexture2DParams() */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTexture == nullptr || pData == nullptr, mR_ArgumentNull);

  if (upload)
  {
    pTexture->uploadState = mRP_US_NotInitialized;
    pTexture->resolution = size;
    pTexture->resolutionF = (mVec2f)pTexture->resolution;
    pTexture->foreignTexture = false;
    pTexture->textureParams = textureParams;

#if defined(mRENDERER_OPENGL)
    pTexture->sampleCount = 0;

    mERROR_IF(textureUnit >= 32, mR_IndexOutOfBounds);
    pTexture->textureUnit = (GLuint)textureUnit;

    glActiveTexture(GL_TEXTURE0 + (GLuint)pTexture->textureUnit);
    glGenTextures(1, &pTexture->textureId);
    mDEFER_ON_ERROR(glDeleteTextures(1, &pTexture->textureId));

    glBindTexture(pTexture->sampleCount > 0 ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, pTexture->textureId);
    mERROR_CHECK(mTexture2DParams_ApplyToBoundTexture(textureParams, pTexture->sampleCount > 0));

    if (textureParams.minFilter >= mRP_TMagFM_NearestNeighborNearestMipMap && textureParams.minFilter <= mRP_TMagFM_BilinearInterpolationBlendMipMap)
      pTexture->isMipMapTexture = true;

#else
    mRETURN_RESULT(mR_NotInitialized);
#endif

    mERROR_CHECK(mTexture_SetTo(*pTexture, pData, size, pixelFormat, true));
  }
  else
  {
    mPtr<mImageBuffer> imageBuffer;
    mDEFER_CALL(&imageBuffer, mImageBuffer_Destroy);
    mERROR_CHECK(mImageBuffer_Create(&imageBuffer, nullptr, (void *)pData, size, pixelFormat));

    mERROR_CHECK(mTexture_Create(pTexture, imageBuffer, upload, textureUnit, textureParams));
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mTexture_CreateFromUnownedIndex, OUT mTexture *pTexture, int textureIndex, const size_t textureUnit /* = 0 */, const size_t sampleCount /* = 0 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTexture == nullptr, mR_ArgumentNull);

  pTexture->uploadState = mRP_US_NotInitialized;
  pTexture->resolution = mVec2s(1);
  pTexture->resolutionF = (mVec2f)pTexture->resolution;
  pTexture->foreignTexture = true;

#if defined(mRENDERER_OPENGL)
  pTexture->sampleCount = sampleCount;

  mERROR_IF(textureUnit >= 32, mR_IndexOutOfBounds);
  pTexture->textureUnit = (GLuint)textureUnit;
  pTexture->textureId = textureIndex;
  
  mVec2t<GLint> resolution;

  glActiveTexture(GL_TEXTURE0 + (GLuint)pTexture->textureUnit);
  glBindTexture(pTexture->sampleCount > 0 ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, pTexture->textureId);
  glGetTexLevelParameteriv(pTexture->sampleCount > 0 ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &resolution.x);
  glGetTexLevelParameteriv(pTexture->sampleCount > 0 ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &resolution.y);

  pTexture->resolution = (mVec2s)resolution;
  pTexture->resolutionF = (mVec2f)resolution;
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  pTexture->uploadState = mRP_US_Ready;

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}

mFUNCTION(mTexture_Allocate, OUT mTexture *pTexture, const mVec2s size, const mPixelFormat pixelFormat /* = mPF_R8G8B8A8 */, const size_t textureUnit /* = 0 */, const mTexture2DParams &textureParams /* = mTexture2DParams() */)
{
  mFUNCTION_SETUP();

  pTexture->uploadState = mRP_US_NotInitialized;
  pTexture->resolution = size;
  pTexture->resolutionF = (mVec2f)pTexture->resolution;
  pTexture->foreignTexture = false;
  pTexture->textureParams = textureParams;

#if defined(mRENDERER_OPENGL)
  pTexture->sampleCount = 0;

  mERROR_IF(textureUnit >= 32, mR_IndexOutOfBounds);
  pTexture->textureUnit = (GLuint)textureUnit;

  glActiveTexture(GL_TEXTURE0 + (GLuint)pTexture->textureUnit);
  glGenTextures(1, &pTexture->textureId);
  mDEFER_ON_ERROR(glDeleteTextures(1, &pTexture->textureId));

  glBindTexture(pTexture->sampleCount > 0 ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, pTexture->textureId);

  GLenum glPixelFormat = GL_RGBA;
  mERROR_CHECK(mRenderParams_PixelFormatToGLenumChannels(pixelFormat, &glPixelFormat));

  GLenum glType = GL_UNSIGNED_BYTE;
  mERROR_CHECK(mRenderParams_PixelFormatToGLenumDataType(pixelFormat, &glType));

  glTexImage2D(pTexture->sampleCount > 0 ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, 0, glPixelFormat, (GLsizei)size.x, (GLsizei)size.y, 0, glPixelFormat, glType, nullptr);

  mGL_DEBUG_ERROR_CHECK();

  mERROR_CHECK(mTexture2DParams_ApplyToBoundTexture(textureParams, pTexture->sampleCount > 0));

  if (textureParams.minFilter >= mRP_TMagFM_NearestNeighborNearestMipMap && textureParams.minFilter <= mRP_TMagFM_BilinearInterpolationBlendMipMap)
    pTexture->isMipMapTexture = true;
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  pTexture->uploadState = mRP_US_Ready;

  mRETURN_SUCCESS();
}

mFUNCTION(mTexture_Destroy, IN_OUT mTexture *pTexture)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTexture == nullptr, mR_ArgumentNull);

  mGL_DEBUG_ERROR_CHECK();

  if (!pTexture->foreignTexture)
  {
#if defined(mRENDERER_OPENGL)
    if (pTexture->uploadState != mRP_US_NotInitialized)
      glDeleteTextures(1, &pTexture->textureId);
#endif
  }
  
#if defined(mRENDERER_OPENGL)
  pTexture->textureId = (GLuint)-1;
#endif

  if (pTexture->imageBuffer != nullptr)
    mERROR_CHECK(mImageBuffer_Destroy(&pTexture->imageBuffer));

  mGL_DEBUG_ERROR_CHECK();

  pTexture->uploadState = mRP_US_NotInitialized;

  mRETURN_SUCCESS();
}

mFUNCTION(mTexture_GetUploadState, mTexture &texture, OUT mRenderParams_UploadState *pUploadState)
{
  mFUNCTION_SETUP();

  mERROR_IF(pUploadState == nullptr, mR_ArgumentNull);

  *pUploadState = texture.uploadState;

  mRETURN_SUCCESS();
}

mFUNCTION(mTexture_Upload, mTexture &texture)
{
  mFUNCTION_SETUP();

  mERROR_IF(texture.textureId == 0, mR_NotInitialized);
  mERROR_IF(texture.uploadState == mRP_US_NotInitialized, mR_NotInitialized);

  if (texture.uploadState == mRP_US_Ready)
    mRETURN_SUCCESS();

  mERROR_IF(texture.imageBuffer == nullptr, mR_NotInitialized);

  mPROFILE_SCOPED("mTexture_Upload");

#if defined (mRENDERER_OPENGL)
#ifndef GIT_BUILD
  size_t bytes = 0;
  mERROR_CHECK(mPixelFormat_GetSize(texture.imageBuffer->pixelFormat, texture.imageBuffer->currentSize, &bytes));
  mASSERT((bytes & 3) == 0, "OpenGL expects this texture to be 4 byte aligned. mTexture currently doesn't do this.");
#endif
#endif

  mGL_DEBUG_ERROR_CHECK();

  texture.uploadState = mRP_US_Uploading;

#if defined (mRENDERER_OPENGL)
  glBindTexture(texture.sampleCount > 0 ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, texture.textureId);

  mGL_DEBUG_ERROR_CHECK();

  mERROR_CHECK(mTexture2DParams_ApplyToBoundTexture(texture.textureParams, texture.sampleCount > 0));

  if (texture.imageBuffer->pixelFormat == mPF_YUV420 || texture.imageBuffer->pixelFormat == mPF_YUV422 || texture.imageBuffer->pixelFormat == mPF_B8G8R8 || texture.imageBuffer->pixelFormat == mPF_B8G8R8A8)
  {
    mPtr<mImageBuffer> imageBuffer;
    mDEFER_CALL(&imageBuffer, mImageBuffer_Destroy);
    mERROR_CHECK(mImageBuffer_Create(&imageBuffer, &mDefaultTempAllocator, texture.imageBuffer->currentSize, mPF_R8G8B8A8));
    mERROR_CHECK(mPixelFormat_TransformBuffer(texture.imageBuffer, imageBuffer));

    glTexImage2D(texture.sampleCount > 0 ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, 0, GL_RGBA, (GLsizei)imageBuffer->currentSize.x, (GLsizei)imageBuffer->currentSize.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, imageBuffer->pPixels);
  }
  else
  {
    GLenum glPixelFormat = GL_RGBA;
    mERROR_CHECK(mRenderParams_PixelFormatToGLenumChannels(texture.imageBuffer->pixelFormat, &glPixelFormat));

    GLenum glType = GL_UNSIGNED_BYTE;
    mERROR_CHECK(mRenderParams_PixelFormatToGLenumDataType(texture.imageBuffer->pixelFormat, &glType));

    glTexImage2D(texture.sampleCount > 0 ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, 0, glPixelFormat, (GLsizei)texture.imageBuffer->currentSize.x, (GLsizei)texture.imageBuffer->currentSize.y, 0, glPixelFormat, glType, texture.imageBuffer->pPixels);
  }

  texture.resolution = texture.imageBuffer->currentSize;
  texture.resolutionF = (mVec2f)texture.resolution;

  if (texture.isMipMapTexture)
    glGenerateMipmap(texture.sampleCount > 0 ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D);

  mGL_DEBUG_ERROR_CHECK();

#else
  mRETURN_RESULT(mR_NotImplemented)
#endif

  mGL_DEBUG_ERROR_CHECK();

  texture.uploadState = mRP_US_Ready;

  mERROR_CHECK(mImageBuffer_Destroy(&texture.imageBuffer));

  mRETURN_SUCCESS();
}

mFUNCTION(mTexture_Bind, mTexture &texture, const size_t textureUnit /* = 0 */)
{
  mFUNCTION_SETUP();

  mPROFILE_SCOPED("mTexture_Bind");

  if (texture.uploadState != mRP_US_Ready)
    mERROR_CHECK(mTexture_Upload(texture));

  texture.textureUnit = (GLuint)textureUnit;

  glActiveTexture(GL_TEXTURE0 + texture.textureUnit);
  glBindTexture(texture.sampleCount > 0 ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, texture.textureId);

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}

mFUNCTION(mTexture_SetTo, mTexture &texture, mPtr<mImageBuffer> &imageBuffer, const bool upload /* = true */)
{
  mFUNCTION_SETUP();

  mERROR_IF(imageBuffer == nullptr, mR_ArgumentNull);

#if defined (mRENDERER_OPENGL)
#ifndef GIT_BUILD
  size_t bytes = 0;
  mERROR_CHECK(mPixelFormat_GetSize(imageBuffer->pixelFormat, imageBuffer->currentSize, &bytes));
  mASSERT((bytes & 3) == 0, "OpenGL expects this texture to be 4 byte aligned. mTexture currently doesn't do this.");
#endif
#endif

  texture.uploadState = mRP_US_NotInitialized;
  texture.resolution = imageBuffer->currentSize;
  texture.resolutionF = (mVec2f)texture.resolution;

#if defined(mRENDERER_OPENGL)
  glActiveTexture(GL_TEXTURE0 + (GLuint)texture.textureUnit);

  texture.imageBuffer = imageBuffer;
#else
  mRETURN_RESULT(mR_NotInitialized);
#endif

  texture.uploadState = mRP_US_NotUploaded;

  if (upload)
    mERROR_CHECK(mTexture_Upload(texture));

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}

mFUNCTION(mTexture_SetTo, mTexture &texture, const uint8_t *pData, const mVec2s &size, const mPixelFormat pixelFormat /* = mPF_B8G8R8A8 */, const bool upload /* = true */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pData == nullptr, mR_ArgumentNull);

  mPROFILE_SCOPED("mTexture_SetTo");

#if defined (mRENDERER_OPENGL)
#ifndef GIT_BUILD
  size_t bytes = 0;
  mERROR_CHECK(mPixelFormat_GetSize(pixelFormat, size, &bytes));
  mASSERT((bytes & 3) == 0, "OpenGL expects this texture to be 4 byte aligned. mTexture currently doesn't do this.");
#endif
#endif

  if (texture.textureId == 0)
  {
    mERROR_CHECK(mTexture_Create(&texture, pData, size, pixelFormat, upload));
    mRETURN_SUCCESS();
  }

  if (!upload)
  {
    mPtr<mImageBuffer> imageBuffer;
    mDEFER_CALL(&imageBuffer, mImageBuffer_Destroy);
    mERROR_CHECK(mImageBuffer_Create(&imageBuffer, &mDefaultAllocator, reinterpret_cast<const uint8_t *>(pData), size, pixelFormat));

    mERROR_CHECK(mTexture_SetTo(texture, imageBuffer, upload));
  }
  else
  {
    texture.uploadState = mRP_US_NotUploaded;

    mGL_DEBUG_ERROR_CHECK();

    texture.uploadState = mRP_US_Uploading;
    mDEFER_ON_ERROR(texture.uploadState = mRP_US_NotInitialized);

#if defined (mRENDERER_OPENGL)
    texture.sampleCount = 0;
    glBindTexture(texture.sampleCount > 0 ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, texture.textureId);

    mGL_DEBUG_ERROR_CHECK();

    if (pixelFormat == mPF_YUV420 || pixelFormat == mPF_YUV422 || pixelFormat == mPF_B8G8R8 || pixelFormat == mPF_B8G8R8A8)
    {
      mPtr<mImageBuffer> imageBuffer;
      mDEFER_CALL(&imageBuffer, mImageBuffer_Destroy);
      mERROR_CHECK(mImageBuffer_Create(&imageBuffer, &mDefaultTempAllocator, size, mPF_R8G8B8A8));
      mERROR_CHECK(mImageBuffer_AllocateBuffer(imageBuffer, size, mPF_R8G8B8A8));
      mERROR_CHECK(mPixelFormat_TransformBuffer(texture.imageBuffer, imageBuffer));

      glTexImage2D(texture.sampleCount > 0 ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, 0, GL_RGBA, (GLsizei)imageBuffer->currentSize.x, (GLsizei)imageBuffer->currentSize.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, imageBuffer->pPixels);
    }
    else
    {
      GLenum glPixelFormat = GL_RGBA;
      mERROR_CHECK(mRenderParams_PixelFormatToGLenumChannels(pixelFormat, &glPixelFormat));

      GLenum glType = GL_UNSIGNED_BYTE;
      mERROR_CHECK(mRenderParams_PixelFormatToGLenumDataType(pixelFormat, &glType));

      glTexImage2D(texture.sampleCount > 0 ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, 0, glPixelFormat, (GLsizei)size.x, (GLsizei)size.y, 0, glPixelFormat, glType, pData);
    }

    texture.resolution = size;
    texture.resolutionF = (mVec2f)texture.resolution;
    texture.uploadState = mRP_US_Ready;

    mGL_DEBUG_ERROR_CHECK();

#else
    mRETURN_RESULT(mR_NotImplemented)
#endif
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mTexture_Download, mTexture &texture, OUT mPtr<mImageBuffer> *pImageBuffer, IN mAllocator *pAllocator, const mPixelFormat pixelFormat /* = mPF_R8G8B8A8 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pImageBuffer == nullptr, mR_ArgumentNull);

  mPROFILE_SCOPED("mTexture_Download");

#if defined(mRENDERER_OPENGL)
  mERROR_IF(texture.sampleCount > 0, mR_NotSupported);

  if (texture.foreignTexture)
  {
    mVec2t<GLint> resolution;

    glBindTexture(texture.sampleCount > 0 ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, texture.textureId);
    glGetTexLevelParameteriv(texture.sampleCount > 0 ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &resolution.x);
    glGetTexLevelParameteriv(texture.sampleCount > 0 ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &resolution.y);

    texture.resolution = (mVec2s)resolution;
    texture.resolutionF = (mVec2f)resolution;
  }

  GLenum glPixelFormat = GL_RGBA;

  GLenum glType = GL_UNSIGNED_BYTE;
  mERROR_CHECK(mRenderParams_PixelFormatToGLenumDataType(pixelFormat, &glType));

  mPixelFormat internalPixelFormat = mPF_B8G8R8A8;

  switch (pixelFormat)
  {
  case mPF_B8G8R8:
  case mPF_R8G8B8:
  case mPF_R16G16B16:
  case mPF_Rf16Gf16Bf16:
  case mPF_Rf32Gf32Bf32:
    glPixelFormat = GL_RGB;
    internalPixelFormat = mPF_R8G8B8;
    break;

  case mPF_B8G8R8A8:
  case mPF_R8G8B8A8:
  case mPF_R16G16B16A16:
  case mPF_Rf16Gf16Bf16Af16:
  case mPF_Rf32Gf32Bf32Af32:
    glPixelFormat = GL_RGBA;
    internalPixelFormat = mPF_R8G8B8A8;
    break;

  case mPF_Monochrome8:
  case mPF_Monochrome16:
  case mPF_Monochromef16:
  case mPF_Monochromef32:
    glPixelFormat = GL_RED;
    internalPixelFormat = pixelFormat;
    break;

  case mPF_YUV444:
  case mPF_YUV440:
  case mPF_YUV422:
  case mPF_YUV420:
  case mPF_YUV411:
  default:
    mRETURN_RESULT(mR_InvalidParameter);
    break;
  }

  mPtr<mImageBuffer> imageBuffer;
  mERROR_CHECK(mImageBuffer_Create(&imageBuffer, internalPixelFormat == pixelFormat ? pAllocator : &mDefaultTempAllocator, texture.resolution, internalPixelFormat));

  glBindTexture(texture.sampleCount > 0 ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, texture.textureId);
  glGetTexImage(texture.sampleCount > 0 ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D, 0, glPixelFormat, glType, imageBuffer->pPixels);

  mERROR_CHECK(mImageBuffer_FlipY(imageBuffer));

  if (internalPixelFormat != pixelFormat)
  {
    mERROR_CHECK(mImageBuffer_Create(pImageBuffer, pAllocator, texture.resolution, pixelFormat));
    mERROR_CHECK(mPixelFormat_TransformBuffer(imageBuffer, *pImageBuffer));
  }
  else
  {
    *pImageBuffer = imageBuffer;
  }
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mCreateResource, OUT mTexture *pTexture, const mString &filename)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mTexture_Create(pTexture, filename, false));

  mRETURN_SUCCESS();
}

mFUNCTION(mDestroyResource, IN_OUT mTexture *pTexture)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mTexture_Destroy(pTexture));

  mRETURN_SUCCESS();
}

mFUNCTION(mDestruct, IN_OUT mTexture *pTexture)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mTexture_Destroy(pTexture));

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mTexture3D_Create, OUT mTexture3D *pTexture, const uint8_t *pData, const mVec3s &size, const mPixelFormat pixelFormat /* = mPF_B8G8R8A8 */, const size_t textureUnit /* = 0 */, const mTexture3DParams &textureParams /* = mTexture3DParams() */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTexture == nullptr || pData == nullptr, mR_ArgumentNull);

  pTexture->resolution = size;
  pTexture->resolutionF = mVec3f(size);
  pTexture->textureUnit = (GLuint)textureUnit;

  glGenTextures(1, &pTexture->textureId);
  mDEFER_ON_ERROR(glDeleteTextures(1, &pTexture->textureId));

  glBindTexture(GL_TEXTURE_3D, pTexture->textureId);

  mGL_DEBUG_ERROR_CHECK();

  mERROR_CHECK(mTexture3DParams_ApplyToBoundTexture(textureParams, false));

  GLenum glPixelFormat = GL_RGBA;
  mERROR_CHECK(mRenderParams_PixelFormatToGLenumChannels(pixelFormat, &glPixelFormat));

  GLenum glType = GL_UNSIGNED_BYTE;
  mERROR_CHECK(mRenderParams_PixelFormatToGLenumDataType(pixelFormat, &glType));

  glTexImage3D(GL_TEXTURE_3D, 0, glPixelFormat, (GLsizei)size.x, (GLsizei)size.y, (GLsizei)size.z, 0, glPixelFormat, glType, pData);

  mGL_ERROR_CHECK();

  pTexture->uploadState = mRP_US_Ready;

  mRETURN_SUCCESS();
};

mFUNCTION(mTexture3D_Allocate, OUT mTexture3D *pTexture, const mVec3s &size, const mPixelFormat pixelFormat /* = mPF_B8G8R8A8 */, const size_t textureUnit /* = 0 */, const mTexture3DParams &textureParams /* = mTexture3DParams() */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTexture == nullptr, mR_ArgumentNull);

  pTexture->resolution = size;
  pTexture->resolutionF = mVec3f(size);
  pTexture->textureUnit = (GLuint)textureUnit;

  glGenTextures(1, &pTexture->textureId);
  mDEFER_ON_ERROR(glDeleteTextures(1, &pTexture->textureId));

  glBindTexture(GL_TEXTURE_3D, pTexture->textureId);

  mGL_DEBUG_ERROR_CHECK();

  mERROR_CHECK(mTexture3DParams_ApplyToBoundTexture(textureParams, false));

  GLenum glPixelFormat = GL_RGBA;
  mERROR_CHECK(mRenderParams_PixelFormatToGLenumChannels(pixelFormat, &glPixelFormat));

  GLenum glType = GL_UNSIGNED_BYTE;
  mERROR_CHECK(mRenderParams_PixelFormatToGLenumDataType(pixelFormat, &glType));

  glTexImage3D(GL_TEXTURE_3D, 0, glPixelFormat, (GLsizei)size.x, (GLsizei)size.y, (GLsizei)size.z, 0, glPixelFormat, glType, nullptr);

  mGL_ERROR_CHECK();

  pTexture->uploadState = mRP_US_Ready;

  mRETURN_SUCCESS();
}

mFUNCTION(mTexture3D_Destroy, IN_OUT mTexture3D *pTexture)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTexture == nullptr, mR_ArgumentNull);
  
#if defined(mRENDERER_OPENGL)
  if (pTexture->uploadState != mRP_US_NotInitialized)
    glDeleteTextures(1, &pTexture->textureId);

  pTexture->textureId = (GLuint)-1;
#endif

  mGL_DEBUG_ERROR_CHECK();

  pTexture->uploadState = mRP_US_NotInitialized;

  mRETURN_SUCCESS();
}

mFUNCTION(mTexture3D_SetTo, mTexture3D &texture, const uint8_t *pData, const mVec3s &size, const mPixelFormat pixelFormat /* = mPF_B8G8R8A8 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pData == nullptr, mR_ArgumentNull);
  mERROR_IF(texture.uploadState == mRP_US_NotInitialized, mR_ResourceStateInvalid);

  mPROFILE_SCOPED("mTexture3D_SetTo");

  glBindTexture(GL_TEXTURE_3D, texture.textureId);

  GLenum glPixelFormat = GL_RGBA;
  mERROR_CHECK(mRenderParams_PixelFormatToGLenumChannels(pixelFormat, &glPixelFormat));

  GLenum glType = GL_UNSIGNED_BYTE;
  mERROR_CHECK(mRenderParams_PixelFormatToGLenumDataType(pixelFormat, &glType));

  glTexImage3D(GL_TEXTURE_3D, 0, glPixelFormat, (GLsizei)size.x, (GLsizei)size.y, (GLsizei)size.z, 0, glPixelFormat, glType, pData);

  mGL_ERROR_CHECK();

  texture.uploadState = mRP_US_Ready;

  mRETURN_SUCCESS();
}

mFUNCTION(mTexture3D_Bind, mTexture3D &texture, const size_t textureUnit /* = 0 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(texture.uploadState != mRP_US_Ready, mR_ResourceStateInvalid);

  mPROFILE_SCOPED("mTexture3D_Bind");

  texture.textureUnit = (GLuint)textureUnit;

  glActiveTexture(GL_TEXTURE0 + texture.textureUnit);
  glBindTexture(GL_TEXTURE_3D, texture.textureId);

  mRETURN_SUCCESS();
}

mFUNCTION(mDestruct, IN_OUT mTexture3D *pTexture)
{
  return mTexture3D_Destroy(pTexture);
}
