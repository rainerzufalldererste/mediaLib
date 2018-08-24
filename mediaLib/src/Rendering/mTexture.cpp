// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "mRenderParams.h"
#include "mTexture.h"

mFUNCTION(mTexture_Create, OUT mTexture * pTexture, mPtr<mImageBuffer>& imageBuffer, const bool upload /* = true */, const size_t textureUnit /* = 0 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTexture == nullptr || imageBuffer == nullptr, mR_ArgumentNull);

  pTexture->uploadState = mRP_US_NotInitialized;
  pTexture->resolution = imageBuffer->currentSize;
  pTexture->resolutionF = (mVec2f)pTexture->resolution;

#if defined(mRENDERER_OPENGL)
  mERROR_IF(textureUnit >= 32, mR_IndexOutOfBounds);
  pTexture->textureUnit = (GLuint)textureUnit;

  glActiveTexture(GL_TEXTURE0 + (GLuint)pTexture->textureUnit);
  glGenTextures(1, &pTexture->textureId);

  pTexture->imageBuffer = imageBuffer;
#else
  mRETURN_RESULT(mR_NotInitialized);
#endif

  pTexture->uploadState = mRP_US_NotUploaded;

  if (upload)
    mERROR_CHECK(mTexture_Upload(*pTexture));

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}

mFUNCTION(mTexture_Create, OUT mTexture *pTexture, const std::string &filename, const bool upload /* = true */, const size_t textureUnit /* = 0 */)
{
  mFUNCTION_SETUP();

  mPtr<mImageBuffer> imageBuffer;
  mDEFER_DESTRUCTION(&imageBuffer, mImageBuffer_Destroy);
  mERROR_CHECK(mImageBuffer_CreateFromFile(&imageBuffer, nullptr, filename));

  mERROR_CHECK(mTexture_Create(pTexture, imageBuffer, upload, textureUnit));

  mRETURN_SUCCESS();
}

mFUNCTION(mTexture_Create, OUT mTexture *pTexture, const uint8_t *pData, const mVec2s &size, const mPixelFormat pixelFormat /* = mPF_B8G8R8A8 */, const bool upload /* = true */, const size_t textureUnit /* = 0 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTexture == nullptr || pData == nullptr, mR_ArgumentNull);

  mPtr<mImageBuffer> imageBuffer;
  mDEFER_DESTRUCTION(&imageBuffer, mImageBuffer_Destroy);

  if (upload)
  {
    mERROR_CHECK(mImageBuffer_Create(&imageBuffer, nullptr, (void *)pData, size, pixelFormat));
  }
  else
  {
    mERROR_CHECK(mImageBuffer_Create(&imageBuffer, nullptr, size, pixelFormat));

    size_t bufferSize;
    mERROR_CHECK(mPixelFormat_GetSize(pixelFormat, size, &bufferSize));

    mERROR_CHECK(mAllocator_Copy(nullptr, imageBuffer->pPixels, pData, bufferSize));
  }

  mERROR_CHECK(mTexture_Create(pTexture, imageBuffer, upload, textureUnit));

  mRETURN_SUCCESS();
}

mFUNCTION(mTexture_Destroy, IN_OUT mTexture * pTexture)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTexture == nullptr, mR_ArgumentNull);

#if defined(mRENDERER_OPENGL)
  if(pTexture->uploadState != mRP_US_NotInitialized)
    glDeleteTextures(1, &pTexture->textureId);

  pTexture->textureId = (GLuint)-1;
#endif

  if (pTexture->imageBuffer != nullptr)
    mERROR_CHECK(mImageBuffer_Destroy(&pTexture->imageBuffer));

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

  mERROR_IF(texture.uploadState == mRP_US_NotInitialized, mR_NotInitialized);

  if (texture.uploadState == mRP_US_Ready)
    mRETURN_SUCCESS();

  mERROR_IF(texture.imageBuffer == nullptr, mR_NotInitialized);

  texture.uploadState = mRP_US_Uploading;

#if defined (mRENDERER_OPENGL)
  glBindTexture(GL_TEXTURE_2D, texture.textureId);

  if (texture.imageBuffer->pixelFormat == mPF_R8G8B8A8)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, (GLsizei)texture.imageBuffer->currentSize.x, (GLsizei)texture.imageBuffer->currentSize.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture.imageBuffer->pPixels);
  else if (texture.imageBuffer->pixelFormat == mPF_R8G8B8)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, (GLsizei)texture.imageBuffer->currentSize.x, (GLsizei)texture.imageBuffer->currentSize.y, 0, GL_RGB, GL_UNSIGNED_BYTE, texture.imageBuffer->pPixels);
  else if (texture.imageBuffer->pixelFormat == mPF_Monochrome8)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, (GLsizei)texture.imageBuffer->currentSize.x, (GLsizei)texture.imageBuffer->currentSize.y, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, texture.imageBuffer->pPixels);
  else
  {
    mPtr<mImageBuffer> imageBuffer;
    mDEFER_DESTRUCTION(&imageBuffer, mImageBuffer_Destroy);
    mERROR_CHECK(mImageBuffer_Create(&imageBuffer, nullptr, texture.imageBuffer->currentSize, mPF_R8G8B8A8));
    mERROR_CHECK(mPixelFormat_TransformBuffer(texture.imageBuffer, imageBuffer));

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, (GLsizei)imageBuffer->currentSize.x, (GLsizei)imageBuffer->currentSize.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, imageBuffer->pPixels);
  }

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

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

  if (texture.uploadState != mRP_US_Ready)
    mERROR_CHECK(mTexture_Upload(texture));

  texture.textureUnit = (GLuint)textureUnit;

  glActiveTexture(GL_TEXTURE0 + texture.textureUnit);
  glBindTexture(GL_TEXTURE_2D, texture.textureId);

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}

mFUNCTION(mTexture_SetTo, mTexture &texture, mPtr<mImageBuffer> &imageBuffer, const bool upload /* = true */)
{
  mFUNCTION_SETUP();

  mERROR_IF(imageBuffer == nullptr, mR_ArgumentNull);

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

  mPtr<mImageBuffer> imageBuffer;
  mDEFER_DESTRUCTION(&imageBuffer, mImageBuffer_Destroy);
  mERROR_CHECK(mImageBuffer_Create(&imageBuffer, nullptr, (void *)pData, size, pixelFormat));

  mERROR_CHECK(mTexture_SetTo(texture, imageBuffer, upload));

  mRETURN_SUCCESS();
}

mFUNCTION(mCreateResource, OUT mTexture * pTexture, const std::string & filename)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mTexture_Create(pTexture, filename, false));

  mRETURN_SUCCESS();
}

mFUNCTION(mDestroyResource, IN_OUT mTexture * pTexture)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mTexture_Destroy(pTexture));

  mRETURN_SUCCESS();
}
