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
  else if (texture.imageBuffer->pixelFormat == mPF_B8G8R8A8)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_BGRA, (GLsizei)texture.imageBuffer->currentSize.x, (GLsizei)texture.imageBuffer->currentSize.y, 0, GL_BGRA, GL_UNSIGNED_BYTE, texture.imageBuffer->pPixels);
  else if (texture.imageBuffer->pixelFormat == mPF_B8G8R8)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_BGR, (GLsizei)texture.imageBuffer->currentSize.x, (GLsizei)texture.imageBuffer->currentSize.y, 0, GL_BGR, GL_UNSIGNED_BYTE, texture.imageBuffer->pPixels);
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
