// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "mMesh.h"

mFUNCTION(mMesh_Destroy, IN_OUT mPtr<mMesh> *pMesh)
{
  mFUNCTION_SETUP();

  mERROR_IF(pMesh == nullptr, mR_ArgumentNull);
  mERROR_CHECK(mSharedPointer_Destroy(pMesh));

  mRETURN_SUCCESS();
}

mFUNCTION(mMesh_Destroy_Internal, mMesh * pMesh)
{
  mFUNCTION_SETUP();

  mERROR_IF(pMesh == nullptr, mR_ArgumentNull);

#if defined (mRENDERER_OPENGL)
  if (pMesh->hasVbo)
  {
    glDeleteBuffers(1, &pMesh->vbo);
    pMesh->hasVbo = false;
  }

  for (size_t i = 0; i < pMesh->textures.count; ++i)
  {
    mPtr<mTexture> texture;
    mERROR_CHECK(mArray_PopAt(pMesh->textures, i, &texture));
    mERROR_CHECK(mSharedPointer_Destroy(&texture));
  }

  mERROR_CHECK(mArray_Destroy(&pMesh->textures));
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mMesh_Upload, mPtr<mMesh> &data)
{
  mFUNCTION_SETUP();

  mERROR_IF(data == nullptr, mR_ArgumentNull);
  mERROR_IF(data->uploadState == mRP_US_NotInitialized, mR_NotInitialized);

  if (data->uploadState == mRP_US_Ready)
    mRETURN_SUCCESS();

#if defined (mRENDERER_OPENGL)

  for (size_t i = 0; i < data->textures.count; ++i)
  {
    mPtr<mTexture> texture;
    mDEFER_DESTRUCTION(&texture, mSharedPointer_Destroy);
    mERROR_CHECK(mArray_PeekAt(data->textures, i, &texture));
    mERROR_CHECK(mTexture_Upload(*texture.GetPointer()));
  }

#else
  mRETURN_SUCCESS(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mMesh_GetUploadState, mPtr<mMesh>& data, OUT mRenderParams_UploadState * pUploadState)
{
  mFUNCTION_SETUP();

  mERROR_IF(data == nullptr || pUploadState == nullptr, mR_ArgumentNull);

  *pUploadState = data->uploadState;

  mRETURN_SUCCESS();
}

mFUNCTION(mMesh_Render, mPtr<mMesh>& data)
{
  mFUNCTION_SETUP();

  mERROR_IF(data == nullptr, mR_ArgumentNull);

  if (data->uploadState != mRP_US_Ready)
    mERROR_CHECK(mMesh_Upload(data));

#if defined (mRENDERER_OPENGL)

  for (size_t i = 0; i < data->textures.count; ++i)
  {
    mPtr<mTexture> texture;
    mDEFER_DESTRUCTION(&texture, mSharedPointer_Destroy);
    mERROR_CHECK(mArray_PeekAt(data->textures, i, &texture));
    mERROR_CHECK(mTexture_Bind(*texture.GetPointer(), i));
  }

  if (data->hasVbo)
    glBindBuffer(GL_ARRAY_BUFFER, data->vbo);

  switch (data->triangleRenderMode)
  {
  case mRP_RM_TriangleList:
    glDrawArrays(GL_TRIANGLES, 0, (GLuint)data->primitiveCount);
    break;

  case mRP_RM_TriangleStrip:
    glDrawArrays(GL_TRIANGLE_STRIP, 0, (GLuint)data->primitiveCount);
    break;

  case mRP_RM_TriangleFan:
    glDrawArrays(GL_TRIANGLE_FAN, 0, (GLuint)data->primitiveCount);
    break;

  default:
    mRETURN_RESULT(mR_OperationNotSupported);
    break;
  }

#else
  mRETURN_SUCCESS(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mMesh_Render, mPtr<mMesh>& data, mMatrix & matrix)
{
  mFUNCTION_SETUP();

  mERROR_IF(data == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mShader_SetUniform(data->shader, "matrix0", matrix));

  mERROR_CHECK(mMesh_Render(data));

  mRETURN_SUCCESS();
}
