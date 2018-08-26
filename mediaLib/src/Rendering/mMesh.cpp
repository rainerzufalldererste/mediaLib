// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "mMesh.h"

mFUNCTION(mMeshAttributeContainer_Destroy, IN_OUT mPtr<mMeshAttributeContainer> *pMeshAttributeContainer)
{
  mFUNCTION_SETUP();

  mERROR_IF(pMeshAttributeContainer == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Destroy(pMeshAttributeContainer));

  mRETURN_SUCCESS();
}

mFUNCTION(mMeshAttributeContainer_Destroy_Internal, IN mMeshAttributeContainer *pMeshAttributeContainer)
{
  mFUNCTION_SETUP();

  mERROR_IF(pMeshAttributeContainer == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mBinaryChunk_Destroy(&pMeshAttributeContainer->attributes));

  mRETURN_SUCCESS();
}

mFUNCTION(mMesh_Create, OUT mPtr<mMesh> *pMesh, IN mAllocator *pAllocator, mPtr<mQueue<mMeshFactory_AttributeInformation>> &attributeInformation, mPtr<mShader>& shader, mPtr<mBinaryChunk>& data, mPtr<mQueue<mPtr<mTexture>>> &textures, const mRenderParams_VertexRenderMode triangleRenderMode /* = mRP_VRM_TriangleList */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pMesh == nullptr || attributeInformation == nullptr || shader == nullptr || data == nullptr || textures == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Allocate(pMesh, pAllocator, (std::function<void(mMesh *)>)[](mMesh *pData) { mMesh_Destroy_Internal(pData); }, 1));

  (*pMesh)->dataSize = 0;
  
  size_t attributeCount = 0;
  mERROR_CHECK(mQueue_GetCount(attributeInformation, &attributeCount));

  for (size_t i = 0; i < attributeCount; ++i)
  {
    mMeshFactory_AttributeInformation *pInfo;
    mERROR_CHECK(mQueue_PointerAt(attributeInformation, i, &pInfo));

    pInfo->offset = (*pMesh)->dataSize;
    (*pMesh)->dataSize += pInfo->size;
  }

  mERROR_IF((*pMesh)->dataSize == 0, mR_InvalidParameter);

  size_t completeDataSize = 0;
  mERROR_CHECK(mBinaryChunk_GetWriteBytes(data, &completeDataSize));

  (*pMesh)->primitiveCount = completeDataSize / (*pMesh)->dataSize;
  (*pMesh)->shader = shader;
  (*pMesh)->uploadState = mRenderParams_UploadState::mRP_US_NotInitialized;
  (*pMesh)->triangleRenderMode = triangleRenderMode;
  (*pMesh)->information = attributeInformation;

  size_t textureCount = 0;
  mERROR_CHECK(mQueue_GetCount(textures, &textureCount));

  mERROR_CHECK(mArray_Create(&(*pMesh)->textures, pAllocator, textureCount));

  for (size_t i = 0; i < textureCount; ++i)
  {
    mPtr<mTexture> *pTexture;
    mERROR_CHECK(mQueue_PointerAt(textures, i, &pTexture));
    mERROR_CHECK(mArray_PutAt((*pMesh)->textures, i, pTexture));
  }

#if defined (mRENDERER_OPENGL)
  (*pMesh)->hasVbo = true;

  glGenBuffers(1, &(*pMesh)->vbo);
  glBindBuffer(GL_ARRAY_BUFFER, (*pMesh)->vbo);
  glBufferData(GL_ARRAY_BUFFER, data->writeBytes, data->pData, GL_STATIC_DRAW);

  GLuint index = 0;

  for (size_t i = 0; i < attributeCount; ++i)
  {
    mMeshFactory_AttributeInformation info;
    mERROR_CHECK(mQueue_PeekAt((*pMesh)->information, i, &info));

    if (info.size > 0 && info.type == mMF_AIT_Attribute)
    {
      glEnableVertexAttribArray((GLuint)index);
      glVertexAttribPointer((GLuint)index, (GLint)(info.size / info.individualSubTypeSize), info.dataType, GL_FALSE, (GLsizei)(*pMesh)->dataSize, (const void *)info.offset);
      ++index;
    }

    mGL_DEBUG_ERROR_CHECK();
  }

  if ((*pMesh)->textures.count > 0)
    (*pMesh)->uploadState = mRP_US_NotUploaded;
  else
    (*pMesh)->uploadState = mRP_US_Ready;

#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

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
  mERROR_CHECK(mQueue_Destroy(&pMesh->information));
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

  mERROR_CHECK(mShader_Bind(*data->shader.GetPointer()));

  mGL_DEBUG_ERROR_CHECK();

  if (data->hasVbo)
    glBindBuffer(GL_ARRAY_BUFFER, data->vbo);

  size_t informationCount;
  mERROR_CHECK(mQueue_GetCount(data->information, &informationCount));

  GLuint index = 0;

  for (size_t i = 0; i < informationCount; ++i)
  {
    mMeshFactory_AttributeInformation info;
    mERROR_CHECK(mQueue_PeekAt(data->information, i, &info));

    if (info.size > 0 && info.type == mMF_AIT_Attribute)
    {
      glEnableVertexAttribArray((GLuint)index);
      glVertexAttribPointer((GLuint)index, (GLint)(info.size / info.individualSubTypeSize), info.dataType, GL_FALSE, (GLsizei)data->dataSize, (const void *)info.offset);
      ++index;
    }

    mGL_DEBUG_ERROR_CHECK();
  }

  switch (data->triangleRenderMode)
  {
  case mRP_VRM_Points:
    glDrawArrays(GL_POINTS, 0, (GLuint)data->primitiveCount);
    break;

  case mRP_VRM_LineList:
    glDrawArrays(GL_LINES, 0, (GLuint)data->primitiveCount);
    break;

  case mRP_VRM_LineStrip:
    glDrawArrays(GL_LINE_STRIP, 0, (GLuint)data->primitiveCount);
    break;

  case mRP_VRM_LineLoop:
    glDrawArrays(GL_LINE_LOOP, 0, (GLuint)data->primitiveCount);
    break;

  case mRP_VRM_TriangleList:
    glDrawArrays(GL_TRIANGLES, 0, (GLuint)data->primitiveCount);
    break;

  case mRP_VRM_TriangleStrip:
    glDrawArrays(GL_TRIANGLE_STRIP, 0, (GLuint)data->primitiveCount);
    break;

  case mRP_VRM_TriangleFan:
    glDrawArrays(GL_TRIANGLE_FAN, 0, (GLuint)data->primitiveCount);
    break;

  case mRP_VRM_QuadList:
    glDrawArrays(GL_QUADS, 0, (GLuint)data->primitiveCount);
    break;

  case mRP_VRM_QuadStrip:
    glDrawArrays(GL_QUAD_STRIP, 0, (GLuint)data->primitiveCount);
    break;

  case mRP_VRM_Polygon:
    glDrawArrays(GL_POLYGON, 0, (GLuint)data->primitiveCount);
    break;

  default:
    mRETURN_RESULT(mR_OperationNotSupported);
    break;
  }

  glBindBuffer(GL_ARRAY_BUFFER, 0);

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
