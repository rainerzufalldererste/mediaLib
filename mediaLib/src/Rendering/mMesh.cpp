#include "mMesh.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "/CBF/D8lFrLrm9/RVC0gtajh+bAXvLQfRpUUyzYnnEMme1rJwnplOmFFpOTWoNpyYCX+32ySNlD04zCq"
#endif

mFUNCTION(mMeshAttributeContainer_Destroy, IN_OUT mPtr<mMeshAttributeContainer> *pMeshAttributeContainer)
{
  mFUNCTION_SETUP();

  mERROR_IF(pMeshAttributeContainer == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Destroy(pMeshAttributeContainer));

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mMeshAttributeContainer_Destroy_Internal, IN mMeshAttributeContainer *pMeshAttributeContainer)
{
  mFUNCTION_SETUP();

  mERROR_IF(pMeshAttributeContainer == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mBinaryChunk_Destroy(&pMeshAttributeContainer->attributes));

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mMesh_Create, OUT mPtr<mMesh> *pMesh, IN mAllocator *pAllocator, const std::initializer_list<mPtr<mMeshAttributeContainer>> &attributeInformation, mPtr<mShader> &shader, mPtr<mQueue<mKeyValuePair<mString, mPtr<mTexture>>>> &textures, const mRenderParams_VertexRenderMode triangleRenderMode /* = mRP_VRM_TriangleList */)
{
  mFUNCTION_SETUP();

  mPtr<mQueue<mPtr<mMeshAttributeContainer>>> queue;
  mDEFER_CALL(&queue, mQueue_Destroy);
  mERROR_CHECK(mQueue_Create(&queue, pAllocator));
  mERROR_CHECK(mQueue_Reserve(queue, attributeInformation.size()));

  for (const mPtr<mMeshAttributeContainer> &item : attributeInformation)
    mERROR_CHECK(mQueue_PushBack(queue, item));

  mERROR_CHECK(mMesh_Create(pMesh, pAllocator, queue, shader, textures, triangleRenderMode));

  mRETURN_SUCCESS();
}

mFUNCTION(mMesh_Create, OUT mPtr<mMesh> *pMesh, IN mAllocator *pAllocator, mPtr<mQueue<mPtr<mMeshAttributeContainer>>> &attributeInformation, mPtr<mShader> &shader, mPtr<mQueue<mKeyValuePair<mString, mPtr<mTexture>>>> &textures, const mRenderParams_VertexRenderMode triangleRenderMode /* = mRP_VRM_TriangleList */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pMesh == nullptr || attributeInformation == nullptr || shader == nullptr || textures == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Allocate(pMesh, pAllocator, (std::function<void(mMesh *)>)[](mMesh *pData) { mMesh_Destroy_Internal(pData); }, 1));

  size_t attributeCount = 0;
  mERROR_CHECK(mQueue_GetCount(attributeInformation, &attributeCount));

  mPtr<mQueue<mMeshAttribute>> info;
  mDEFER_CALL(&info, mQueue_Destroy);
  mERROR_CHECK(mQueue_Create(&info, pAllocator));

#if defined (mRENDERER_OPENGL)

  mERROR_CHECK(mShader_Bind(*shader.GetPointer()));

  size_t count = 0;
  size_t offset = 0;

  for (size_t i = 0; i < attributeCount; ++i)
  {
    mPtr<mMeshAttributeContainer> meshAttrib;
    mDEFER_CALL(&meshAttrib, mMeshAttributeContainer_Destroy);
    mERROR_CHECK(mQueue_PeekAt(attributeInformation, i, &meshAttrib));

    if (i == 0)
      count = meshAttrib->attributeCount;
    else
      mERROR_IF(count != meshAttrib->attributeCount, mR_InvalidParameter);

    const GLint location = glGetAttribLocation(shader->shaderProgram, meshAttrib->attributeName);

    mERROR_CHECK(mQueue_PushBack(info, mMeshAttribute(location, meshAttrib->size, meshAttrib->subComponentSize, meshAttrib->attributeCount, offset, meshAttrib->dataType)));

    offset += meshAttrib->size;
  }

  mPtr<mBinaryChunk> data;
  mDEFER_CALL(&data, mBinaryChunk_Destroy);
  mERROR_CHECK(mBinaryChunk_Create(&data, pAllocator));

  for (size_t i = 0; i < count; ++i)
  {
    for (size_t j = 0; j < attributeCount; ++j)
    {
      mPtr<mMeshAttributeContainer> meshAttrib;
      mDEFER_CALL(&meshAttrib, mMeshAttributeContainer_Destroy);
      mERROR_CHECK(mQueue_PeekAt(attributeInformation, j, &meshAttrib));
      
      mERROR_CHECK(mBinaryChunk_WriteBytes(data, meshAttrib->attributes->pData + meshAttrib->size * i, meshAttrib->size));
    }
  }

  (*pMesh)->primitiveCount = count;
  (*pMesh)->dataSize = offset;
  (*pMesh)->information = info;
  (*pMesh)->shader = shader;
  (*pMesh)->uploadState = mRenderParams_UploadState::mRP_US_NotInitialized;
  (*pMesh)->triangleRenderMode = triangleRenderMode;
  (*pMesh)->textures = textures;

  (*pMesh)->hasVbo = true;
  glGenBuffers(1, &(*pMesh)->vbo);
  mGL_ERROR_CHECK();
  mSTDRESULT = mR_Success;
  glBindBuffer(GL_ARRAY_BUFFER, (*pMesh)->vbo);
  glBufferData(GL_ARRAY_BUFFER, data->writeBytes, data->pData, GL_STATIC_DRAW);

  for (size_t i = 0; i < attributeCount; ++i)
  {
    mMeshAttribute attrInfo;
    mERROR_CHECK(mQueue_PeekAt((*pMesh)->information, i, &attrInfo));

    if (attrInfo.dataEntrySize > 0)
    {
      glEnableVertexAttribArray((GLuint)attrInfo.attributeIndex);
      glVertexAttribPointer((GLuint)attrInfo.attributeIndex, (GLint)(attrInfo.dataEntrySize / attrInfo.dataEntrySubComponentSize), attrInfo.dataType, GL_FALSE, (GLsizei)(*pMesh)->dataSize, (const void *)attrInfo.offset);
    }

    mGL_DEBUG_ERROR_CHECK();
  }

  size_t textureCount = 0;
  mERROR_CHECK(mQueue_GetCount(textures, &textureCount));

  if (textureCount > 0)
    (*pMesh)->uploadState = mRP_US_NotUploaded;
  else
    (*pMesh)->uploadState = mRP_US_Ready;

#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mMesh_Create, OUT mPtr<mMesh> *pMesh, IN mAllocator *pAllocator, mPtr<mQueue<mMeshAttribute>> &attributeInformation, mPtr<mShader> &shader, mPtr<mBinaryChunk> &data, mPtr<mQueue<mKeyValuePair<mString, mPtr<mTexture>>>> &textures, const mRenderParams_VertexRenderMode triangleRenderMode /* = mRP_VRM_TriangleList */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pMesh == nullptr || attributeInformation == nullptr || shader == nullptr || data == nullptr || textures == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Allocate(pMesh, pAllocator, (std::function<void(mMesh *)>)[](mMesh *pData) { mMesh_Destroy_Internal(pData); }, 1));

  (*pMesh)->dataSize = 0;
  
  size_t attributeCount = 0;
  mERROR_CHECK(mQueue_GetCount(attributeInformation, &attributeCount));

  for (size_t i = 0; i < attributeCount; ++i)
  {
    mMeshAttribute *pInfo;
    mERROR_CHECK(mQueue_PointerAt(attributeInformation, i, &pInfo));

    pInfo->offset = (*pMesh)->dataSize;
    (*pMesh)->dataSize += pInfo->dataEntrySize;
  }

  mERROR_IF((*pMesh)->dataSize == 0, mR_InvalidParameter);

  size_t completeDataSize = 0;
  mERROR_CHECK(mBinaryChunk_GetWriteBytes(data, &completeDataSize));

  (*pMesh)->primitiveCount = completeDataSize / (*pMesh)->dataSize;
  (*pMesh)->shader = shader;
  (*pMesh)->uploadState = mRenderParams_UploadState::mRP_US_NotInitialized;
  (*pMesh)->triangleRenderMode = triangleRenderMode;
  (*pMesh)->information = attributeInformation;
  (*pMesh)->textures = textures;

#if defined (mRENDERER_OPENGL)
  (*pMesh)->hasVbo = true;

  glGenBuffers(1, &(*pMesh)->vbo);
  glBindBuffer(GL_ARRAY_BUFFER, (*pMesh)->vbo);
  glBufferData(GL_ARRAY_BUFFER, data->writeBytes, data->pData, GL_STATIC_DRAW);

  for (size_t i = 0; i < attributeCount; ++i)
  {
    mMeshAttribute info;
    mERROR_CHECK(mQueue_PeekAt((*pMesh)->information, i, &info));

    if (info.dataEntrySize > 0)
    {
      glEnableVertexAttribArray((GLuint)info.attributeIndex);
      glVertexAttribPointer((GLuint)info.attributeIndex, (GLint)(info.dataEntrySize / info.dataEntrySubComponentSize), info.dataType, GL_FALSE, (GLsizei)(*pMesh)->dataSize, (const void *)info.offset);
    }

    mGL_DEBUG_ERROR_CHECK();
  }

  size_t textureCount = 0;
  mERROR_CHECK(mQueue_GetCount(textures, &textureCount));

  if (textureCount > 0)
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

mFUNCTION(mMesh_Upload, mPtr<mMesh> &data)
{
  mFUNCTION_SETUP();

  mERROR_IF(data == nullptr, mR_ArgumentNull);
  mERROR_IF(data->uploadState == mRP_US_NotInitialized, mR_NotInitialized);

  if (data->uploadState == mRP_US_Ready)
    mRETURN_SUCCESS();

#if defined (mRENDERER_OPENGL)

  size_t textureCount;
  mERROR_CHECK(mQueue_GetCount(data->textures, &textureCount));

  for (size_t i = 0; i < textureCount; ++i)
  {
    mKeyValuePair<mString, mPtr<mTexture>> *pTexture;
    mERROR_CHECK(mQueue_PointerAt(data->textures, i, &pTexture));
    mERROR_CHECK(mTexture_Upload(*pTexture->value));
  }

#else
  mRETURN_SUCCESS(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mMesh_GetUploadState, mPtr<mMesh> &data, OUT mRenderParams_UploadState *pUploadState)
{
  mFUNCTION_SETUP();

  mERROR_IF(data == nullptr || pUploadState == nullptr, mR_ArgumentNull);

  *pUploadState = data->uploadState;

  mRETURN_SUCCESS();
}

mFUNCTION(mMesh_Render, mPtr<mMesh> &data)
{
  mFUNCTION_SETUP();

  mERROR_IF(data == nullptr, mR_ArgumentNull);

  if (data->uploadState != mRP_US_Ready)
    mERROR_CHECK(mMesh_Upload(data));

#if defined (mRENDERER_OPENGL)

  size_t textureCount;
  mERROR_CHECK(mQueue_GetCount(data->textures, &textureCount));

  for (size_t i = 0; i < textureCount; ++i)
  {
    mKeyValuePair<mString, mPtr<mTexture>> *pTexture;
    mERROR_CHECK(mQueue_PointerAt(data->textures, i, &pTexture));
    mERROR_CHECK(mTexture_Bind(*pTexture->value, i));
    mERROR_CHECK(mShader_SetUniform(*data->shader, pTexture->key.c_str(), *pTexture->value));
  }

  mERROR_CHECK(mShader_Bind(*data->shader));

  mGL_DEBUG_ERROR_CHECK();

  if (data->hasVbo)
    glBindBuffer(GL_ARRAY_BUFFER, data->vbo);

  size_t informationCount;
  mERROR_CHECK(mQueue_GetCount(data->information, &informationCount));

  for (size_t i = 0; i < informationCount; ++i)
  {
    mMeshAttribute info;
    mERROR_CHECK(mQueue_PeekAt(data->information, i, &info));

    if (info.dataEntrySize > 0)
    {
      glEnableVertexAttribArray((GLuint)info.attributeIndex);
      glVertexAttribPointer((GLuint)info.attributeIndex, (GLint)(info.dataEntrySize / info.dataEntrySubComponentSize), info.dataType, GL_FALSE, (GLsizei)data->dataSize, (const void *)info.offset);
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

mFUNCTION(mMesh_Render, mPtr<mMesh> &data, mMatrix &matrix)
{
  mFUNCTION_SETUP();

  mERROR_IF(data == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mShader_SetUniform(data->shader, "matrix0", matrix));

  mERROR_CHECK(mMesh_Render(data));

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mMesh_Destroy_Internal, mMesh *pMesh)
{
  mFUNCTION_SETUP();

  mERROR_IF(pMesh == nullptr, mR_ArgumentNull);

#if defined (mRENDERER_OPENGL)
  if (pMesh->hasVbo)
  {
    glDeleteBuffers(1, &pMesh->vbo);
    pMesh->hasVbo = false;
  }

  size_t textureCount;
  mERROR_CHECK(mQueue_GetCount(pMesh->textures, &textureCount));

  for (size_t i = 0; i < textureCount; ++i)
  {
    mKeyValuePair<mString, mPtr<mTexture>> texture;
    mERROR_CHECK(mQueue_PopFront(pMesh->textures, &texture));
    mERROR_CHECK(mDestruct(&texture));
  }

  mERROR_CHECK(mQueue_Destroy(&pMesh->textures));
  mERROR_CHECK(mQueue_Destroy(&pMesh->information));
#endif

  mRETURN_SUCCESS();
}
