#ifndef mInstancedRenderDataBuffer_h__
#define mInstancedRenderDataBuffer_h__

#include "mRenderDataBuffer.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "yHtqJ6HuV+9zl9900d7SUg8ypi41HzYFySTwZ4xypYFPPfLGwli9Adpj5totvTWNd3P82zhix6OdsWRQ"
#endif

// Template Parameter 1 should be a `mRenderDataBuffer` of the model being instanced.
// Remaining Template Parameters should be `mDRBAttribute`s like `mRDB_FloatAttribute`.
template <typename T, typename... Args>
struct mInstancedRenderDataBuffer
{
  T renderDataBuffer;
  bool constantlyChangedInstanceData;
  bool validInstanceVBO;
  size_t instanceCount;

#if defined(mRENDERER_OPENGL)
  GLuint instanceVBO;
#endif
};

template <typename T, typename... Args>
mFUNCTION(mInstancedRenderDataBuffer_Create, IN mInstancedRenderDataBuffer<T, Args...> *pBuffer, const mPtr<mShader> &shader, bool constantlyChangingInstancedVertexData = false, bool constantlyChangingInstanceData = true);

template <typename T, typename... Args>
mFUNCTION(mInstancedRenderDataBuffer_Create, IN mInstancedRenderDataBuffer<T, Args...> *pBuffer, IN mAllocator *pAllocator, const mString &vertexShaderSource, const mString &fragmentShaderSource, bool constantlyChangingInstancedData = false, bool constantlyChangingInstanceData = true);

template <typename T, typename... Args>
mFUNCTION(mInstancedRenderDataBuffer_Destroy, IN mInstancedRenderDataBuffer<T, Args...> *pBuffer);

template <typename U, typename T, typename... Args>
mFUNCTION(mInstancedRenderDataBuffer_SetInstancedVertexBuffer, mInstancedRenderDataBuffer<T, Args...> &buffer, const U *pData, const size_t count);

template <typename T, typename... Args>
mFUNCTION(mInstancedRenderDataBuffer_SetInstanceBuffer, mInstancedRenderDataBuffer<T, Args...> &buffer, const uint32_t *pData, const size_t count);

template <typename T, typename... Args>
mFUNCTION(mInstancedRenderDataBuffer_SetInstancedVertexCount, mInstancedRenderDataBuffer<T, Args...> &buffer, const size_t count);

template <typename T, typename... Args>
mFUNCTION(mInstancedRenderDataBuffer_SetInstanceCount, mInstancedRenderDataBuffer<T, Args...> &buffer, const size_t count);

template <typename T, typename... Args>
mFUNCTION(mInstancedRenderDataBuffer_SetVertexRenderMode, mInstancedRenderDataBuffer<T, Args...> &buffer, const mRenderParams_VertexRenderMode renderMode);

template <typename T, typename... Args>
mFUNCTION(mInstancedRenderDataBuffer_Draw, mInstancedRenderDataBuffer<T, Args...> &buffer);

//////////////////////////////////////////////////////////////////////////

template<typename T, typename ...Args>
inline mFUNCTION(mInstancedRenderDataBuffer_Create, IN mInstancedRenderDataBuffer<T, Args...> *pBuffer, const mPtr<mShader> &shader, bool constantlyChangingInstancedVertexData /* = false */, bool constantlyChangingInstanceData /* = true */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pBuffer == nullptr || shader == nullptr, mR_ArgumentNull);

  pBuffer->validInstanceVBO = false;
  pBuffer->constantlyChangedInstanceData = constantlyChangingInstanceData;

  mERROR_CHECK(mRenderDataBuffer_Create(&pBuffer->renderDataBuffer, shader, constantlyChangingInstancedVertexData));

#if defined(mRENDERER_OPENGL)
  glGenBuffers(1, &pBuffer->instanceVBO);
#endif

  mGL_ERROR_CHECK();

  mRETURN_SUCCESS();
}

template<typename T, typename ...Args>
inline mFUNCTION(mInstancedRenderDataBuffer_Create, IN mInstancedRenderDataBuffer<T, Args...> *pBuffer, IN mAllocator *pAllocator, const mString &vertexShaderSource, const mString &fragmentShaderSource, bool constantlyChangingInstancedVertexData /* = false */, bool constantlyChangingInstanceData /* = true */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pBuffer == nullptr, mR_ArgumentNull);

  pBuffer->validInstanceVBO = false;
  pBuffer->constantlyChangedInstanceData = constantlyChangingInstanceData;

  mERROR_CHECK(mRenderDataBuffer_Create(&pBuffer->renderDataBuffer, pAllocator, vertexShaderSource, fragmentShaderSource, constantlyChangingInstancedVertexData));

#if defined(mRENDERER_OPENGL)
  glGenBuffers(1, &pBuffer->instanceVBO);
#endif

  mGL_ERROR_CHECK();

  mRETURN_SUCCESS();
}

template<typename T, typename ...Args>
inline mFUNCTION(mInstancedRenderDataBuffer_Destroy, IN mInstancedRenderDataBuffer<T, Args...> *pBuffer)
{
  mFUNCTION_SETUP();

  mGL_DEBUG_ERROR_CHECK();

#if defined(mRENDERER_OPENGL)
  if (pBuffer->instanceVBO != 0)
  {
    glDeleteBuffers(1, &pBuffer->instanceVBO);
    pBuffer->instanceVBO = 0;
  }
#endif

  mRenderDataBuffer_Destroy(&pBuffer->renderDataBuffer);

  mRETURN_SUCCESS();
}

template<typename U, typename T, typename ...Args>
inline mFUNCTION(mInstancedRenderDataBuffer_SetInstancedVertexBuffer, mInstancedRenderDataBuffer<T, Args...> &buffer, const U *pData, const size_t count)
{
  return mRenderDataBuffer_SetVertexBuffer(buffer.renderDataBuffer, pData, count);
}

template<typename U, typename T, typename ...Args>
inline mFUNCTION(mInstancedRenderDataBuffer_SetInstanceBuffer, mInstancedRenderDataBuffer<T, Args...> &buffer, const U *pData, const size_t count)
{
  mFUNCTION_SETUP();

  mERROR_IF(pData == nullptr, mR_ArgumentNull);

  const size_t singleBlockSize = mRDBAttributeQuery_Internal<Args...>::GetSize();

  mPROFILE_SCOPED("mInstancedRenderDataBuffer_SetInstanceBuffer");

#if defined(mRENDERER_OPENGL)
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ARRAY_BUFFER, buffer.instanceVBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(U) * count, pData, buffer.constantlyChangedInstanceData ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW);
#endif

  buffer.validInstanceVBO = true;
  buffer.instanceCount = (sizeof(U) * count) / singleBlockSize;

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}

template<typename T, typename ...Args>
inline mFUNCTION(mInstancedRenderDataBuffer_SetInstancedVertexCount, mInstancedRenderDataBuffer<T, Args...> &buffer, const size_t count)
{
  return mRenderDataBuffer_SetRenderCount(buffer.renderDataBuffer, count);
}

template<typename T, typename ...Args>
inline mFUNCTION(mInstancedRenderDataBuffer_SetInstanceCount, mInstancedRenderDataBuffer<T, Args...> &buffer, const size_t count)
{
  mFUNCTION_SETUP();

  buffer.instanceCount = count;

  mRETURN_SUCCESS();
}

template<typename T, typename ...Args>
inline mFUNCTION(mInstancedRenderDataBuffer_SetVertexRenderMode, mInstancedRenderDataBuffer<T, Args...> &buffer, const mRenderParams_VertexRenderMode renderMode)
{
  return mRenderDataBuffer_SetVertexRenderMode(buffer.renderDataBuffer, renderMode);
}

template<typename T, typename ...Args>
inline mFUNCTION(mInstancedRenderDataBuffer_Draw, mInstancedRenderDataBuffer<T, Args...> &buffer)
{
  mFUNCTION_SETUP();

  mERROR_IF(!buffer.renderDataBuffer.validVBO || !buffer.validInstanceVBO, mR_ResourceStateInvalid);

  mPROFILE_SCOPED("mInstancedRenderDataBuffer_Draw");

  mERROR_CHECK(mRenderDataBuffer_BindAttributes(buffer.renderDataBuffer));

#if defined(mRENDERER_OPENGL)
  glEnableVertexAttribArray(2);
  glBindBuffer(GL_ARRAY_BUFFER, buffer.instanceVBO);
#endif

  mERROR_CHECK((mIDBAttributeQuery_Internal_SetAttributes<Args...>(buffer.renderDataBuffer.shader.GetPointer(), true)));

#if defined(mRENDERER_OPENGL)
  glDrawArraysInstanced(buffer.renderDataBuffer.vertexRenderMode, 0, (GLsizei)buffer.renderDataBuffer.count, (GLsizei)buffer.instanceCount);
#endif

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}

#endif // mInstancedRenderDataBuffer_h__
