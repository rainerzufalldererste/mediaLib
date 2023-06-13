#ifndef mIndexedRenderDataBuffer_h__
#define mIndexedRenderDataBuffer_h__

#include "mRenderDataBuffer.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "2HXrz0PQ89Q0a7uGkEErw0pfCzGh3IHGh5yUi6BIOHFy9h3uOr7Z+DBab5SoCeskjHZiWJMGhYIBaaiQ"
#endif

// Template Parameters should be `mDRBAttribute`s like `mRDB_FloatAttribute`.
template <typename... Args>
struct mIndexedRenderDataBuffer : mRenderDataBuffer<Args...>
{
  bool constantlyChangedIndices;
  bool validIBO;

#if defined(mRENDERER_OPENGL)
  GLuint ibo;
#endif
};

template <typename... Args>
mFUNCTION(mIndexedRenderDataBuffer_Create, IN mIndexedRenderDataBuffer<Args...> *pBuffer, const mPtr<mShader> &shader, bool constantlyChangedVertices = false, bool constantlyChangedIndices = false);

template <typename... Args>
mFUNCTION(mIndexedRenderDataBuffer_Create, IN mIndexedRenderDataBuffer<Args...> *pBuffer, IN mAllocator *pAllocator, const mString &vertexShaderSource, const mString &fragmentShaderSource, bool constantlyChangedVertices = false, bool constantlyChangedIndices = false);

template <typename... Args>
mFUNCTION(mIndexedRenderDataBuffer_Destroy, IN mIndexedRenderDataBuffer<Args...> *pBuffer);

template <typename U, typename... Args>
mFUNCTION(mIndexedRenderDataBuffer_SetVertexBuffer, mIndexedRenderDataBuffer<Args...> &buffer, const U *pData, const size_t count);

template <typename... Args>
mFUNCTION(mIndexedRenderDataBuffer_SetIndexBuffer, mIndexedRenderDataBuffer<Args...> &buffer, const uint32_t *pData, const size_t count);

template <typename... Args>
mFUNCTION(mIndexedRenderDataBuffer_SetRenderCount, mIndexedRenderDataBuffer<Args...> &buffer, const size_t count);

template <typename... Args>
mFUNCTION(mIndexedRenderDataBuffer_SetVertexRenderMode, mIndexedRenderDataBuffer<Args...> &buffer, const mRenderParams_VertexRenderMode renderMode);

template <typename... Args>
mFUNCTION(mIndexedRenderDataBuffer_Draw, mIndexedRenderDataBuffer<Args...> &buffer);

//////////////////////////////////////////////////////////////////////////

template<typename ...Args>
inline mFUNCTION(mIndexedRenderDataBuffer_Create, IN mIndexedRenderDataBuffer<Args...> *pBuffer, const mPtr<mShader> &shader, bool constantlyChangedVertices /* = false */, bool constantlyChangedIndices /* = false */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pBuffer == nullptr || shader == nullptr, mR_ArgumentNull);

  pBuffer->validIBO = pBuffer->validVBO = false; // Upload buffer data to validate buffer.
  pBuffer->shader = shader;
  pBuffer->constantlyChangedVertices = constantlyChangedVertices;
  pBuffer->constantlyChangedIndices = constantlyChangedIndices;

#if defined(mRENDERER_OPENGL)
  glGenVertexArrays(1, &pBuffer->vao);
  glBindVertexArray(pBuffer->vao);

  glGenBuffers(1, &pBuffer->ibo);
  glGenBuffers(1, &pBuffer->vbo);
#endif

  pBuffer->vertexRenderMode = mRP_VRM_TriangleList;

  mGL_ERROR_CHECK();

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mIndexedRenderDataBuffer_Create, IN mIndexedRenderDataBuffer<Args...> *pBuffer, IN mAllocator *pAllocator, const mString &vertexShaderSource, const mString &fragmentShaderSource, bool constantlyChangedVertices /* = false */, bool constantlyChangedIndices /* = false */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pBuffer == nullptr, mR_ArgumentNull);

  pBuffer->validIBO = pBuffer->validVBO = false; // Upload buffer data to validate buffer.
  
  mERROR_CHECK(mSharedPointer_Allocate(&pBuffer->shader, pAllocator, (std::function<void(mShader *)>)[](mShader *pData) { mShader_Destroy(pData); }, 1));
  mERROR_CHECK(mShader_Create(pBuffer->shader.GetPointer(), vertexShaderSource, fragmentShaderSource));
  
  pBuffer->constantlyChangedVertices = constantlyChangedVertices;
  pBuffer->constantlyChangedIndices = constantlyChangedIndices;

#if defined(mRENDERER_OPENGL)
  glGenVertexArrays(1, &pBuffer->vao);
  glBindVertexArray(pBuffer->vao);

  glGenBuffers(1, &pBuffer->ibo);
  glGenBuffers(1, &pBuffer->vbo);
#endif

  pBuffer->vertexRenderMode = mRP_VRM_TriangleList;

  mGL_ERROR_CHECK();

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mIndexedRenderDataBuffer_Destroy, IN mIndexedRenderDataBuffer<Args...> *pBuffer)
{
  mFUNCTION_SETUP();

  mGL_DEBUG_ERROR_CHECK();

#if defined(mRENDERER_OPENGL)
  if (pBuffer->vbo != 0)
  {
    glDeleteBuffers(1, &pBuffer->vbo);
    pBuffer->vbo = 0;
  }

  if (pBuffer->ibo != 0)
  {
    glDeleteBuffers(1, &pBuffer->ibo);
    pBuffer->ibo = 0;
  }

  if (pBuffer->vao != 0)
  {
    glDeleteVertexArrays(1, &pBuffer->vao);
    pBuffer->vao = 0;
  }

  mERROR_CHECK(mSharedPointer_Destroy(&pBuffer->shader));
#endif

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}

template<typename U, typename ...Args>
inline mFUNCTION(mIndexedRenderDataBuffer_SetVertexBuffer, mIndexedRenderDataBuffer<Args...> &buffer, const U *pData, const size_t count)
{
  mFUNCTION_SETUP();

  mERROR_IF(pData == nullptr, mR_ArgumentNull);

  //const size_t singleBlockSize = mRDBAttributeQuery_Internal<Args...>::GetSize();

  mPROFILE_SCOPED("mIndexedRenderDataBuffer_SetVertexBuffer");

#if defined(mRENDERER_OPENGL)
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ARRAY_BUFFER, buffer.vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(U) * count, pData, buffer.constantlyChangedVertices ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW);
#endif

  buffer.validVBO = true;

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mIndexedRenderDataBuffer_SetIndexBuffer, mIndexedRenderDataBuffer<Args...> &buffer, const uint32_t *pData, const size_t count)
{
  mFUNCTION_SETUP();

  mERROR_IF(pData == nullptr, mR_ArgumentNull);

  mPROFILE_SCOPED("mIndexedRenderDataBuffer_SetIndexBuffer");

#if defined(mRENDERER_OPENGL)
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer.ibo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint32_t) * count, pData, buffer.constantlyChangedIndices ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW);
#endif

  buffer.count = count;
  buffer.validIBO = true;

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mIndexedRenderDataBuffer_SetRenderCount, mIndexedRenderDataBuffer<Args...> &buffer, const size_t count)
{
  mFUNCTION_SETUP();

  buffer.count = count;

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mIndexedRenderDataBuffer_SetVertexRenderMode, mIndexedRenderDataBuffer<Args...> &buffer, const mRenderParams_VertexRenderMode renderMode)
{
  mFUNCTION_SETUP();

#if defined(mRENDERER_OPENGL)
  mERROR_IF(renderMode == mRP_VRM_Polygon || renderMode == mRP_VRM_QuadList || renderMode == mRP_VRM_QuadStrip, mR_NotSupported);
#endif

  buffer.vertexRenderMode = renderMode;

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mIndexedRenderDataBuffer_Draw, mIndexedRenderDataBuffer<Args...> &buffer)
{
  mFUNCTION_SETUP();

  mERROR_IF(!buffer.validIBO || !buffer.validVBO, mR_ResourceStateInvalid);

  mPROFILE_SCOPED("mIndexedRenderDataBuffer_Draw");

#if defined(mRENDERER_OPENGL)
  glBindVertexArray(buffer.vao);
  glBindBuffer(GL_ARRAY_BUFFER, buffer.vbo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer.ibo);
#endif

  mERROR_CHECK(mShader_Bind(*buffer.shader));
  mERROR_CHECK((mIDBAttributeQuery_Internal_SetAttributes<Args...>(buffer.shader.GetPointer())));

#if defined(mRENDERER_OPENGL)
  glDrawElements(buffer.vertexRenderMode, (GLsizei)buffer.count, GL_UNSIGNED_INT, nullptr);
#endif

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}

#endif // mIndexedRenderDataBuffer_h__
