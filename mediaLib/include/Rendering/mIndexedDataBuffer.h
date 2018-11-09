#ifndef mIndexedDataBuffer_h__
#define mIndexedDataBuffer_h__

#include "mRenderParams.h"
#include "mShader.h"

struct mIDBAttribute 
{
};

template <size_t TCount, const char *TAttributeName> 
struct mIDB_FloatAttribute : mIDBAttribute
{
  static size_t SingleCount() { return sizeof(float_t); };
  static const char *AttributeName() { return TAttributeName; };
  static size_t ValuesPerBlock() { return TCount; };
  static size_t DataSize() { return sizeof(float_t) * TCount; };

#if defined(mRENDERER_OPENGL)
  static GLenum DataType() { return GL_FLOAT; };
#endif
};

template <typename ...Args>
struct mIDBAttributeQuery_Internal
{
  static size_t GetSize();

  static mFUNCTION(SetAttribute, IN mShader *pShader, const size_t totalSize, const size_t offset);
};

// Invalid to have no attributes or attributes of size 0.
//template <>
//struct mIDBAttributeQuery_Internal<>
//{
//  static size_t GetSize() { return 0; };
//
//  static mFUNCTION(SetAttribute, IN mShader * /* pShader */, const size_t /* totalSize */, const size_t /* offset */) { return mR_Success; };
//};

template <typename T>
struct mIDBAttributeQuery_Internal<T>
{
  static size_t GetSize() { return T::DataSize(); };

  static mFUNCTION(SetAttribute, IN mShader *pShader, const size_t totalSize, const size_t offset)
  {
    mFUNCTION_SETUP();

#if defined(mRENDERER_OPENGL)
    const GLuint attributeIndex = glGetAttribLocation(pShader->shaderProgram, T::AttributeName());
    glEnableVertexAttribArray(attributeIndex);
    glVertexAttribPointer(attributeIndex, (GLint)sizeof(float_t), T::DataType(), GL_FALSE, (GLsizei)totalSize, (const void *)offset);
#endif

    mRETURN_SUCCESS();
  }
};

template <typename T, typename... Args>
struct mIDBAttributeQuery_Internal <T, Args...>
{
  static size_t GetSize() { return T::DataSize() + mIDBAttributeQuery_Internal<Args...>::GetSize(); };

  static mFUNCTION(SetAttribute, IN mShader *pShader, const size_t totalSize, const size_t offset)
  {
    mFUNCTION_SETUP();

    mERROR_CHECK(mIDBAttributeQuery_Internal<T>::SetAttribute(pShader, totalSize, offset));
    mERROR_CHECK(mIDBAttributeQuery_Internal<Args...>::SetAttribute(pShader, totalSize, offset + T::DataSize()));

    mRETURN_SUCCESS();
  }
};

template <typename... Args>
mFUNCTION(mIDBAttributeQuery_Internal_SetAttributes, IN mShader *pShader)
{
  mFUNCTION_SETUP();

  const size_t totalSize = mIDBAttributeQuery_Internal<Args...>::GetSize();
  mERROR_CHECK(mIDBAttributeQuery_Internal<Args...>::SetAttribute(pShader, totalSize, 0));

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

template <typename... Args>
struct mIndexedDataBuffer
{
  size_t count;
  bool constantlyChangedVertices;
  bool constantlyChangedIndices;
  bool validIBO;
  bool validVBO;

#if defined(mRENDERER_OPENGL)
  GLuint ibo;
  GLuint vbo;
#endif

  mPtr<mShader> shader;
  mRenderParams_VertexRenderMode vertexRenderMode;
};

template <typename... Args>
mFUNCTION(mIndexedDataBuffer_Create, IN mIndexedDataBuffer<Args...> *pBuffer, const mPtr<mShader> &shader, bool constantlyChangedVertices = false, bool constantlyChangedIndices = false);

template <typename... Args>
mFUNCTION(mIndexedDataBuffer_Create, IN mIndexedDataBuffer<Args...> *pBuffer, IN mAllocator *pAllocator, const mString &vertexShaderSource, const mString &fragmentShaderSource, bool constantlyChangedVertices = false, bool constantlyChangedIndices = false);

template <typename... Args>
mFUNCTION(mIndexedDataBuffer_Destroy, IN mIndexedDataBuffer<Args...> *pBuffer);

template <typename U, typename... Args>
mFUNCTION(mIndexedDataBuffer_SetVertexBuffer, mIndexedDataBuffer<Args...> &buffer, const U *pData, const size_t count);

template <typename... Args>
mFUNCTION(mIndexedDataBuffer_SetIndexBuffer, mIndexedDataBuffer<Args...> &buffer, const uint32_t *pData, const size_t count);

template <typename... Args>
mFUNCTION(mIndexedDataBuffer_SetRenderCount, mIndexedDataBuffer<Args...> &buffer, const size_t count);

template <typename... Args>
mFUNCTION(mIndexedDataBuffer_SetVertexRenderMode, mIndexedDataBuffer<Args...> &buffer, const mRenderParams_VertexRenderMode renderMode);

template <typename... Args>
mFUNCTION(mIndexedDataBuffer_Draw, mIndexedDataBuffer<Args...> &buffer);

//////////////////////////////////////////////////////////////////////////

template<typename ...Args>
inline mFUNCTION(mIndexedDataBuffer_Create, IN mIndexedDataBuffer<Args...> *pBuffer, const mPtr<mShader> &shader, bool constantlyChangedVertices /* = false */, bool constantlyChangedIndices /* = false */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pBuffer == nullptr || shader == nullptr, mR_ArgumentNull);

  pBuffer->validIBO = pBuffer->validVBO = false; // Upload buffer data to validate buffer.
  pBuffer->shader = shader;
  pBuffer->constantlyChangedVertices = constantlyChangedVertices;
  pBuffer->constantlyChangedIndices = constantlyChangedIndices;

#if defined(mRENDERER_OPENGL)
  glGenBuffers(1, &pBuffer->ibo);
  glGenBuffers(1, &pBuffer->vbo);
#endif

  pBuffer->vertexRenderMode = mRP_VRM_TriangleList;

  mGL_ERROR_CHECK();

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mIndexedDataBuffer_Create, IN mIndexedDataBuffer<Args...> *pBuffer, IN mAllocator *pAllocator, const mString &vertexShaderSource, const mString &fragmentShaderSource, bool constantlyChangedVertices /* = false */, bool constantlyChangedIndices /* = false */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pBuffer == nullptr, mR_ArgumentNull);

  pBuffer->validIBO = pBuffer->validVBO = false; // Upload buffer data to validate buffer.
  
  mERROR_CHECK(mSharedPointer_Allocate(&pBuffer->shader, pAllocator, (std::function<void(mShader *)>)[](mShader *pData) { mShader_Destroy(pData); }, 1));
  mERROR_CHECK(mShader_Create(pBuffer->shader.GetPointer(), vertexShaderSource, fragmentShaderSource));
  
  pBuffer->constantlyChangedVertices = constantlyChangedVertices;
  pBuffer->constantlyChangedIndices = constantlyChangedIndices;

#if defined(mRENDERER_OPENGL)
  glGenBuffers(1, &pBuffer->ibo);
  glGenBuffers(1, &pBuffer->vbo);
#endif

  mGL_ERROR_CHECK();

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mIndexedDataBuffer_Destroy, IN mIndexedDataBuffer<Args...> *pBuffer)
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

  mERROR_CHECK(mSharedPointer_Destroy(&pBuffer->shader));
#endif

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}

template<typename U, typename ...Args>
inline mFUNCTION(mIndexedDataBuffer_SetVertexBuffer, mIndexedDataBuffer<Args...> &buffer, const U *pData, const size_t count)
{
  mFUNCTION_SETUP();

  mERROR_IF(pData == nullptr, mR_ArgumentNull);

  const size_t singleBlockSize = mIDBAttributeQuery_Internal<Args...>::GetSize();

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
inline mFUNCTION(mIndexedDataBuffer_SetIndexBuffer, mIndexedDataBuffer<Args...> &buffer, const uint32_t *pData, const size_t count)
{
  mFUNCTION_SETUP();

  mERROR_IF(pData == nullptr, mR_ArgumentNull);

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
inline mFUNCTION(mIndexedDataBuffer_SetRenderCount, mIndexedDataBuffer<Args...> &buffer, const size_t count)
{
  mFUNCTION_SETUP();

  buffer.count = count;

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mIndexedDataBuffer_SetVertexRenderMode, mIndexedDataBuffer<Args...> &buffer, const mRenderParams_VertexRenderMode renderMode)
{
  mFUNCTION_SETUP();

#if defined(mRENDERER_OPENGL)
  mERROR_IF(renderMode == mRP_VRM_Polygon || renderMode == mRP_VRM_QuadList || renderMode == mRP_VRM_QuadStrip, mR_NotSupported);
#endif

  buffer.vertexRenderMode = renderMode;

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mIndexedDataBuffer_Draw, mIndexedDataBuffer<Args...> &buffer)
{
  mFUNCTION_SETUP();

  mERROR_IF(!buffer.validIBO || !buffer.validVBO, mR_ResourceStateInvalid);

#if defined(mRENDERER_OPENGL)
  glBindBuffer(GL_ARRAY_BUFFER, buffer.vbo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer.ibo);
#endif

  mERROR_CHECK(mShader_Bind(*buffer.shader));
  mERROR_CHECK((mIDBAttributeQuery_Internal_SetAttributes<Args...>(buffer.shader.GetPointer())));

#if defined(mRENDERER_OPENGL)
  glDrawElements(buffer.vertexRenderMode, (GLsizei)buffer.count, GL_UNSIGNED_INT, nullptr);
#endif

  mRETURN_SUCCESS();
}

#endif // mIndexedDataBuffer_h__
