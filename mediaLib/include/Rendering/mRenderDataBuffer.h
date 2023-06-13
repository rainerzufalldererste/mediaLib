#ifndef mRenderDataBuffer_h__
#define mRenderDataBuffer_h__

#include "mRenderParams.h"
#include "mShader.h"
#include "mProfiler.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "6HywpGrbnlE+OSrRAs8rWtSVR96XnFDaMS3jrJz+RaoyqFEtgFoouzd2pqC3A+ARE7hM46gZvjJfkmBa"
#endif

struct mRDBAttribute
{
};

template <size_t TCount, const char *TAttributeName>
struct mRDB_FloatAttribute : mRDBAttribute
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
struct mRDBAttributeQuery_Internal
{
  static size_t GetSize();

  static mFUNCTION(SetAttribute, IN mShader *pShader, const size_t totalSize, const size_t offset, const bool instanced);
};

template <>
struct mRDBAttributeQuery_Internal<>
{
  static size_t GetSize() { return 0; };

  static mFUNCTION(SetAttribute, IN mShader * /* pShader */, const size_t /* totalSize */, const size_t /* offset */, const bool /* instanced */) { return mR_Success; };
};

template <typename T>
struct mRDBAttributeQuery_Internal<T>
{
  static size_t GetSize() { return T::DataSize(); };

  static mFUNCTION(SetAttribute, IN mShader *pShader, const size_t totalSize, const size_t offset, const bool instanced)
  {
    mFUNCTION_SETUP();
     
#if defined(mRENDERER_OPENGL)
    const mShaderAttributeIndex_t attributeIndex = mShader_GetAttributeIndex(*pShader, T::AttributeName());
    mGL_DEBUG_ERROR_CHECK();

    glEnableVertexAttribArray(attributeIndex);
    glVertexAttribPointer(attributeIndex, (GLint)sizeof(float_t), T::DataType(), GL_FALSE, (GLsizei)totalSize, (const void *)offset);

    if (instanced)
      glVertexAttribDivisor(attributeIndex, 1);
#endif

    mGL_DEBUG_ERROR_CHECK();

    mRETURN_SUCCESS();
  }
};

template <typename T, typename... Args>
struct mRDBAttributeQuery_Internal <T, Args...>
{
  static size_t GetSize() { return T::DataSize() + mRDBAttributeQuery_Internal<Args...>::GetSize(); };

  static mFUNCTION(SetAttribute, IN mShader *pShader, const size_t totalSize, const size_t offset, const bool instanced)
  {
    mFUNCTION_SETUP();

    mERROR_CHECK(mRDBAttributeQuery_Internal<T>::SetAttribute(pShader, totalSize, offset, instanced));
    mERROR_CHECK(mRDBAttributeQuery_Internal<Args...>::SetAttribute(pShader, totalSize, offset + T::DataSize(), instanced));

    mRETURN_SUCCESS();
  }
};

template <typename... Args>
mFUNCTION(mIDBAttributeQuery_Internal_SetAttributes, IN mShader *pShader, const bool instanced = false)
{
  mFUNCTION_SETUP();

  const size_t totalSize = mRDBAttributeQuery_Internal<Args...>::GetSize();
  mERROR_CHECK(mRDBAttributeQuery_Internal<Args...>::SetAttribute(pShader, totalSize, 0, instanced));

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

// Template Parameters should be `mDRBAttribute`s like `mRDB_FloatAttribute`.
template <typename... Args>
struct mRenderDataBuffer
{
  size_t count;
  bool constantlyChangedVertices;
  bool validVBO;

#if defined(mRENDERER_OPENGL)
  GLuint vao, vbo;
#endif

  mPtr<mShader> shader;
  mRenderParams_VertexRenderMode vertexRenderMode;
};

template <typename... Args>
mFUNCTION(mRenderDataBuffer_Create, IN mRenderDataBuffer<Args...> *pBuffer, const mPtr<mShader> &shader, bool constantlyChangedVertices = false);

template <typename... Args>
mFUNCTION(mRenderDataBuffer_Create, IN mRenderDataBuffer<Args...> *pBuffer, IN mAllocator *pAllocator, const mString &vertexShaderSource, const mString &fragmentShaderSource, bool constantlyChangedVertices = false);

template <typename... Args>
mFUNCTION(mRenderDataBuffer_Destroy, IN mRenderDataBuffer<Args...> *pBuffer);

template <typename U, typename... Args>
mFUNCTION(mRenderDataBuffer_SetVertexBuffer, mRenderDataBuffer<Args...> &buffer, const U *pData, const size_t count);

template <typename... Args>
mFUNCTION(mRenderDataBuffer_SetRenderCount, mRenderDataBuffer<Args...> &buffer, const size_t count);

template <typename... Args>
mFUNCTION(mRenderDataBuffer_SetVertexRenderMode, mRenderDataBuffer<Args...> &buffer, const mRenderParams_VertexRenderMode renderMode);

template<typename... Args>
mFUNCTION(mRenderDataBuffer_BindAttributes, mRenderDataBuffer<Args...> &buffer);

template <typename... Args>
mFUNCTION(mRenderDataBuffer_Draw, mRenderDataBuffer<Args...> &buffer);

//////////////////////////////////////////////////////////////////////////

template<typename ...Args>
inline mFUNCTION(mRenderDataBuffer_Create, IN mRenderDataBuffer<Args...> *pBuffer, const mPtr<mShader> &shader, bool constantlyChangedVertices /* = false */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pBuffer == nullptr || shader == nullptr, mR_ArgumentNull);

  pBuffer->validVBO = false; // Upload buffer data to validate buffer.
  pBuffer->shader = shader;
  pBuffer->constantlyChangedVertices = constantlyChangedVertices;

#if defined(mRENDERER_OPENGL)
  glGenVertexArrays(1, &pBuffer->vao);
  glBindVertexArray(pBuffer->vao);

  glGenBuffers(1, &pBuffer->vbo);
  glBindVertexArray(0);
#endif

  pBuffer->vertexRenderMode = mRP_VRM_TriangleList;

  mGL_ERROR_CHECK();

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mRenderDataBuffer_Create, IN mRenderDataBuffer<Args...> *pBuffer, IN mAllocator *pAllocator, const mString &vertexShaderSource, const mString &fragmentShaderSource, bool constantlyChangedVertices /* = false */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pBuffer == nullptr, mR_ArgumentNull);

  pBuffer->validVBO = false; // Upload buffer data to validate buffer.

  mDEFER_CALL_ON_ERROR(&pBuffer->shader, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Allocate(&pBuffer->shader, pAllocator, (std::function<void(mShader *)>)[](mShader *pData) { mShader_Destroy(pData); }, 1));
  mERROR_CHECK(mShader_Create(pBuffer->shader.GetPointer(), vertexShaderSource, fragmentShaderSource));

  pBuffer->constantlyChangedVertices = constantlyChangedVertices;

#if defined(mRENDERER_OPENGL)
  glGenVertexArrays(1, &pBuffer->vao);
  glBindVertexArray(pBuffer->vao);

  glGenBuffers(1, &pBuffer->vbo);
  glBindVertexArray(0);
#endif

  pBuffer->vertexRenderMode = mRP_VRM_TriangleList;

  mGL_ERROR_CHECK();

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mRenderDataBuffer_Destroy, IN mRenderDataBuffer<Args...> *pBuffer)
{
  mFUNCTION_SETUP();

  mGL_DEBUG_ERROR_CHECK();

#if defined(mRENDERER_OPENGL)
  if (pBuffer->vbo != 0)
  {
    glDeleteBuffers(1, &pBuffer->vbo);
    pBuffer->vbo = 0;
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
inline mFUNCTION(mRenderDataBuffer_SetVertexBuffer, mRenderDataBuffer<Args...> &buffer, const U *pData, const size_t count)
{
  mFUNCTION_SETUP();

  mERROR_IF(pData == nullptr, mR_ArgumentNull);

  const size_t singleBlockSize = mRDBAttributeQuery_Internal<Args...>::GetSize();
  mERROR_IF(singleBlockSize == 0, mR_ResourceStateInvalid);

  mPROFILE_SCOPED("mRenderDataBuffer_SetVertexBuffer");

#if defined(mRENDERER_OPENGL)
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ARRAY_BUFFER, buffer.vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(U) * count, pData, buffer.constantlyChangedVertices ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW);

  mGL_DEBUG_ERROR_CHECK();
#endif

  buffer.validVBO = true;
  buffer.count = (sizeof(U) * count) / singleBlockSize;

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mRenderDataBuffer_SetRenderCount, mRenderDataBuffer<Args...> &buffer, const size_t count)
{
  mFUNCTION_SETUP();

  buffer.count = count;

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mRenderDataBuffer_SetVertexRenderMode, mRenderDataBuffer<Args...> &buffer, const mRenderParams_VertexRenderMode renderMode)
{
  mFUNCTION_SETUP();

#if defined(mRENDERER_OPENGL)
  mERROR_IF(renderMode == mRP_VRM_Polygon || renderMode == mRP_VRM_QuadList || renderMode == mRP_VRM_QuadStrip, mR_NotSupported);
#endif

  buffer.vertexRenderMode = renderMode;

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mRenderDataBuffer_BindAttributes, mRenderDataBuffer<Args...> &buffer)
{
  mFUNCTION_SETUP();

  mERROR_IF(!buffer.validVBO && mRDBAttributeQuery_Internal<Args...>::GetSize() > 0, mR_ResourceStateInvalid);

  mGL_DEBUG_ERROR_CHECK();

#if defined(mRENDERER_OPENGL)
  glBindVertexArray(buffer.vao);
  glBindBuffer(GL_ARRAY_BUFFER, buffer.vbo);
  mGL_DEBUG_ERROR_CHECK();
#endif

  mERROR_CHECK(mShader_Bind(*buffer.shader));
  mERROR_CHECK((mIDBAttributeQuery_Internal_SetAttributes<Args...>(buffer.shader.GetPointer())));

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mRenderDataBuffer_Draw, mRenderDataBuffer<Args...> &buffer)
{
  mFUNCTION_SETUP();

  mPROFILE_SCOPED("mRenderDataBuffer_Draw");

  mERROR_CHECK(mRenderDataBuffer_BindAttributes(buffer));

#if defined(mRENDERER_OPENGL)
  glDrawArrays(buffer.vertexRenderMode, 0, (GLsizei)buffer.count);
#endif

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}
#endif // mRenderDataBuffer_h__
