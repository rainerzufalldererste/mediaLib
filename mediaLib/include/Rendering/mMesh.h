#ifndef mMesh_h__
#define mMesh_h__

#include "mRenderParams.h"
#include "mShader.h"
#include "mBinaryChunk.h"
#include "mArray.h"
#include "mTexture.h"
#include <initializer_list>
#include "mKeyValuePair.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "/+Me+ot7sZez8oj9vEmsK0z1uqgYT2gM1wZwbj9kgHBUVaJO3PVWMJo2qdFhWw5DTOYZCgOXHyVusQPQ"
#endif

enum mRenderObjectParamType
{
  mROPT_Position,
  mROPT_TexCoord,
  mROPT_Colour,
  mROPT_TextureBlendFactor,
  mROPT_NormalDirection,
  mROPT_AlphaChannel,
  mROPT_TextureAlphaFlag,
  mROPT_Scale,
};

enum mRenderObjectParamSubType
{
  mROPST_2dPosition,
  mROPST_3dPosition,
  mROPST_2dTextureCoordinate,
  mROPST_Colour,
  mROPST_TextureBlendFactor,
  mROPST_3dNormalDirection,
  mROPST_AlphaChannel,
  mROPST_TextureAlphaFlag,

  // uniforms
  mROPST_ScaleUniform,
  mROPST_Scale2dUniform,
};

struct mRenderObjectParam
{
  // Base Type.
};

#define mRenderObjectParam_Position_AttributeName "position"
#define mRenderObjectParam_MatrixAttributeName "matrix"
#define mMeshTexcoord_AttributeName "texcoord"
#define mMeshColour_AttributeName "colour"
#define mMeshTextureBlendFactor_AttributeName "textureMix"
#define mRenderObjectParam_TextureAttributeName "texture"
#define mMeshNormalPosition_AttributeName "normal"
#define mMeshAlphaChannel_AttributeName "alpha"
#define mMeshScaleUniform_AttributeName "scale"
#define mMeshScale2dUniform_AttributeName "scale2d"
#define mMeshScreenSize_AttributeName "screenSize"
#define mMeshOutColour_AttributeName "outColour"

enum mMeshFactory_AttributeInformationType
{
  mMF_AIT_Attribute,
  mMF_AIT_Uniform,
  mMF_AIT_Flag,
};

struct mMesh2dPosition : mRenderObjectParam
{
  mVec2f position;

  mMesh2dPosition(const mVec2f &position) : position(position) {}
  mMesh2dPosition(const float_t x, const float_t y) : position(mVec2f(x, y)) {}

  static const mRenderObjectParamType type = mROPT_Position;
  static const mRenderObjectParamSubType subType = mROPST_2dPosition;
  static const size_t size = sizeof(mVec2f);
  static const mMeshFactory_AttributeInformationType attributeInformationType = mMF_AIT_Attribute;

  static inline const std::string attributeName(const size_t index = 0) 
  {
    char text[1024];
    sprintf_s(text, "%s%" PRIu64, mRenderObjectParam_Position_AttributeName, index);
    return std::string(text);
  };
};

struct mMesh3dPosition : mRenderObjectParam
{
  mVec3f position;

  mMesh3dPosition(const mVec3f &position) : position(position) {}
  mMesh3dPosition(const float_t x, const float_t y, const float_t z) : position(mVec3f(x, y, z)) {}

  static const mRenderObjectParamType type = mROPT_Position;
  static const mRenderObjectParamSubType subType = mROPST_3dPosition;
  static const size_t size = sizeof(mVec3f);
  static const mMeshFactory_AttributeInformationType attributeInformationType = mMF_AIT_Attribute;

  static inline const std::string attributeName(const size_t index = 0)
  {
    char text[1024];
    sprintf_s(text, "%s%" PRIu64, mRenderObjectParam_Position_AttributeName, index);
    return std::string(text);
  };
};

struct mMeshTexcoord : mRenderObjectParam
{
  mVec2f position;

  mMeshTexcoord(const mVec2f &position) : position(position) {}
  mMeshTexcoord(const float_t x, const float_t y) : position(mVec2f(x, y)) {}

  static const mRenderObjectParamType type = mROPT_TexCoord;
  static const mRenderObjectParamSubType subType = mROPST_2dTextureCoordinate;
  static const size_t size = sizeof(mVec2f);
  static const mMeshFactory_AttributeInformationType attributeInformationType = mMF_AIT_Attribute;

  static inline const std::string attributeName(const size_t index = 0)
  {
    char text[1024];
    sprintf_s(text, "%s%" PRIu64, mMeshTexcoord_AttributeName, index);
    return std::string(text);
  };
};

struct mMeshColour : mRenderObjectParam
{
  mVec3f colour;

  mMeshColour(const mVec3f &colour) : colour(colour) {}
  mMeshColour(const float_t r, const float_t g, const float_t b) : colour(mVec3f(r, g, b)) {}

  static const mRenderObjectParamType type = mROPT_Colour;
  static const mRenderObjectParamSubType subType = mROPST_Colour;
  static const size_t size = sizeof(mVec3f);
  static const mMeshFactory_AttributeInformationType attributeInformationType = mMF_AIT_Attribute;

  static inline const std::string attributeName(const size_t index = 0)
  {
    char text[1024];
    sprintf_s(text, "%s%" PRIu64, mMeshColour_AttributeName, index);
    return std::string(text);
  };
};

struct mMeshTextureBlendFactor : mRenderObjectParam
{
  float_t blendFactor;

  mMeshTextureBlendFactor(const float_t blendFactor) : blendFactor(blendFactor) {}

  static const mRenderObjectParamType type = mROPT_TextureBlendFactor;
  static const mRenderObjectParamSubType subType = mROPST_TextureBlendFactor;
  static const size_t size = sizeof(float_t);
  static const mMeshFactory_AttributeInformationType attributeInformationType = mMF_AIT_Attribute;

  static inline const std::string attributeName(const size_t index = 0)
  {
    char text[1024];
    sprintf_s(text, "%s%" PRIu64, mMeshTextureBlendFactor_AttributeName, index);
    return std::string(text);
  };
};

struct mMeshNormalPosition : mRenderObjectParam
{
  mVec3f direction;

  mMeshNormalPosition(const mVec3f &direction) : direction(direction) {}
  mMeshNormalPosition(const float_t x, const float_t y, const float_t z) : direction(mVec3f(x, y, z)) {}

  static const mRenderObjectParamType type = mROPT_NormalDirection;
  static const mRenderObjectParamSubType subType = mROPST_3dNormalDirection;
  static const size_t size = sizeof(mVec3f);
  static const mMeshFactory_AttributeInformationType attributeInformationType = mMF_AIT_Attribute;

  static inline const std::string attributeName(const size_t index = 0)
  {
    char text[1024];
    sprintf_s(text, "%s%" PRIu64, mMeshNormalPosition_AttributeName, index);
    return std::string(text);
  };
};

struct mMeshAlphaChannel : mRenderObjectParam
{
  float_t alpha;

  mMeshAlphaChannel(const float_t alpha) : alpha(alpha) {}

  static const mRenderObjectParamType type = mROPT_AlphaChannel;
  static const mRenderObjectParamSubType subType = mROPST_AlphaChannel;
  static const size_t size = sizeof(float_t);
  static const mMeshFactory_AttributeInformationType attributeInformationType = mMF_AIT_Attribute;

  static inline const std::string attributeName(const size_t index = 0)
  {
    char text[1024];
    sprintf_s(text, "%s%" PRIu64, mMeshAlphaChannel_AttributeName, index);
    return std::string(text);
  };
};

struct mMeshScaleUniform : mRenderObjectParam
{
  mMeshScaleUniform() {}
  mMeshScaleUniform(nullptr_t) {}

  static const mRenderObjectParamType type = mROPT_Scale;
  static const mRenderObjectParamSubType subType = mROPST_ScaleUniform;
  static const size_t size = 0;
  static const mMeshFactory_AttributeInformationType attributeInformationType = mMF_AIT_Uniform;

  static inline const std::string uniformName(const size_t index = 0)
  {
    char text[1024];
    sprintf_s(text, "%s%" PRIu64, mMeshScaleUniform_AttributeName, index);
    return std::string(text);
  };
};

struct mMeshScale2dUniform : mRenderObjectParam
{
  mMeshScale2dUniform() {}
  mMeshScale2dUniform(nullptr_t) {}

  static const mRenderObjectParamType type = mROPT_Scale;
  static const mRenderObjectParamSubType subType = mROPST_Scale2dUniform;
  static const size_t size = 0;
  static const mMeshFactory_AttributeInformationType attributeInformationType = mMF_AIT_Uniform;

  static inline const std::string uniformName(const size_t index = 0)
  {
    char text[1024];
    sprintf_s(text, "%s%" PRIu64, mMeshScale2dUniform_AttributeName, index);
    return std::string(text);
  };
};

struct mMeshTextureAlphaFlag : mRenderObjectParam
{
  mMeshTextureAlphaFlag() {}
  mMeshTextureAlphaFlag(nullptr_t) {}

  static const mRenderObjectParamType type = mROPT_TextureAlphaFlag;
  static const mRenderObjectParamSubType subType = mROPST_TextureAlphaFlag;
  static const size_t size = 0;
  static const mMeshFactory_AttributeInformationType attributeInformationType = mMF_AIT_Flag;
};

struct mMeshFactory_AttributeInformation
{
  size_t size;
  size_t offset;
  mMeshFactory_AttributeInformationType type;
  size_t individualSubTypeSize;
  char name[32];

#if defined(mRENDERER_OPENGL)
  GLenum dataType;

  inline mMeshFactory_AttributeInformation(const char name[32], const size_t size, const size_t offset, const mMeshFactory_AttributeInformationType type, const GLenum dataType, const size_t individualSubTypeSize) : size(size), offset(offset), type(type), dataType(dataType), individualSubTypeSize(individualSubTypeSize) 
  {
    mMemcpy((char *)this->name, name, 32);
  }
#else
  inline mMeshFactory_AttributeInformation(const char name[32], const size_t size, const size_t offset, const mMeshFactory_AttributeInformationType type, const size_t individualSubTypeSize) : size(size), offset(offset), type(type), individualSubTypeSize(individualSubTypeSize) 
  {
    mMemcpy((char *)this->name, name, 32);
  }
#endif

  inline mMeshFactory_AttributeInformation() = default;
};

struct mMeshFactory_Internal
{
  uint8_t position : 1;
  uint8_t position3d : 1;
  uint8_t textureBlendFactor : 1;
  uint8_t colour : 1;
  uint8_t alpha : 1;
  uint8_t useTextureAlpha : 1;
  uint8_t normal : 1;
  uint8_t scaleUniform : 1;
  uint8_t scale2dUniform : 1;
  size_t textureCoordCount;
  size_t size;

  char vertShader[1024];
  char vertShaderImpl[1024];
  char fragShader[1024];

  mPtr<mQueue<mMeshFactory_AttributeInformation>> information;
};

template <typename ...Args>
mFUNCTION(mMeshFactory_Internal_Build, mMeshFactory_Internal *pMeshFactory, Args&&... args);

template <typename ...T>
struct mMeshFactory_Internal_Unpacker;

template <typename ...Args>
struct mMeshFactory
{
  mMeshFactory_Internal meshFactoryInternal;
  mPtr<mBinaryChunk> values;
  mRenderParams_VertexRenderMode triangleRenderMode;
};

struct mMeshAttributeContainer
{
  char attributeName[64];
  size_t size;
  size_t subComponentSize;
  size_t attributeCount;
  mPtr<mBinaryChunk> attributes;
#if defined(mRENDERER_OPENGL)
  GLenum dataType;
#endif
};

struct mMeshAttribute
{
  size_t attributeIndex;
  size_t dataEntrySize;
  size_t dataEntrySubComponentSize;
  size_t count;
  size_t offset;

#if defined(mRENDERER_OPENGL)
  GLenum dataType;

  inline mMeshAttribute(const size_t attributeIndex, const size_t dataEntrySize, const size_t dataEntrySubComponentSize, const size_t count, const size_t offset, const GLenum dataType) :
    attributeIndex(attributeIndex),
    dataEntrySize(dataEntrySize),
    dataEntrySubComponentSize(dataEntrySubComponentSize),
    count(count),
    offset(offset),
    dataType(dataType)
  {}
#endif

  inline mMeshAttribute() = default;
};

struct mMesh
{
  mPtr<mShader> shader;
  size_t dataSize;
  mRenderParams_UploadState uploadState;
  mPtr<mQueue<mKeyValuePair<mString, mPtr<mTexture>>>> textures;
  mRenderParams_VertexRenderMode triangleRenderMode;
  mPtr<mQueue<mMeshAttribute>> information;
  size_t primitiveCount;
#if defined (mRENDERER_OPENGL)
  bool hasVbo;
  GLuint vbo;
#endif
};

template <typename ...Args>
mFUNCTION(mMeshFactory_Destroy_Internal, mMeshFactory<Args...> *pFactory);

template <typename ...Args>
mFUNCTION(mMeshFactory_Create, OUT mPtr<mMeshFactory<Args...>> *pFactory, IN mAllocator *pAllocator, const mRenderParams_VertexRenderMode triangleRenderMode = mRP_VRM_TriangleList);

template <typename ...Args>
mFUNCTION(mMeshFactory_Destroy, IN_OUT mPtr<mMeshFactory<Args...>> *pFactory);

template<typename ...Args, typename ...TTextures>
mFUNCTION(mMeshFactory_CreateMesh, mPtr<mMeshFactory<Args...>> &factory, OUT mPtr<mMesh> *pMesh, IN mAllocator *pAllocator, TTextures... textures);

template <typename ...Args>
mFUNCTION(mMeshFactory_AppendData, mPtr<mMeshFactory<Args...>> &factory, Args&&... args);

template <typename ...Args>
mFUNCTION(mMeshFactory_GrowBack, mPtr<mMeshFactory<Args...>> &factory, const size_t items);

template <typename ...Args>
mFUNCTION(mMeshFactory_Clear, mPtr<mMeshFactory<Args...>> &factory);

template <typename T>
mFUNCTION(mMeshAttributeContainer_Create, OUT mPtr<mMeshAttributeContainer> *pMeshAttributeContainer, IN mAllocator *pAllocator, char attributeName[64], const std::initializer_list<T> &items);

template <typename T>
mFUNCTION(mMeshAttributeContainer_Create, OUT mPtr<mMeshAttributeContainer> *pMeshAttributeContainer, IN mAllocator *pAllocator, char attributeName[64], mPtr<mQueue<T>> &items);

template <typename T>
mFUNCTION(mMeshAttributeContainer_ResolveType_Internal, mPtr<mMeshAttributeContainer> &attributeContainer);

mFUNCTION(mMeshAttributeContainer_Destroy, IN_OUT mPtr<mMeshAttributeContainer> *pMeshAttributeContainer);

mFUNCTION(mMeshAttributeContainer_Destroy_Internal, IN mMeshAttributeContainer *pMeshAttributeContainer);

mFUNCTION(mMesh_Create, OUT mPtr<mMesh> *pMesh, IN mAllocator *pAllocator, const std::initializer_list<mPtr<mMeshAttributeContainer>> &attributeInformation, mPtr<mShader> &shader, mPtr<mQueue<mKeyValuePair<mString, mPtr<mTexture>>>> &textures, const mRenderParams_VertexRenderMode triangleRenderMode = mRP_VRM_TriangleList);

mFUNCTION(mMesh_Create, OUT mPtr<mMesh> *pMesh, IN mAllocator *pAllocator, mPtr<mQueue<mPtr<mMeshAttributeContainer>>> &attributeInformation, mPtr<mShader> &shader, mPtr<mQueue<mKeyValuePair<mString, mPtr<mTexture>>>> &textures, const mRenderParams_VertexRenderMode triangleRenderMode = mRP_VRM_TriangleList);

mFUNCTION(mMesh_Create, OUT mPtr<mMesh> *pMesh, IN mAllocator *pAllocator, mPtr<mQueue<mMeshAttribute>> &attributeInformation, mPtr<mShader> &shader, mPtr<mBinaryChunk> &data, mPtr<mQueue<mKeyValuePair<mString, mPtr<mTexture>>>> &textures, const mRenderParams_VertexRenderMode triangleRenderMode = mRP_VRM_TriangleList);

mFUNCTION(mMesh_Destroy, IN_OUT mPtr<mMesh> *pMesh);

mFUNCTION(mMesh_Destroy_Internal, mMesh *pMesh);

mFUNCTION(mMesh_Upload, mPtr<mMesh> &data);

mFUNCTION(mMesh_GetUploadState, mPtr<mMesh> &data, OUT mRenderParams_UploadState *pUploadState);

mFUNCTION(mMesh_Render, mPtr<mMesh> &data);

mFUNCTION(mMesh_Render, mPtr<mMesh> &data, mMatrix &matrix);

//////////////////////////////////////////////////////////////////////////

template <typename ...Args>
mFUNCTION(mMeshFactory_Create, OUT mPtr<mMeshFactory<Args...>> *pFactory, IN mAllocator *pAllocator, const mRenderParams_VertexRenderMode triangleRenderMode /* = mRP_VRM_TriangleList */)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mSharedPointer_Allocate(pFactory, pAllocator, (std::function<void(mMeshFactory<Args...> *)>)[](mMeshFactory<Args...> *pData) {mMeshFactory_Destroy_Internal(pData);}, 1));

  mERROR_CHECK(mBinaryChunk_Create(&(*pFactory)->values, pAllocator));
  mERROR_CHECK(mQueue_Create(&(*pFactory)->meshFactoryInternal.information, pAllocator));

  mERROR_CHECK(mMeshFactory_Internal_Build<Args...>(&(*pFactory)->meshFactoryInternal));

  (*pFactory)->triangleRenderMode = triangleRenderMode;

  mRETURN_SUCCESS();
}

template <typename ...Args>
mFUNCTION(mMeshFactory_Destroy, IN_OUT mPtr<mMeshFactory<Args...>> *pFactory)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mSharedPointer_Destroy(pFactory));

  mRETURN_SUCCESS();
}

template <typename ...MeshArgs>
struct mMeshFactory_Internal_TextureParameterUnpacker;

template <>
struct mMeshFactory_Internal_TextureParameterUnpacker <>
{
  static mFUNCTION(Unpack, mPtr<mQueue<mKeyValuePair<mString, mPtr<mTexture>>>> *pTextureArray, const size_t index)
  {
    mFUNCTION_SETUP();

    mUnused(index, pTextureArray);

    mRETURN_SUCCESS();
  }
};

template <>
struct mMeshFactory_Internal_TextureParameterUnpacker <mPtr<mImageBuffer>&>
{
  static mFUNCTION(Unpack, mPtr<mQueue<mKeyValuePair<mString, mPtr<mTexture>>>> *pTextureArray, const size_t index, mPtr<mImageBuffer> &imageBuffer)
  {
    mFUNCTION_SETUP();

    mPtr<mTexture> texture;
    mDEFER_CALL(&texture, mSharedPointer_Destroy);
    mERROR_CHECK(mSharedPointer_Allocate(&texture, nullptr, (std::function<void(mTexture *)>)[](mTexture *pData) {mTexture_Destroy(pData); }, 1));
    mERROR_CHECK(mTexture_Create(texture.GetPointer(), imageBuffer, false, index));

    char name[64];
    mERROR_IF(0 > sprintf_s(name, "_" mRenderObjectParam_TextureAttributeName "%" PRIu64, index), mR_InternalError);

    mERROR_CHECK(mQueue_PushBack(*pTextureArray, mKeyValuePair<mString, mPtr<mTexture>>(name, texture)));

    mRETURN_SUCCESS();
  }
};

template <>
struct mMeshFactory_Internal_TextureParameterUnpacker <mPtr<mTexture>&>
{
  static mFUNCTION(Unpack, mPtr<mQueue<mKeyValuePair<mString, mPtr<mTexture>>>> *pTextureArray, const size_t index, mPtr<mTexture> &texture)
  {
    mFUNCTION_SETUP();

    char name[64];
    mERROR_IF(0 > sprintf_s(name, "_" mRenderObjectParam_TextureAttributeName "%" PRIu64, index), mR_InternalError);

    mERROR_CHECK(mQueue_PushBack(*pTextureArray, mKeyValuePair<mString, mPtr<mTexture>>(name, texture)));

    mRETURN_SUCCESS();
  }
};

template <>
struct mMeshFactory_Internal_TextureParameterUnpacker <mPtr<mImageBuffer>>
{
  static mFUNCTION(Unpack, mPtr<mQueue<mKeyValuePair<mString, mPtr<mTexture>>>> *pTextureArray, const size_t index, mPtr<mImageBuffer> imageBuffer)
  {
    mFUNCTION_SETUP();

    mPtr<mTexture> texture;
    mDEFER_CALL(&texture, mSharedPointer_Destroy);
    mERROR_CHECK(mSharedPointer_Allocate(&texture, nullptr, (std::function<void(mTexture *)>)[](mTexture *pData) {mTexture_Destroy(pData); }, 1));
    mERROR_CHECK(mTexture_Create(texture.GetPointer(), imageBuffer, false, index));

    char name[64];
    mERROR_IF(0 > sprintf_s(name, "_" mRenderObjectParam_TextureAttributeName "%" PRIu64, index), mR_InternalError);

    mERROR_CHECK(mQueue_PushBack(*pTextureArray, mKeyValuePair<mString, mPtr<mTexture>>(name, texture)));

    mRETURN_SUCCESS();
  }
};

template <>
struct mMeshFactory_Internal_TextureParameterUnpacker <mPtr<mTexture>>
{
  static mFUNCTION(Unpack, mPtr<mQueue<mKeyValuePair<mString, mPtr<mTexture>>>> *pTextureArray, const size_t index, mPtr<mTexture> texture)
  {
    mFUNCTION_SETUP();

    char name[64];
    mERROR_IF(0 > sprintf_s(name, "_" mRenderObjectParam_TextureAttributeName "%" PRIu64, index), mR_InternalError);

    mERROR_CHECK(mQueue_PushBack(*pTextureArray, mKeyValuePair<mString, mPtr<mTexture>>(name, texture)));

    mRETURN_SUCCESS();
  }
};

template <typename T, typename ...Args>
struct mMeshFactory_Internal_TextureParameterUnpacker <T, Args...>
{
  static mFUNCTION(Unpack, mPtr<mQueue<mKeyValuePair<mString, mPtr<mTexture>>>> *pTextureArray, const size_t index, T t, Args&&... args)
  {
    mFUNCTION_SETUP();

    mERROR_CHECK(mMeshFactory_Internal_TextureParameterUnpacker<T>::Unpack(pTextureArray, index, t));
    mERROR_CHECK(mMeshFactory_Internal_TextureParameterUnpacker<Args...>::Unpack(pTextureArray, index + 1, std::forward<Args>(args)...));

    mRETURN_SUCCESS();
  }
};

template<typename ...Args, typename ...TTextures>
inline mFUNCTION(mMeshFactory_CreateMesh, mPtr<mMeshFactory<Args...>> &factory, OUT mPtr<mMesh> *pMesh, IN mAllocator *pAllocator, TTextures... textures)
{
  mFUNCTION_SETUP();

  mERROR_IF(factory == nullptr || pMesh == nullptr, mR_ArgumentNull);

  mPtr<mShader> shader;
  mDEFER_CALL(&shader, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Allocate(&shader, pAllocator, (std::function<void(mShader *)>)[](mShader *pShader) { mShader_Destroy(pShader); }, 1));
  mERROR_CHECK(mShader_Create(shader.GetPointer(), factory->meshFactoryInternal.vertShader, factory->meshFactoryInternal.fragShader));
  mERROR_CHECK(mShader_Bind(*shader.GetPointer()));

  mERROR_CHECK(mSharedPointer_Allocate(pMesh, pAllocator, (std::function<void(mMesh *)>)[](mMesh *pData) { mMesh_Destroy_Internal(pData); }, 1));

  (*pMesh)->uploadState = mRP_US_NotInitialized;
  (*pMesh)->triangleRenderMode = factory->triangleRenderMode;
  (*pMesh)->shader = shader;

  switch (factory->triangleRenderMode)
  {
  case mRP_VRM_TriangleList:
  case mRP_VRM_TriangleStrip:
  case mRP_VRM_TriangleFan:
  case mRP_VRM_Points:
  case mRP_VRM_LineList:
  case mRP_VRM_LineLoop:
  case mRP_VRM_LineStrip:
  case mRP_VRM_QuadList:
  case mRP_VRM_QuadStrip:
  case mRP_VRM_Polygon:
    (*pMesh)->triangleRenderMode = factory->triangleRenderMode;
    (*pMesh)->primitiveCount = factory->values->writeBytes / factory->meshFactoryInternal.size;
    break;

  default:
    mRETURN_RESULT(mR_OperationNotSupported);
    break;
  }

#if defined (mRENDERER_OPENGL)
  (*pMesh)->hasVbo = true;

  size_t informationCount;
  mERROR_CHECK(mQueue_GetCount(factory->meshFactoryInternal.information, &informationCount));

  mERROR_CHECK(mQueue_Create(&(*pMesh)->information, pAllocator));

  mERROR_CHECK(mShader_Bind(*shader.GetPointer()));

  size_t attributeInformationCount = 0;

  for (size_t i = 0; i < informationCount; ++i)
  {
    mMeshFactory_AttributeInformation *pItem;
    mERROR_CHECK(mQueue_PointerAt(factory->meshFactoryInternal.information, i, &pItem));

    if (pItem->type != mMF_AIT_Attribute)
      continue;

    ++attributeInformationCount;

    const GLint index = glGetAttribLocation(shader->shaderProgram, pItem->name);
    mERROR_IF(index < 0, mR_InternalError);

    mERROR_CHECK(mQueue_PushBack((*pMesh)->information, mMeshAttribute(index, pItem->size, pItem->individualSubTypeSize, 0, pItem->offset, pItem->dataType)));
  }

  (*pMesh)->dataSize = factory->meshFactoryInternal.size;
  glGenBuffers(1, &(*pMesh)->vbo);
  glBindBuffer(GL_ARRAY_BUFFER, (*pMesh)->vbo);
  glBufferData(GL_ARRAY_BUFFER, factory->values->writeBytes, factory->values->pData, GL_STATIC_DRAW);

  for (size_t i = 0; i < attributeInformationCount; ++i)
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

  mERROR_IF(factory->meshFactoryInternal.textureCoordCount != sizeof...(TTextures), mR_InvalidParameter);
  mERROR_CHECK(mQueue_Create(&(*pMesh)->textures, nullptr));
  mERROR_CHECK(mQueue_Reserve((*pMesh)->textures, factory->meshFactoryInternal.textureCoordCount));
  mERROR_CHECK(mMeshFactory_Internal_TextureParameterUnpacker<TTextures...>::Unpack(&(*pMesh)->textures, 0, std::forward<TTextures>(textures)...));

  if (factory->meshFactoryInternal.textureCoordCount > 0)
    (*pMesh)->uploadState = mRP_US_NotUploaded;
  else
    (*pMesh)->uploadState = mRP_US_Ready;

#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mMeshFactory_Destroy_Internal, mMeshFactory<Args...> *pFactory)
{
  mFUNCTION_SETUP();

  mERROR_IF(pFactory == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mBinaryChunk_Destroy(&pFactory->values));
  mERROR_CHECK(mQueue_Destroy(&pFactory->meshFactoryInternal.information));

  mRETURN_SUCCESS();
}

template <typename ...Args, typename T>
mFUNCTION(mMeshFactory_Append_Internal_Unpack, mPtr<mMeshFactory<Args...>> &factory, const size_t /*index*/, const size_t /*offset*/, T param)
{
  mFUNCTION_SETUP();

#pragma warning(push)
#pragma warning(disable: 4127) // constant expression.
  if (T::size == 0)
    mRETURN_SUCCESS();
#pragma warning(pop)

  mERROR_CHECK(mBinaryChunk_WriteData(factory->values, param));

  mRETURN_SUCCESS();
}

template <typename ...Args, typename T, typename ...RenderArgs>
mFUNCTION(mMeshFactory_Append_Internal_Unpack, mPtr<mMeshFactory<Args...>> &factory, const size_t index, const size_t offset, T&& param, RenderArgs&&... renderArgs)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mMeshFactory_Append_Internal_Unpack(factory, index, offset, param));
  mERROR_CHECK(mMeshFactory_Append_Internal_Unpack(factory, index + 1, offset + T::size, std::forward<RenderArgs>(renderArgs)...));

  mRETURN_SUCCESS();
}

template <typename ...Args>
mFUNCTION(mMeshFactory_AppendData, mPtr<mMeshFactory<Args...>> &factory, Args&&... args)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mMeshFactory_Append_Internal_Unpack(factory, 0, 0, std::forward<Args>(args)...));

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mMeshFactory_GrowBack, mPtr<mMeshFactory<Args...>> &factory, const size_t items)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mBinaryChunk_GrowBack(factory->values, factory->meshFactoryInternal.size * items));

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mMeshFactory_Clear, mPtr<mMeshFactory<Args...>> &factory)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mBinaryChunk_ResetWrite(factory->values));

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mMeshAttributeContainer_Create, OUT mPtr<mMeshAttributeContainer> *pMeshAttributeContainer, IN mAllocator *pAllocator, char attributeName[64], const std::initializer_list<T> &items)
{
  mFUNCTION_SETUP();

  mERROR_IF(pMeshAttributeContainer == nullptr, mR_ArgumentNull);
  mERROR_IF(items.size() == 0, mR_InvalidParameter);

  mERROR_CHECK(mSharedPointer_Allocate(pMeshAttributeContainer, pAllocator, (std::function<void(mMeshAttributeContainer *)>)[](mMeshAttributeContainer *pData) {mMeshAttributeContainer_Destroy_Internal(pData); }, 1));

  mERROR_CHECK(mBinaryChunk_Create(&(*pMeshAttributeContainer)->attributes, pAllocator));

  mERROR_CHECK(mMeshAttributeContainer_ResolveType_Internal<T>(*pMeshAttributeContainer));

  (*pMeshAttributeContainer)->attributeCount = items.size();
  mERROR_CHECK(mAllocator_Copy(pAllocator, (*pMeshAttributeContainer)->attributeName, attributeName, 64));
  (*pMeshAttributeContainer)->attributeName[63] = '\0';

  for (T item: items)
    mERROR_CHECK(mBinaryChunk_Write((*pMeshAttributeContainer)->attributes, &item));

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mMeshAttributeContainer_Create, OUT mPtr<mMeshAttributeContainer> *pMeshAttributeContainer, IN mAllocator *pAllocator, char attributeName[64], mPtr<mQueue<T>> &items)
{
  mFUNCTION_SETUP();

  mERROR_IF(pMeshAttributeContainer == nullptr, mR_ArgumentNull);

  size_t queueCount = 0;
  mERROR_CHECK(mQueue_GetCount(items, &queueCount));

  mERROR_IF(queueCount == 0, mR_InvalidParameter);

  mERROR_CHECK(mSharedPointer_Allocate(pMeshAttributeContainer, pAllocator, (std::function<void(mMeshAttributeContainer *)>)[](mMeshAttributeContainer *pData) {mMeshAttributeContainer_Destroy_Internal(pData); }, 1));

  mERROR_CHECK(mBinaryChunk_Create(&(*pMeshAttributeContainer)->attributes, pAllocator));

  mERROR_CHECK(mMeshAttributeContainer_ResolveType_Internal<T>(*pMeshAttributeContainer));

  (*pMeshAttributeContainer)->attributeCount == queueCount;
  mERROR_CHECK(mAllocator_Copy(pAllocator, (*pMeshAttributeContainer)->attributeName, attributeName, 64));
  (*pMeshAttributeContainer)->attributeName[63] = '\0';

  for (size_t i = 0; i < queueCount; ++i)
  {
    T *pItem;
    mERROR_CHECK(mQueue_PointerAt(items, i, &pItem));
    mERROR_CHECK(mBinaryChunk_Write((*pMeshAttributeContainer)->attributes, pItem));
  }

  mRETURN_SUCCESS();
}

template<>
inline mFUNCTION(mMeshAttributeContainer_ResolveType_Internal<float_t>, mPtr<mMeshAttributeContainer> &attributeContainer)
{
  mFUNCTION_SETUP();

  attributeContainer->size = sizeof(float_t);
  attributeContainer->subComponentSize = sizeof(float_t);

#if defined (mRENDERER_OPENGL)
  attributeContainer->dataType = GL_FLOAT;
#endif

  mRETURN_SUCCESS();
}

template<>
inline mFUNCTION(mMeshAttributeContainer_ResolveType_Internal<mVec2f>, mPtr<mMeshAttributeContainer> &attributeContainer)
{
  mFUNCTION_SETUP();

  attributeContainer->size = sizeof(mVec2f);
  attributeContainer->subComponentSize = sizeof(float_t);

#if defined (mRENDERER_OPENGL)
  attributeContainer->dataType = GL_FLOAT;
#endif

  mRETURN_SUCCESS();
}

template<>
inline mFUNCTION(mMeshAttributeContainer_ResolveType_Internal<mVec3f>, mPtr<mMeshAttributeContainer> &attributeContainer)
{
  mFUNCTION_SETUP();

  attributeContainer->size = sizeof(mVec3f);
  attributeContainer->subComponentSize = sizeof(float_t);

#if defined (mRENDERER_OPENGL)
  attributeContainer->dataType = GL_FLOAT;
#endif

  mRETURN_SUCCESS();
}

template<>
inline mFUNCTION(mMeshAttributeContainer_ResolveType_Internal<mVec4f>, mPtr<mMeshAttributeContainer> &attributeContainer)
{
  mFUNCTION_SETUP();

  attributeContainer->size = sizeof(mVec4f);
  attributeContainer->subComponentSize = sizeof(float_t);

#if defined (mRENDERER_OPENGL)
  attributeContainer->dataType = GL_FLOAT;
#endif

  mRETURN_SUCCESS();
}

template<>
inline mFUNCTION(mMeshAttributeContainer_ResolveType_Internal<mVector>, mPtr<mMeshAttributeContainer> &attributeContainer)
{
  mFUNCTION_SETUP();

  attributeContainer->size = sizeof(mVector);
  attributeContainer->subComponentSize = sizeof(float_t);

#if defined (mRENDERER_OPENGL)
  attributeContainer->dataType = GL_FLOAT;
#endif

  mRETURN_SUCCESS();
}

template <typename T>
struct mMeshFactory_Internal_Unpacker <T>
{
  static mFUNCTION(Unpack, mMeshFactory_Internal *pMeshFactory)
  {
    mFUNCTION_SETUP();

    switch (T::type)
    {
    case mROPT_Position:
      mERROR_IF(pMeshFactory->position != 0, mR_InvalidParameter);
      pMeshFactory->position = 1;

      switch (T::subType)
      {
      case mROPST_2dPosition:
        pMeshFactory->position3d = 0;
        mERROR_IF(0 > sprintf_s(pMeshFactory->vertShader, "%s\nin vec2 " mRenderObjectParam_Position_AttributeName "0;", pMeshFactory->vertShader), mR_InternalError);
        break;

      case mROPST_3dPosition:
        pMeshFactory->position3d = 1;
        mERROR_IF(0 > sprintf_s(pMeshFactory->vertShader, "%s\nin vec3 " mRenderObjectParam_Position_AttributeName "0;\nuniform mat4 " mRenderObjectParam_MatrixAttributeName "0;", pMeshFactory->vertShader), mR_InternalError);
        break;

      default:
        mRETURN_RESULT(mR_OperationNotSupported);
        break;
      }

      mERROR_CHECK(mQueue_PushBack(pMeshFactory->information, mMeshFactory_AttributeInformation(mRenderObjectParam_Position_AttributeName "0", T::size, pMeshFactory->size, mMF_AIT_Attribute, GL_FLOAT, sizeof(float_t))));

      break;

    case mROPT_TexCoord:
      ++pMeshFactory->textureCoordCount;
      mERROR_IF(pMeshFactory->textureCoordCount > 2, mR_InvalidParameter);

      switch (T::subType)
      {
      case mROPST_2dTextureCoordinate:
        mERROR_IF(0 > sprintf_s(pMeshFactory->vertShader, "%s\nin vec2 " mMeshTexcoord_AttributeName "%" PRIu64 ";\nout vec2 _" mMeshTexcoord_AttributeName "%" PRIu64 ";", pMeshFactory->vertShader, pMeshFactory->textureCoordCount - 1, pMeshFactory->textureCoordCount - 1), mR_InternalError);
        mERROR_IF(0 > sprintf_s(pMeshFactory->vertShaderImpl, "%s\n\t_" mMeshTexcoord_AttributeName "%" PRIu64 " = " mMeshTexcoord_AttributeName "%" PRIu64 ";", pMeshFactory->vertShaderImpl, pMeshFactory->textureCoordCount - 1, pMeshFactory->textureCoordCount - 1), mR_InternalError);
        mERROR_IF(0 > sprintf_s(pMeshFactory->fragShader, "%s\nin vec2 _" mMeshTexcoord_AttributeName "%" PRIu64 ";\nuniform sampler2D _" mRenderObjectParam_TextureAttributeName "%" PRIu64 ";", pMeshFactory->fragShader, pMeshFactory->textureCoordCount - 1, pMeshFactory->textureCoordCount - 1), mR_InternalError);
        break;

      default:
        mRETURN_RESULT(mR_OperationNotSupported);
        break;
      }

      mERROR_CHECK(mQueue_PushBack(pMeshFactory->information, mMeshFactory_AttributeInformation(mMeshTexcoord::attributeName(pMeshFactory->textureCoordCount - 1).c_str(), T::size, pMeshFactory->size, mMF_AIT_Attribute, GL_FLOAT, sizeof(float_t))));

      break;

    case mROPT_Colour:
      mERROR_IF(pMeshFactory->colour != 0, mR_InvalidParameter);
      pMeshFactory->colour = 1;

      switch (T::subType)
      {
      case mROPST_Colour:
        mERROR_IF(0 > sprintf_s(pMeshFactory->vertShader, "%s\nin vec3 " mMeshColour_AttributeName "0;\nout vec3 _" mMeshColour_AttributeName "0;", pMeshFactory->vertShader), mR_InternalError);
        mERROR_IF(0 > sprintf_s(pMeshFactory->vertShaderImpl, "%s\n\t_" mMeshColour_AttributeName "0 = " mMeshColour_AttributeName "0;", pMeshFactory->vertShaderImpl), mR_InternalError);
        mERROR_IF(0 > sprintf_s(pMeshFactory->fragShader, "%s\nin vec3 _" mMeshColour_AttributeName "0;", pMeshFactory->fragShader), mR_InternalError);
        break;

      default:
        mRETURN_RESULT(mR_OperationNotSupported);
        break;
      }

      mERROR_CHECK(mQueue_PushBack(pMeshFactory->information, mMeshFactory_AttributeInformation(mMeshColour_AttributeName "0", T::size, pMeshFactory->size, mMF_AIT_Attribute, GL_FLOAT, sizeof(float_t))));

      break;

    case mROPT_TextureBlendFactor:
      mERROR_IF(pMeshFactory->textureBlendFactor != 0, mR_InvalidParameter);
      pMeshFactory->textureBlendFactor = 1;

      switch (T::subType)
      {
      case mROPST_TextureBlendFactor:
        mERROR_IF(0 > sprintf_s(pMeshFactory->vertShader, "%s\nin float " mMeshTextureBlendFactor_AttributeName "0;\nout float _" mMeshColour_AttributeName "0;", pMeshFactory->vertShader), mR_InternalError);
        mERROR_IF(0 > sprintf_s(pMeshFactory->vertShaderImpl, "%s\n\t_" mMeshTextureBlendFactor_AttributeName "0 = " mMeshTextureBlendFactor_AttributeName "0;", pMeshFactory->vertShaderImpl), mR_InternalError);
        mERROR_IF(0 > sprintf_s(pMeshFactory->fragShader, "%s\nin vec3 _" mMeshTextureBlendFactor_AttributeName "0;", pMeshFactory->fragShader), mR_InternalError);
        break;

      default:
        mRETURN_RESULT(mR_OperationNotSupported);
        break;
      }

      mERROR_CHECK(mQueue_PushBack(pMeshFactory->information, mMeshFactory_AttributeInformation(mMeshTextureBlendFactor_AttributeName "0", T::size, pMeshFactory->size, mMF_AIT_Attribute, GL_FLOAT, sizeof(float_t))));

      break;

    case mROPT_NormalDirection:
      mERROR_IF(pMeshFactory->normal != 0, mR_InvalidParameter);
      pMeshFactory->normal = 1;

      // TODO: implement.

      mERROR_CHECK(mQueue_PushBack(pMeshFactory->information, mMeshFactory_AttributeInformation("", T::size, pMeshFactory->size, mMF_AIT_Attribute, GL_FLOAT, sizeof(float_t))));

      break;

    case mROPT_AlphaChannel:
      mERROR_IF(pMeshFactory->alpha != 0, mR_InvalidParameter);

      switch (T::subType)
      {
      case mROPST_AlphaChannel:
        pMeshFactory->alpha = 1;
        mERROR_IF(0 > sprintf_s(pMeshFactory->vertShader, "%s\nin float " mMeshAlphaChannel_AttributeName "0;\nout float _" mMeshAlphaChannel_AttributeName "0;", pMeshFactory->vertShader), mR_InternalError);
        mERROR_IF(0 > sprintf_s(pMeshFactory->vertShaderImpl, "%s\n\t_" mMeshAlphaChannel_AttributeName "0 = " mMeshAlphaChannel_AttributeName "0;", pMeshFactory->vertShaderImpl), mR_InternalError);
        mERROR_IF(0 > sprintf_s(pMeshFactory->fragShader, "%s\nin vec3 _" mMeshAlphaChannel_AttributeName "0;", pMeshFactory->fragShader), mR_InternalError);
        break;

      default:
        mRETURN_RESULT(mR_OperationNotSupported);
        break;
      }

      mERROR_CHECK(mQueue_PushBack(pMeshFactory->information, mMeshFactory_AttributeInformation(mMeshAlphaChannel_AttributeName "0", T::size, pMeshFactory->size, mMF_AIT_Attribute, GL_FLOAT, sizeof(float_t))));

      break;

    case mROPT_TextureAlphaFlag:
      // mERROR_IF(pMeshFactory->useTextureAlpha != 0, mR_InvalidParameter); // will be ignored anyways.
      pMeshFactory->useTextureAlpha = 1;

      mERROR_CHECK(mQueue_PushBack(pMeshFactory->information, mMeshFactory_AttributeInformation("", T::size, pMeshFactory->size, mMF_AIT_Flag, GL_FALSE, 1)));

      break;

    case mROPT_Scale:

      switch (T::subType)
      {
      case mROPST_ScaleUniform:
        mERROR_IF(0 > sprintf_s(pMeshFactory->vertShader, "%s\nuniform float " mMeshScaleUniform_AttributeName "0;", pMeshFactory->vertShader), mR_InternalError);
        // mERROR_IF(pMeshFactory->scaleUniform != 0, mR_InvalidParameter); // will be ignored anyways.
        pMeshFactory->scaleUniform = 1;

        mERROR_CHECK(mQueue_PushBack(pMeshFactory->information, mMeshFactory_AttributeInformation(mMeshScaleUniform_AttributeName "0", T::size, pMeshFactory->size, mMF_AIT_Uniform, GL_FLOAT, sizeof(float_t))));

        break;

      case mROPST_Scale2dUniform:
        mERROR_IF(0 > sprintf_s(pMeshFactory->vertShader, "%s\nuniform vec2 " mMeshScale2dUniform_AttributeName "0;", pMeshFactory->vertShader), mR_InternalError);
        // mERROR_IF(pMeshFactory->scale2dUniform != 0, mR_InvalidParameter); // will be ignored anyways.
        pMeshFactory->scale2dUniform = 1;

        mERROR_CHECK(mQueue_PushBack(pMeshFactory->information, mMeshFactory_AttributeInformation(mMeshScale2dUniform_AttributeName "0",  T::size, pMeshFactory->size, mMF_AIT_Uniform, GL_FLOAT, sizeof(float_t))));

        break;

      default:
        mRETURN_RESULT(mR_OperationNotSupported);
        break;
      }

      break;

    default:
      mRETURN_RESULT(mR_OperationNotSupported);
      break;
    }

    pMeshFactory->size += T::size;

    mRETURN_SUCCESS();
  }
};

template <typename T, typename ...Args>
struct mMeshFactory_Internal_Unpacker <T, Args...>
{
  static mFUNCTION(Unpack, mMeshFactory_Internal *pMeshFactory)
  {
    mFUNCTION_SETUP();

    mERROR_IF(pMeshFactory == nullptr, mR_ArgumentNull);

    mERROR_CHECK(mMeshFactory_Internal_Unpacker<T>::Unpack(pMeshFactory));
    mERROR_CHECK(mMeshFactory_Internal_Unpacker<Args...>::Unpack(pMeshFactory));

    mRETURN_SUCCESS();
  }
};

template <typename ...Args>
mFUNCTION(mMeshFactory_Internal_Build, mMeshFactory_Internal *pMeshFactory)
{
  mFUNCTION_SETUP();

  mERROR_IF(pMeshFactory == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mQueue_Create(&pMeshFactory->information, nullptr));

  mERROR_IF(0 > sprintf_s(pMeshFactory->vertShader, "#version 150 core\n\n"), mR_InternalError);
  mERROR_IF(0 > sprintf_s(pMeshFactory->vertShaderImpl, "void main()\n{"), mR_InternalError);
  mERROR_IF(0 > sprintf_s(pMeshFactory->fragShader, "#version 150 core\n\nout vec4 " mMeshOutColour_AttributeName ";\n"), mR_InternalError);

  mERROR_CHECK(mMeshFactory_Internal_Unpacker<Args...>().Unpack(pMeshFactory));

  mERROR_IF(pMeshFactory->size == 0, mR_OperationNotSupported);
  mERROR_IF(pMeshFactory->position == 0, mR_InvalidParameter);
  mERROR_IF(pMeshFactory->textureBlendFactor != 0 && pMeshFactory->textureCoordCount == 0, mR_InvalidParameter);

  if (pMeshFactory->position3d == 0)
    mERROR_IF(0 > sprintf_s(pMeshFactory->vertShaderImpl, "%s\n\tgl_Position = vec4(position0, 0.0, 1.0);", pMeshFactory->vertShaderImpl), mR_InternalError);
  else
    mERROR_IF(0 > sprintf_s(pMeshFactory->vertShaderImpl, "%s\n\tgl_Position = matrix0 * vec4(position0, 1.0);", pMeshFactory->vertShaderImpl), mR_InternalError);

  if (pMeshFactory->scaleUniform)
    mERROR_IF(0 > sprintf_s(pMeshFactory->vertShaderImpl, "%s\n\tgl_Position.xyz *= " mMeshScaleUniform_AttributeName "0;", pMeshFactory->vertShaderImpl), mR_InternalError);

  if (pMeshFactory->scale2dUniform)
    mERROR_IF(0 > sprintf_s(pMeshFactory->vertShaderImpl, "%s\n\tgl_Position.xy *= " mMeshScale2dUniform_AttributeName "0;", pMeshFactory->vertShaderImpl), mR_InternalError);

  mERROR_IF(0 > sprintf_s(pMeshFactory->vertShaderImpl, "%s\n}", pMeshFactory->vertShaderImpl), mR_InternalError);

  mERROR_IF(0 > sprintf_s(pMeshFactory->fragShader, "%s\n\nvoid main()\n{", pMeshFactory->fragShader), mR_InternalError);

  if (pMeshFactory->textureCoordCount > 0)
  {
    if (pMeshFactory->textureBlendFactor)
      mERROR_IF(0 > sprintf_s(pMeshFactory->fragShader, "%s\n\tvec4 ret = mix(texture(" mRenderObjectParam_TextureAttributeName "0, " mMeshTexcoord_AttributeName "0), texture(" mRenderObjectParam_TextureAttributeName "1, " mMeshTexcoord_AttributeName "1), " mMeshTextureBlendFactor_AttributeName ");", pMeshFactory->fragShader), mR_InternalError);
    else
      mERROR_IF(0 > sprintf_s(pMeshFactory->fragShader, "%s\n\tvec4 ret = texture(_" mRenderObjectParam_TextureAttributeName "0, _" mMeshTexcoord_AttributeName "0);", pMeshFactory->fragShader), mR_InternalError);

    if (!pMeshFactory->useTextureAlpha)
      mERROR_IF(0 > sprintf_s(pMeshFactory->fragShader, "%s\n\tret.a = 1.0;", pMeshFactory->fragShader), mR_InternalError);
  }
  else
  {
    mERROR_IF(0 > sprintf_s(pMeshFactory->fragShader, "%s\n\tvec4 ret = vec4(1);", pMeshFactory->fragShader), mR_InternalError);
  }

  if (pMeshFactory->colour)
    mERROR_IF(0 > sprintf_s(pMeshFactory->fragShader, "%s\n\tret *= vec4(_" mMeshColour_AttributeName "0, 1);", pMeshFactory->fragShader), mR_InternalError);

  if (pMeshFactory->alpha)
    mERROR_IF(0 > sprintf_s(pMeshFactory->fragShader, "%s\n\tret.a *= _" mMeshAlphaChannel_AttributeName "0);", pMeshFactory->fragShader), mR_InternalError);

  mERROR_IF(0 > sprintf_s(pMeshFactory->fragShader, "%s\n\t" mMeshOutColour_AttributeName " = ret;\n}", pMeshFactory->fragShader), mR_InternalError);

  mERROR_IF(0 > sprintf_s(pMeshFactory->vertShader, "%s\n\n%s", pMeshFactory->vertShader, pMeshFactory->vertShaderImpl), mR_InternalError);
  pMeshFactory->vertShaderImpl[0] = 0;

  mRETURN_SUCCESS();
}

#endif // mMesh_h__
