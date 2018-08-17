// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef mMesh_h__
#define mMesh_h__

#include "mRenderParams.h"
#include "mShader.h"

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
};

struct mRenderObjectParam
{
  // Base Type.
};

#define mRenderObjectParam_Position_AttributeName "position"
#define mRenderObjectParam_MatrixAttributeName "matrix"
#define mRenderObjectParam_Texcoord_AttributeName "texcoord"
#define mRenderObjectParam_Colour_AttributeName "colour"
#define mRenderObjectParam_TextureBlendFactor_AttributeName "textureMix"
#define mRenderObjectParam_TextureAttributeName "texture"
#define mRenderObjectParam_NormalPosition_AttributeName "normal"
#define mRenderObjectParam_AlphaChannel_AttributeName "alpha"
#define mRenderObjectParam_ScaleUniform_AttributeName "scale"

struct mRenderObjectParam_2dPosition : mRenderObjectParam
{
  mVec2f position;

  mRenderObjectParam_2dPosition(const mVec2f &position) : position(position) {}

  static const mRenderObjectParamType type = mROPT_Position;
  static const mRenderObjectParamSubType subType = mROPST_2dPosition;
  static const size_t size = sizeof(mVec2f);
};

struct mRenderObjectParam_3dPosition : mRenderObjectParam
{
  mVec3f position;

  mRenderObjectParam_3dPosition(const mVec3f &position) : position(position) {}

  static const mRenderObjectParamType type = mROPT_Position;
  static const mRenderObjectParamSubType subType = mROPST_3dPosition;
  static const size_t size = sizeof(mVec3f);
};

struct mRenderObjectParam_Texcoord : mRenderObjectParam
{
  mVec2f position;

  mRenderObjectParam_Texcoord(const mVec2f &position) : position(position) {}

  static const mRenderObjectParamType type = mROPT_TexCoord;
  static const mRenderObjectParamSubType subType = mROPST_2dTextureCoordinate;
  static const size_t size = sizeof(mVec2f);
};

struct mRenderObjectParam_Colour : mRenderObjectParam
{
  mVec3f colour;

  mRenderObjectParam_Colour(const mVec3f &colour) : colour(colour) {}

  static const mRenderObjectParamType type = mROPT_Colour;
  static const mRenderObjectParamSubType subType = mROPST_Colour;
  static const size_t size = sizeof(mVec3f);
};

struct mRenderObjectParam_TextureBlendFactor : mRenderObjectParam
{
  float_t blendFactor;

  mRenderObjectParam_TextureBlendFactor(const float_t blendFactor) : blendFactor(blendFactor) {}

  static const mRenderObjectParamType type = mROPT_TextureBlendFactor;
  static const mRenderObjectParamSubType subType = mROPST_TextureBlendFactor;
  static const size_t size = sizeof(float_t);
};

struct mRenderObjectParam_NormalPosition : mRenderObjectParam
{
  mVec3f direction;

  mRenderObjectParam_NormalPosition(const mVec3f &direction) : direction(direction) {}

  static const mRenderObjectParamType type = mROPT_NormalDirection;
  static const mRenderObjectParamSubType subType = mROPST_3dNormalDirection;
  static const size_t size = sizeof(mVec3f);
};

struct mRenderObjectParam_AlphaChannel : mRenderObjectParam
{
  float_t alpha;

  mRenderObjectParam_AlphaChannel(const float_t alpha) : alpha(alpha) {}

  static const mRenderObjectParamType type = mROPT_AlphaChannel;
  static const mRenderObjectParamSubType subType = mROPST_AlphaChannel;
  static const size_t size = sizeof(float_t);
};

struct mRenderObjectParam_ScaleUniform : mRenderObjectParam
{
  mRenderObjectParam_ScaleUniform() {}

  static const mRenderObjectParamType type = mROPT_Scale;
  static const mRenderObjectParamSubType subType = mROPST_ScaleUniform;
  static const size_t size = 0;
};

struct mRenderObjectParam_TextureAlphaFlag : mRenderObjectParam
{
  mRenderObjectParam_TextureAlphaFlag() {}

  static const mRenderObjectParamType type = mROPT_TextureAlphaFlag;
  static const mRenderObjectParamSubType subType = mROPST_TextureAlphaFlag;
  static const size_t size = 0;
};

enum mMeshFactory_AttributeInformationType
{
  mMF_AIT_Attribute,
  mMF_AIT_Uniform,
};

struct mMeshFactory_AttributeInformation
{
  size_t size;
  size_t offset;
  mMeshFactory_AttributeInformationType type;

  inline mMeshFactory_AttributeInformation() = default;
  inline mMeshFactory_AttributeInformation(const size_t size, const size_t offset, const mMeshFactory_AttributeInformationType type) : size(size), offset(offset), type(type) {}
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
  mPtr<mQueue<uint8_t *>> values;
};


template <typename ...Args>
mFUNCTION(mMeshFactory_Destroy_Internal, mMeshFactory<Args...> *pFactory);

template <typename ...Args>
mFUNCTION(mMeshFactory_Create, mPtr<mMeshFactory<Args...>> *pFactory, mAllocator *pAllocator);

template <typename ...Args>
mFUNCTION(mMeshFactory_Destroy, mPtr<mMeshFactory<Args...>> *pFactory);

struct mMesh
{
  mShader shader;
  void *pData;
  size_t length;
};

//////////////////////////////////////////////////////////////////////////


template <typename ...Args>
mFUNCTION(mMeshFactory_Create, mPtr<mMeshFactory<Args...>> *pFactory, mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mSharedPointer_Allocate(pFactory, pAllocator, (std::function<void(mMeshFactory<Args...> *)>)[](mMeshFactory<Args...> *pData) {mMeshFactory_Destroy_Internal(pData);}, 1));

  mERROR_CHECK(mQueue_Create(&(*pFactory)->values, pAllocator));
  mERROR_CHECK(mQueue_Create(&(*pFactory)->meshFactoryInternal.information, pAllocator));

  mERROR_CHECK(mMeshFactory_Internal_Build<Args...>(&(*pFactory)->meshFactoryInternal));

  mRETURN_SUCCESS();
}

template <typename ...Args>
mFUNCTION(mMeshFactory_Destroy, mPtr<mMeshFactory<Args...>> *pFactory)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mSharedPointer_Destroy(pFactory));

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mMeshFactory_Destroy_Internal, mMeshFactory<Args...>* pFactory)
{
  mFUNCTION_SETUP();

  mERROR_IF(pFactory == nullptr, mR_ArgumentNull);

  printf(pFactory->meshFactoryInternal.vertShader);
  printf("\n");
  printf(pFactory->meshFactoryInternal.vertShaderImpl);
  printf("\n");
  printf(pFactory->meshFactoryInternal.fragShader);
  printf("\n");

  // TODO:
  mASSERT(false, "NOT IMPL");

  mRETURN_SUCCESS();
}

template <typename T>
struct mMeshFactory_Internal_Unpacker <T>
{
  static mFUNCTION(Unpack, mMeshFactory_Internal *pMeshFactory)
  {
    mFUNCTION_SETUP();

    mERROR_CHECK(mQueue_PushBack(pMeshFactory->information, mMeshFactory_AttributeInformation(T::size, pMeshFactory->information, mMF_AIT_Attribute)));

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

      break;

    case mROPT_TexCoord:
      ++pMeshFactory->textureCoordCount;
      mERROR_IF(pMeshFactory->textureCoordCount > 2, mR_InvalidParameter);

      switch (T::subType)
      {
      case mROPST_2dTextureCoordinate:
        mERROR_IF(0 > sprintf_s(pMeshFactory->vertShader, "%s\nin vec2 " mRenderObjectParam_Texcoord_AttributeName "%" PRIu64 ";\nout vec2 _" mRenderObjectParam_Texcoord_AttributeName "%" PRIu64 ";", pMeshFactory->vertShader, pMeshFactory->textureCoordCount - 1, pMeshFactory->textureCoordCount - 1), mR_InternalError);
        mERROR_IF(0 > sprintf_s(pMeshFactory->vertShaderImpl, "%s\n\t_" mRenderObjectParam_Texcoord_AttributeName "%" PRIu64 " = " mRenderObjectParam_Texcoord_AttributeName "%" PRIu64 ";", pMeshFactory->vertShaderImpl, pMeshFactory->textureCoordCount - 1, pMeshFactory->textureCoordCount - 1), mR_InternalError);
        mERROR_IF(0 > sprintf_s(pMeshFactory->fragShader, "%s\nin vec2 _" mRenderObjectParam_Texcoord_AttributeName "%" PRIu64 ";\nuniform sampler2D _" mRenderObjectParam_TextureAttributeName "%" PRIu64 ";", pMeshFactory->fragShader, pMeshFactory->textureCoordCount - 1, pMeshFactory->textureCoordCount - 1), mR_InternalError);
        break;

      default:
        mRETURN_RESULT(mR_OperationNotSupported);
        break;
      }

      break;

    case mROPT_Colour:
      mERROR_IF(pMeshFactory->colour != 0, mR_InvalidParameter);
      pMeshFactory->colour = 1;

      switch (T::subType)
      {
      case mROPST_Colour:
        mERROR_IF(0 > sprintf_s(pMeshFactory->vertShader, "%s\nin vec3 " mRenderObjectParam_Colour_AttributeName "0;\nout vec3 _" mRenderObjectParam_Colour_AttributeName "0;", pMeshFactory->vertShader), mR_InternalError);
        mERROR_IF(0 > sprintf_s(pMeshFactory->vertShaderImpl, "%s\n\t_" mRenderObjectParam_Colour_AttributeName "0 = " mRenderObjectParam_Colour_AttributeName "0;", pMeshFactory->vertShaderImpl), mR_InternalError);
        mERROR_IF(0 > sprintf_s(pMeshFactory->fragShader, "%s\nin vec3 _" mRenderObjectParam_Colour_AttributeName "0;", pMeshFactory->fragShader), mR_InternalError);
        break;

      default:
        mRETURN_RESULT(mR_OperationNotSupported);
        break;
      }

      break;

    case mROPT_TextureBlendFactor:
      mERROR_IF(pMeshFactory->textureBlendFactor != 0, mR_InvalidParameter);
      pMeshFactory->textureBlendFactor = 1;

      switch (T::subType)
      {
      case mROPST_TextureBlendFactor:
        mERROR_IF(0 > sprintf_s(pMeshFactory->vertShader, "%s\nin float " mRenderObjectParam_TextureBlendFactor_AttributeName "0;\nout float _" mRenderObjectParam_Colour_AttributeName "0;", pMeshFactory->vertShader), mR_InternalError);
        mERROR_IF(0 > sprintf_s(pMeshFactory->vertShaderImpl, "%s\n\t_" mRenderObjectParam_Colour_AttributeName "0 = " mRenderObjectParam_Colour_AttributeName "0;", pMeshFactory->vertShaderImpl), mR_InternalError);
        mERROR_IF(0 > sprintf_s(pMeshFactory->fragShader, "%s\nin vec3 _" mRenderObjectParam_Colour_AttributeName "0;", pMeshFactory->fragShader), mR_InternalError);
        break;

      default:
        mRETURN_RESULT(mR_OperationNotSupported);
        break;
      }

      break;

    case mROPT_NormalDirection:
      mERROR_IF(pMeshFactory->normal != 0, mR_InvalidParameter);
      pMeshFactory->normal = 1;

      // TODO: implement.

      break;

    case mROPT_AlphaChannel:
      mERROR_IF(pMeshFactory->alpha != 0, mR_InvalidParameter);
      pMeshFactory->alpha = 1;

      // TODO: implement.

      break;

    case mROPT_TextureAlphaFlag:
      // mERROR_IF(pMeshFactory->useTextureAlpha != 0, mR_InvalidParameter); // will be ignored anyways.
      pMeshFactory->useTextureAlpha = 1;

      // TODO: implement.

      break;

    case mROPT_Scale:
      // mERROR_IF(pMeshFactory->scaleUniform != 0, mR_InvalidParameter); // will be ignored anyways.
      pMeshFactory->scaleUniform = 1;

      // TODO: implement.
      mMeshFactory_AttributeInformation info;
      mERROR_CHECK(mQueue_PopBack(pMeshFactory->information, &info));
      info.type = mMF_AIT_Uniform;
      mERROR_CHECK(mQueue_PushBack(pMeshFactory->information, &info));
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

  mERROR_IF(0 > sprintf_s(pMeshFactory->vertShader, "#version 150 core\nuniform vec2f screenSize0;"), mR_InternalError);
  mERROR_IF(0 > sprintf_s(pMeshFactory->vertShaderImpl, "void main()\n{"), mR_InternalError);
  mERROR_IF(0 > sprintf_s(pMeshFactory->fragShader, "#version 150 core\n\nout vec4 outColour;\nuniform vec2f screenSize0;"), mR_InternalError);

  mERROR_CHECK(mMeshFactory_Internal_Unpacker<Args...>().Unpack(pMeshFactory));

  mERROR_IF(pMeshFactory->size == 0, mR_OperationNotSupported);
  mERROR_IF(pMeshFactory->position == 0, mR_InvalidParameter);
  mERROR_IF(pMeshFactory->textureBlendFactor != 0 && pMeshFactory->textureCoordCount == 0, mR_InvalidParameter);

  if (pMeshFactory->position3d == 0)
    mERROR_IF(0 > sprintf_s(pMeshFactory->vertShaderImpl, "%s\n\tgl_Position = vec4(position0, 0.0, 1.0);", pMeshFactory->vertShaderImpl), mR_InternalError);
  else
    mERROR_IF(0 > sprintf_s(pMeshFactory->vertShaderImpl, "%s\n\tgl_Position = matrix0 * vec4(position0, 1.0);", pMeshFactory->vertShaderImpl), mR_InternalError);

  mERROR_IF(0 > sprintf_s(pMeshFactory->vertShaderImpl, "%s\n\tgl_Position *= " mRenderObjectParam_ScaleUniform_AttributeName ";", pMeshFactory->vertShaderImpl), mR_InternalError);

  mERROR_IF(0 > sprintf_s(pMeshFactory->vertShaderImpl, "%s\n}", pMeshFactory->vertShaderImpl), mR_InternalError);

  mERROR_IF(0 > sprintf_s(pMeshFactory->fragShader, "%s\n\nvoid main()\n{", pMeshFactory->fragShader), mR_InternalError);

  if (pMeshFactory->textureCoordCount > 0)
  {
    if (pMeshFactory->textureBlendFactor)
      mERROR_IF(0 > sprintf_s(pMeshFactory->fragShader, "%s\n\tvec4 ret = mix(texture(" mRenderObjectParam_TextureAttributeName "0, " mRenderObjectParam_Texcoord_AttributeName "0), texture(" mRenderObjectParam_TextureAttributeName "1, " mRenderObjectParam_Texcoord_AttributeName "1), " mRenderObjectParam_TextureBlendFactor_AttributeName ");", pMeshFactory->fragShader), mR_InternalError);
    else
      mERROR_IF(0 > sprintf_s(pMeshFactory->fragShader, "%s\n\tvec4 ret = texture(" mRenderObjectParam_TextureAttributeName "0, " mRenderObjectParam_Texcoord_AttributeName "0);", pMeshFactory->fragShader), mR_InternalError);

    if (!pMeshFactory->useTextureAlpha)
      mERROR_IF(0 > sprintf_s(pMeshFactory->fragShader, "%s\n\tret.a = 1;", pMeshFactory->fragShader), mR_InternalError);
  }
  else
  {
    mERROR_IF(0 > sprintf_s(pMeshFactory->fragShader, "%s\n\tvec4 ret = vec4(1);", pMeshFactory->fragShader), mR_InternalError);
  }

  if (pMeshFactory->colour)
    mERROR_IF(0 > sprintf_s(pMeshFactory->fragShader, "%s\n\tret *= vec4(" mRenderObjectParam_Colour_AttributeName "0, 1);", pMeshFactory->fragShader), mR_InternalError);

  if (pMeshFactory->alpha)
    mERROR_IF(0 > sprintf_s(pMeshFactory->fragShader, "%s\n\tret.a *= " mRenderObjectParam_AlphaChannel_AttributeName "0);", pMeshFactory->fragShader), mR_InternalError);

  mERROR_IF(0 > sprintf_s(pMeshFactory->fragShader, "%s\n\toutColor = ret;\n}", pMeshFactory->fragShader), mR_InternalError);

  mRETURN_SUCCESS();
}

#endif // mMesh_h__
