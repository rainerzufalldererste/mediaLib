#include "mSpriteBatch.h"
// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

template <typename ...Args>
mFUNCTION(mSpriteBatch_Create_Internal, IN_OUT mSpriteBatch<Args...> *pSpriteBatch);

template <typename ...Args>
mFUNCTION(mSpriteBatch_Destroy_Internal, IN_OUT mSpriteBatch<Args...> *pSpriteBatch);

//////////////////////////////////////////////////////////////////////////

template<typename ...Args>
inline mFUNCTION(mSpriteBatch_Create, OUT mPtr<mSpriteBatch<Args...>> *pSpriteBatch, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mSharedPointer_Allocate(pSpriteBatch, pAllocator, (std::function<void(mSpriteBatch<Args...> *)>)[](mSpriteBatch<Args...> *pData) {mSpriteBatch_Destroy_Internal(pData);}, 1));
  mERROR_CHECK(mSpriteBatch_Create_Internal(pSpriteBatch->GetPointer()));

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mSpriteBatch_Destroy, IN_OUT mPtr<mSpriteBatch<Args...>> *pSpriteBatch)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSpriteBatch == nullptr, mR_ArgumentNull);
  mASSERT_DEBUG((*pSpriteBatch)->isStarted == false, "The sprite batch should currently not be started.");

  mERROR_CHECK(mSharedPointer_Destroy(pSpriteBatch));

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

struct mSpriteBatch_ShaderParams
{
  uint8_t textureCrop : 1;
  uint8_t colour : 1;
  uint8_t rotation : 1;
  uint8_t matrixTransform : 1;
  uint8_t textureFlip : 1;
};

template<typename ...Args>
struct mSpriteBatch_GenerateShader;

template<typename T>
struct mSpriteBatch_GenerateShader <T>
{
  static mFUNCTION(UnpackParams, mSpriteBatch_ShaderParams *pParams)
  {
    mFUNCTION_SETUP();

    mERROR_IF(pParams == nullptr, mR_ArgumentNull);

    switch (T::type)
    {
    case mSBE_T_Colour:
      mASSERT_DEBUG(pParams->colour == 0, "Colour parameter cannot be set twice. (will be ignored.)");
      pParams->colour = 1;
      break;

    case mSBE_T_TextureCrop:
      mASSERT_DEBUG(pParams->textureCrop == 0, "TextureCrop parameter cannot be set twice. (will be ignored.)");
      pParams->textureCrop = 1;
      break;

    case mSBE_T_Rotation:
      mASSERT_DEBUG(pParams->rotation == 0, "Rotation parameter cannot be set twice. (will be ignored.)");
      pParams->rotation = 1;
      break;

    case mSBE_T_MatrixTransform:
      mASSERT_DEBUG(pParams->matrixTransform == 0, "MatrixTransform parameter cannot be set twice. (will be ignored.)");
      pParams->matrixTransform = 1;
      break;

    case mSBE_T_TextureFlip:
      mASSERT_DEBUG(pParams->textureFlip == 0, "TextureFlip parameter cannot be set twice. (will be ignored.)");
      pParams->textureFlip = 1;
      break;

    default:
      mRETURN_RESULT(mR_OperationNotSupported);
      break;
    }

    mRETURN_SUCCESS();
  }
};

template<typename T, typename ...Args>
struct mSpriteBatch_GenerateShader <T, Args...>
{
  static mFUNCTION(UnpackParams, mSpriteBatch_ShaderParams *pParams)
  {
    mFUNCTION_SETUP();

    mERROR_CHECK(mSpriteBatch_GenerateShader<T>::UnpackParams(pParams));
    mERROR_CHECK(mSpriteBatch_GenerateShader<Args...>::UnpackParams(pParams));

    mRETURN_SUCCESS();
  }
};

template<typename ...Args>
inline mFUNCTION(mSpriteBatch_Create_Internal, IN_OUT mSpriteBatch<Args...>* pSpriteBatch)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSpriteBatch == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mQueue_Create(&pSpriteBatch->enqueuedRenderObjects, nullptr));

  pSpriteBatch->isStarted = false;
  pSpriteBatch->alphaMode = mSpriteBatch_AlphaMode::mSB_AM_AlphaBlend;
  pSpriteBatch->spriteSortMode = mSpriteBatch_SpriteSortMode::mSB_SSM_BackToFront;
  pSpriteBatch->textureSampleMode = mSpriteBatch_TextureSampleMode::mSB_TSM_LinearFiltering;

  mSpriteBatch_ShaderParams params = { 0 };
  mERROR_CHECK(mSpriteBatch_GenerateShader<Args...>::UnpackParams(&params));

  // Vertex Shader.
  char vertexShader[1024] = "";

  mERROR_IF(0 > sprintf_s(vertexShader, "#version 150 core\n\nin vec2 position0;\nin vec2 texCoord0;\nout vec2 _texCoord0;\n\nuniform vec2 screenSize0;\nuniform vec3 startOffset0;\nuniform vec2 scale0;\n"), mR_InternalError);

  if (params.rotation)
    mERROR_IF(0 > sprintf_s(vertexShader, "%s\nuniform float " mSBERotation_UniformName ";", vertexShader), mR_InternalError);

  if (params.matrixTransform)
    mERROR_IF(0 > sprintf_s(vertexShader, "%s\nuniform mat4 " mSBEMatrixTransform_UniformName ";", vertexShader), mR_InternalError);

  mERROR_IF(0 > sprintf_s(vertexShader, "%s\n\nvoid main()\n{\n\t_texCoord0 = texCoord0;\n\tvec4 position = vec4(position0, 0.0, 1.0);\n", vertexShader), mR_InternalError);

  if (params.rotation)
    mERROR_IF(0 > sprintf_s(vertexShader, "%s\n\tposition.xy -= 0.5;\n\tvec2 oldPos = position.xy;\t\nposition.x = sin(" mSBERotation_UniformName ") * oldPos.y + cos(" mSBERotation_UniformName ") * oldPos.x;\n\tposition.y = sin(" mSBERotation_UniformName ") * oldPos.x + cos(" mSBERotation_UniformName ") * oldPos.y;\n\tposition.xy += 0.5;", vertexShader), mR_InternalError);

  mERROR_IF(0 > sprintf_s(vertexShader, "%s\n\tposition.xy = ((position.xy * scale0) + startOffset0.xy);", vertexShader), mR_InternalError);

  if (params.matrixTransform)
    mERROR_IF(0 > sprintf_s(vertexShader, "%s\n\tposition *= " mSBEMatrixTransform_UniformName ";", vertexShader), mR_InternalError);

  mERROR_IF(0 > sprintf_s(vertexShader, "%s\n\tposition.xy /= screenSize0;\n\tposition.z = startOffset0.z;\n\n\tgl_Position = position;\n}\n", vertexShader), mR_InternalError);

  // Fragment Shader.
  char fragmentShader[1024] = "";

  mERROR_IF(0 > sprintf_s(fragmentShader, "#version 150 core\n\nout vec4 fragColour0;\n\nin vec2 _texCoord0;\nuniform sampler2D texture0;\n"), mR_InternalError);

  if (params.colour)
    mERROR_IF(0 > sprintf_s(fragmentShader, "%s\nuniform vec4 " mSBEColour_UniformName ";", fragmentShader), mR_InternalError);

  if (params.textureFlip)
    mERROR_IF(0 > sprintf_s(fragmentShader, "%s\nuniform vec2 " mSBETextureFlip_UniformName ";", fragmentShader), mR_InternalError);

  if (params.textureCrop)
    mERROR_IF(0 > sprintf_s(fragmentShader, "%s\nuniform vec4 " mSBETextureCrop_UniformName ";", fragmentShader), mR_InternalError);

  mERROR_IF(0 > sprintf_s(fragmentShader, "%s\n\nvoid main()\n{\n\tvec2 texturePosition = _texCoord0;", fragmentShader), mR_InternalError);

  if (params.textureFlip)
    mERROR_IF(0 > sprintf_s(fragmentShader, "%s\n\ttexturePosition = texturePosition * (1 - 2 * " mSBETextureFlip_UniformName ") + " mSBETextureFlip_UniformName ";", fragmentShader), mR_InternalError);

  if (params.textureCrop)
    mERROR_IF(0 > sprintf_s(fragmentShader, "%s\n\ttexturePosition *= (" mSBETextureCrop_UniformName ".zw - " mSBETextureCrop_UniformName ".xy);\n\ttexturePosition += " mSBETextureCrop_UniformName ".xy;", fragmentShader), mR_InternalError);

  mERROR_IF(0 > sprintf_s(fragmentShader, "%s\n\tfragColour0 = texture(texture0, texturePosition);", fragmentShader), mR_InternalError);

  if (params.colour)
    mERROR_IF(0 > sprintf_s(fragmentShader, "%s\n\tfragColour0 *= " mSBEColour_UniformName ";", fragmentShader), mR_InternalError);

  mERROR_IF(0 > sprintf_s(fragmentShader, "%s\n}\n", fragmentShader), mR_InternalError);

  printf(vertexShader);
  printf("\n\n");
  printf(fragmentShader);
  printf("\n\n");

  mERROR_CHECK(mSharedPointer_Allocate(&pSpriteBatch->shader, nullptr, (std::function<void(mShader *)>)[](mShader *pData) {mShader_Destroy(pData);}, 1));
  mERROR_CHECK(mShader_Create(pSpriteBatch->shader.GetPointer(), vertexShader, fragmentShader, "fragColour0"));

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mSpriteBatch_Destroy_Internal, IN_OUT mSpriteBatch<Args...> *pSpriteBatch)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSpriteBatch == nullptr, mR_ArgumentNull);

  size_t queueSize = 0;
  mERROR_CHECK(mQueue_GetCount(pSpriteBatch->enqueuedRenderObjects, &queueSize));

  for (size_t i = 0; i < queueSize; ++i)
  {
    mSpriteBatch_Internal_RenderObject<Args...> renderObject;
    mERROR_CHECK(mQueue_PopFront(pSpriteBatch->enqueuedRenderObjects, &renderObject));
    mERROR_CHECK(mSpriteBatch_Internal_RenderObject_Destroy(&renderObject));
  }

  mERROR_CHECK(mSharedPointer_Destroy(&pSpriteBatch->shader));

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

template<typename ...Args>
inline mFUNCTION(mSpriteBatch_Internal_RenderObject_Create, OUT mSpriteBatch_Internal_RenderObject<Args...>* pRenderObject, const mPtr<mTexture>& texture, const mVec2f position, const mVec2f size, const float_t depth, Args && ...args)
{
  mFUNCTION_SETUP();

  mERROR_IF(pRenderObject == nullptr, mR_ArgumentNull);

  pRenderObject->texture = texture;
  pRenderObject->position = position;
  pRenderObject->size = size;
  pRenderObject->depth = depth;
  pRenderObject->args = std::make_tuple<Args...>(std::forward<Args>(args)...);

  mRETURN_SUCCESS();
}

template <typename ...Args>
mFUNCTION(mSpriteBatch_Internal_RenderObject_Destroy, IN_OUT mSpriteBatch_Internal_RenderObject<Args...> *pRenderObject)
{
  mFUNCTION_SETUP();

  mERROR_IF(pRenderObject == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Destroy(&pRenderObject->texture));
  pRenderObject->args.~tuple();

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mSpriteBatch_Internal_RenderObject_Render, mSpriteBatch_Internal_RenderObject<Args...> &renderObject)
{
  mFUNCTION_SETUP();

  mUnused(renderObject);
  // TODO: Implement;

  mRETURN_SUCCESS();
}
