#include "mSpriteBatch.h"
#include "mForwardTuple.h"

#define mSB_MultisampleCountUniformName "sampleCount0"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "53ORK5GZrCgx9g+p1I5asOmL4y6LNRF/xvfm3mqU9obt1rkVxDsFhtyqActkc//c3IBGNy9dZOqSoL8F"
#endif

template <typename ...Args>
mFUNCTION(mSpriteBatch_Create_Internal, IN_OUT mSpriteBatch<Args...> *pSpriteBatch);

template <typename ...Args>
mFUNCTION(mSpriteBatch_Destroy_Internal, IN_OUT mSpriteBatch<Args...> *pSpriteBatch);

template <typename ...Args>
mFUNCTION(mSpriteBatch_Internal_SetAlphaBlending, mPtr<mSpriteBatch<Args...>> &spriteBatch);

template <typename ...Args>
mFUNCTION(mSpriteBatch_Internal_SetDrawOrder, mPtr<mSpriteBatch<Args...>> &spriteBatch);

template <typename ...Args>
mFUNCTION(mSpriteBatch_Internal_SetTextureFilterMode, mPtr<mSpriteBatch<Args...>> &spriteBatch);

template <typename ...Args>
mFUNCTION(mSpriteBatch_Internal_InitializeMesh, mPtr<mSpriteBatch<Args...>> &spriteBatch);

template <typename ...Args>
mFUNCTION(mSpriteBatch_Internal_BindMesh, mPtr<mSpriteBatch<Args...>> &spriteBatch, mPtr<mShader> &shader);

template <typename ...Args>
bool mSpriteBatch_Internal_DrawInstantly(mPtr<mSpriteBatch<Args...>> &spriteBatch)
{
  return spriteBatch->spriteSortMode == mSB_SSM_None || spriteBatch->alphaMode == mSB_AM_Additive || spriteBatch->alphaMode == mSB_AM_NoAlpha;
}

//////////////////////////////////////////////////////////////////////////

template<typename ...Args>
inline mFUNCTION(mSpriteBatch_Create, OUT mPtr<mSpriteBatch<Args...>> *pSpriteBatch, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mSharedPointer_Allocate(pSpriteBatch, pAllocator, (std::function<void(mSpriteBatch<Args...> *)>)[](mSpriteBatch<Args...> *pData) {mSpriteBatch_Destroy_Internal(pData);}, 1));
  mERROR_CHECK(mSpriteBatch_Create_Internal(pSpriteBatch->GetPointer()));
  mERROR_CHECK(mSpriteBatch_Internal_InitializeMesh(*pSpriteBatch));

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mSpriteBatch_Destroy, IN_OUT mPtr<mSpriteBatch<Args...>> *pSpriteBatch)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSpriteBatch == nullptr || *pSpriteBatch == nullptr, mR_ArgumentNull);
  mASSERT_DEBUG((*pSpriteBatch)->isStarted == false, "The sprite batch should currently not be started.");

  mERROR_CHECK(mSharedPointer_Destroy(pSpriteBatch));

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mSpriteBatch_Begin, mPtr<mSpriteBatch<Args...>> &spriteBatch)
{
  mFUNCTION_SETUP();

  mERROR_IF(spriteBatch == nullptr, mR_ArgumentNull);
  mERROR_IF(spriteBatch->isStarted, mR_ResourceStateInvalid);

  spriteBatch->isStarted = true;

  if (mSpriteBatch_Internal_DrawInstantly(spriteBatch))
  {
    mERROR_CHECK(mSpriteBatch_Internal_BindMesh(spriteBatch, spriteBatch->shader));
    spriteBatch->lastWasMultisampleTexture = false;

    mERROR_CHECK(mSpriteBatch_Internal_SetAlphaBlending(spriteBatch));
    mERROR_CHECK(mSpriteBatch_Internal_SetDrawOrder(spriteBatch));
    mERROR_CHECK(mSpriteBatch_Internal_SetTextureFilterMode(spriteBatch));
  }

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mSpriteBatch_Begin, mPtr<mSpriteBatch<Args...>> &spriteBatch, const mSpriteBatch_SpriteSortMode spriteSortMode, const mSpriteBatch_AlphaMode alphaMode, const mSpriteBatch_TextureSampleMode sampleMode /* = mSB_TSM_LinearFiltering */)
{
  mFUNCTION_SETUP();

  mERROR_IF(spriteBatch == nullptr, mR_ArgumentNull);
  mERROR_IF(spriteBatch->isStarted, mR_ResourceStateInvalid);

  spriteBatch->spriteSortMode = spriteSortMode;
  spriteBatch->alphaMode = alphaMode;
  spriteBatch->textureSampleMode = sampleMode;

  mERROR_CHECK(mSpriteBatch_Begin(spriteBatch));

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mSpriteBatch_DrawWithDepth, mPtr<mSpriteBatch<Args...>> &spriteBatch, mPtr<mTexture> &texture, const mVec2f &position, const float_t depth, Args &&...args)
{
  mFUNCTION_SETUP();

  mERROR_IF(texture == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSpriteBatch_DrawWithDepth(spriteBatch, texture, mRectangle<float_t>(position, texture->resolutionF), depth, std::forward<Args>(args)...));

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mSpriteBatch_DrawWithDepth, mPtr<mSpriteBatch<Args...>> &spriteBatch, mPtr<mTexture> &texture, const mRectangle2D<float_t> &rect, const float_t depth, Args &&...args)
{
  mFUNCTION_SETUP();

  mERROR_IF(spriteBatch == nullptr || texture == nullptr, mR_ArgumentNull);
  mERROR_IF(!spriteBatch->isStarted, mR_ResourceStateInvalid);

  mSpriteBatch_Internal_RenderObject<Args...> renderObject;
  mERROR_CHECK(mSpriteBatch_Internal_RenderObject_Create(&renderObject, texture, rect.position, rect.size, depth, std::forward<Args>(args)...));

  if (mSpriteBatch_Internal_DrawInstantly(spriteBatch))
  {
    const bool isMultisampleTexture = texture->sampleCount != 0;
    mPtr<mShader> &shader = isMultisampleTexture ? spriteBatch->multisampleShader : spriteBatch->shader;

    if (!mShader_IsActive(*shader))
    {
      mERROR_CHECK(mSpriteBatch_Internal_BindMesh(spriteBatch, shader));
      spriteBatch->lastWasMultisampleTexture = isMultisampleTexture;

      mERROR_CHECK(mSpriteBatch_Internal_SetAlphaBlending(spriteBatch));
      mERROR_CHECK(mSpriteBatch_Internal_SetDrawOrder(spriteBatch));
      mERROR_CHECK(mSpriteBatch_Internal_SetTextureFilterMode(spriteBatch));
    }

    mERROR_CHECK(mSpriteBatch_Internal_RenderObject_Render_Raw(renderObject, spriteBatch, *texture, shader));
  }
  else
  {
    mERROR_CHECK(mQueue_PushBack(spriteBatch->enqueuedRenderObjects, renderObject));
  }

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mSpriteBatch_DrawWithDepth, mPtr<mSpriteBatch<Args...>> &spriteBatch, mPtr<mFramebuffer> &framebuffer, const mVec2f &position, const float_t depth, Args &&...args)
{
  mFUNCTION_SETUP();

  mERROR_IF(framebuffer == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSpriteBatch_DrawWithDepth(spriteBatch, framebuffer, mRectangle<float_t>(position, (mVec2f)framebuffer->size), depth, std::forward<Args>(args)...));
  
  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mSpriteBatch_DrawWithDepth, mPtr<mSpriteBatch<Args...>> &spriteBatch, mPtr<mFramebuffer> &framebuffer, const mRectangle2D<float_t> &rect, const float_t depth, Args &&...args)
{
  mFUNCTION_SETUP();

  mERROR_IF(spriteBatch == nullptr || framebuffer == nullptr, mR_ArgumentNull);
  mERROR_IF(!spriteBatch->isStarted, mR_ResourceStateInvalid);

  if (mSpriteBatch_Internal_DrawInstantly(spriteBatch))
  {
    mTexture texture;
    mDEFER_CALL(&texture, mTexture_Destroy);
    mERROR_CHECK(mTexture_CreateFromUnownedIndex(&texture, framebuffer->textureId, 0, framebuffer->sampleCount));

    mERROR_CHECK(mSpriteBatch_DrawWithDepth(spriteBatch, &texture, rect, depth, std::forward<Args>(args)...));
  }
  else
  {
#if defined(_DEBUG) && !defined(GIT_BUILD)
    static bool first = true;

    if (first)
    {
      first = false;
      mFAIL("Please be aware, that this functionality can cause horrible issues if you destroy the related framebuffer before calling `mSpriteBatch_End`.");
    }
#endif

    mPtr<mTexture> texture;
    mDEFER_CALL(&texture, mSharedPointer_Destroy);
    mERROR_CHECK(mSharedPointer_Allocate<mTexture>(&texture, &mDefaultTempAllocator, [](mTexture *pTexture) { mTexture_Destroy(pTexture); }, 1));
    mERROR_CHECK(mTexture_CreateFromUnownedIndex(texture.GetPointer(), framebuffer->textureId, 0, framebuffer->sampleCount));

    mERROR_CHECK(mSpriteBatch_DrawWithDepth(spriteBatch, texture, rect, depth, std::forward<Args>(args)...));
  }

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mSpriteBatch_DrawWithDepth, mPtr<mSpriteBatch<Args...>> &spriteBatch, mTexture *pTexture, const mVec2f &position, const float_t depth, Args &&...args)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTexture == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSpriteBatch_DrawWithDepth(spriteBatch, pTexture, mRectangle2D<float_t>(position, pTexture->resolutionF), depth, std::forward<Args>(args)...));

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mSpriteBatch_DrawWithDepth, mPtr<mSpriteBatch<Args...>> &spriteBatch, mTexture *pTexture, const mRectangle2D<float_t> &rect, const float_t depth, Args &&...args)
{
  mFUNCTION_SETUP();

  mERROR_IF(spriteBatch == nullptr || pTexture == nullptr, mR_ArgumentNull);
  mERROR_IF(!spriteBatch->isStarted, mR_ResourceStateInvalid);

  if (mSpriteBatch_Internal_DrawInstantly(spriteBatch))
  {
    const bool isMultisampleTexture = pTexture->sampleCount != 0;
    mPtr<mShader> &shader = isMultisampleTexture ? spriteBatch->multisampleShader : spriteBatch->shader;

    if (!mShader_IsActive(*shader))
    {
      mERROR_CHECK(mSpriteBatch_Internal_BindMesh(spriteBatch, shader));
      spriteBatch->lastWasMultisampleTexture = isMultisampleTexture;

      mERROR_CHECK(mSpriteBatch_Internal_SetAlphaBlending(spriteBatch));
      mERROR_CHECK(mSpriteBatch_Internal_SetDrawOrder(spriteBatch));
      mERROR_CHECK(mSpriteBatch_Internal_SetTextureFilterMode(spriteBatch));
    }

    mSpriteBatch_Internal_RenderObject<Args...> renderObject;
    mERROR_CHECK(mSpriteBatch_Internal_RenderObject_Create(&renderObject, nullptr, rect.position, rect.size, depth, std::forward<Args>(args)...));

    mERROR_CHECK(mSpriteBatch_Internal_RenderObject_Render_Raw(renderObject, spriteBatch, *pTexture, shader));
  }
  else
  {
#if defined(_DEBUG) && !defined(GIT_BUILD)
    static bool first = true;

    if (first)
    {
      first = false;
      mFAIL("Please be aware, that this functionality can cause horrible issues if you destroy the related texture before calling `mSpriteBatch_End`.");
    }
#endif

    mPtr<mTexture> texture;
    mDEFER_CALL(&texture, mSharedPointer_Destroy);
    mERROR_CHECK(mSharedPointer_Create(&texture, pTexture, mSHARED_POINTER_FOREIGN_RESOURCE));

    mERROR_CHECK(mSpriteBatch_DrawWithDepth(spriteBatch, texture, rect, depth, std::forward<Args>(args)...));
  }

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mSpriteBatch_Draw, mPtr<mSpriteBatch<Args...>> &spriteBatch, mPtr<mTexture> &texture, const mVec2f &position, Args &&...args)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mSpriteBatch_DrawWithDepth(spriteBatch, texture, position, 0, std::forward<Args>(args)...));

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mSpriteBatch_Draw, mPtr<mSpriteBatch<Args...>> &spriteBatch, mPtr<mTexture> &texture, const mRectangle2D<float_t> &rect, Args &&...args)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mSpriteBatch_DrawWithDepth(spriteBatch, texture, rect, 0, std::forward<Args>(args)...));

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mSpriteBatch_Draw, mPtr<mSpriteBatch<Args...>> &spriteBatch, mPtr<mFramebuffer> &framebuffer, const mVec2f &position, Args &&...args)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mSpriteBatch_DrawWithDepth(spriteBatch, framebuffer, position, 0, std::forward<Args>(args)...));

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mSpriteBatch_Draw, mPtr<mSpriteBatch<Args...>> &spriteBatch, mPtr<mFramebuffer> &framebuffer, const mRectangle2D<float_t> &rect, Args &&...args)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mSpriteBatch_DrawWithDepth(spriteBatch, framebuffer, rect, 0, std::forward<Args>(args)...));

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mSpriteBatch_Draw, mPtr<mSpriteBatch<Args...>> &spriteBatch, mTexture *pTexture, const mVec2f &position, Args &&...args)
{
  mFUNCTION_SETUP();

  mERROR_IF(spriteBatch == nullptr || pTexture == nullptr, mR_ArgumentNull);
  mERROR_IF(!spriteBatch->isStarted, mR_ResourceStateInvalid);

  mERROR_CHECK(mSpriteBatch_DrawWithDepth(spriteBatch, pTexture, position, 0, std::forward<Args>(args)...));

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mSpriteBatch_Draw, mPtr<mSpriteBatch<Args...>> &spriteBatch, mTexture *pTexture, const mRectangle2D<float_t> &rect, Args &&...args)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mSpriteBatch_DrawWithDepth(spriteBatch, pTexture, rect, 0, std::forward<Args>(args)...));

  mRETURN_SUCCESS();
}

template <typename ...Args>
mFUNCTION(mSpriteBatch_QuickSortRenderObjects, mPtr<mQueue<mSpriteBatch_Internal_RenderObject<Args...>>> &queue, const size_t left, const size_t right)
{
  mFUNCTION_SETUP();

  if (left == right)
    mRETURN_SUCCESS();

  size_t l = left;
  size_t r = right;
  const size_t pivotIndex = (left + right) / 2;

  mSpriteBatch_Internal_RenderObject<Args...> *pRenderObject;
  mERROR_CHECK(mQueue_PointerAt(queue, pivotIndex, &pRenderObject));

  const float_t pivot = pRenderObject->position.z;

  while (l <= r) 
  {
    mSpriteBatch_Internal_RenderObject<Args...> *pRenderObjectL = nullptr;
    mSpriteBatch_Internal_RenderObject<Args...> *pRenderObjectR = nullptr;

    while (true)
    {
      mERROR_CHECK(mQueue_PointerAt(queue, l, &pRenderObjectL));

      if (pRenderObjectL->position.z < pivot)
        l++;
      else
        break;
    }
    
    while (true)
    {
      mERROR_CHECK(mQueue_PointerAt(queue, r, &pRenderObjectR));

      if (pRenderObjectR->position.z > pivot)
        r--;
      else
        break;
    }

    if (l <= r) 
    {
      std::swap(*pRenderObjectL, *pRenderObjectR);

      l++;
      r--;
    }
  };

  if (left < r)
    mERROR_CHECK(mSpriteBatch_QuickSortRenderObjects(queue, left, r));

  if (l < right)
    mERROR_CHECK(mSpriteBatch_QuickSortRenderObjects(queue, l, right));

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mSpriteBatch_End, mPtr<mSpriteBatch<Args...>> &spriteBatch)
{
  mFUNCTION_SETUP();

  mERROR_IF(spriteBatch == nullptr, mR_ArgumentNull);
  mERROR_IF(!spriteBatch->isStarted, mR_ResourceStateInvalid);
  spriteBatch->isStarted = false;

  if (!mSpriteBatch_Internal_DrawInstantly(spriteBatch))
  {
    mERROR_CHECK(mSpriteBatch_Internal_BindMesh(spriteBatch, spriteBatch->shader));
    spriteBatch->lastWasMultisampleTexture = false;

    mERROR_CHECK(mSpriteBatch_Internal_SetAlphaBlending(spriteBatch));
    mERROR_CHECK(mSpriteBatch_Internal_SetDrawOrder(spriteBatch));
    mERROR_CHECK(mSpriteBatch_Internal_SetTextureFilterMode(spriteBatch));

    size_t count;
    mERROR_CHECK(mQueue_GetCount(spriteBatch->enqueuedRenderObjects, &count));

    if (count == 0)
      mRETURN_SUCCESS();

    mERROR_CHECK(mSpriteBatch_QuickSortRenderObjects(spriteBatch->enqueuedRenderObjects, 0, count - 1));

    if (spriteBatch->spriteSortMode == mSpriteBatch_SpriteSortMode::mSB_SSM_BackToFront)
    {
      for (size_t i = 0; i < count; ++i)
      {
        mSpriteBatch_Internal_RenderObject<Args...> renderObject;
        mERROR_CHECK(mQueue_PopFront(spriteBatch->enqueuedRenderObjects, &renderObject));
        mDEFER_CALL(&renderObject, mSpriteBatch_Internal_RenderObject_Destroy);
        mERROR_CHECK(mSpriteBatch_Internal_RenderObject_Render(renderObject, spriteBatch));
      }
    }
    else
    {
      for (int64_t i = (int64_t)count - 1; i >= 0; i--)
      {
        mSpriteBatch_Internal_RenderObject<Args...> renderObject;
        mERROR_CHECK(mQueue_PopBack(spriteBatch->enqueuedRenderObjects, &renderObject));
        mDEFER_CALL(&renderObject, mSpriteBatch_Internal_RenderObject_Destroy);
        mERROR_CHECK(mSpriteBatch_Internal_RenderObject_Render(renderObject, spriteBatch));
      }
    }

  }
  else
  {
    size_t count;
    mERROR_CHECK(mQueue_GetCount(spriteBatch->enqueuedRenderObjects, &count));

    for (size_t i = 0; i < count; ++i)
    {
      mSpriteBatch_Internal_RenderObject<Args...> renderObject;
      mERROR_CHECK(mQueue_PopFront(spriteBatch->enqueuedRenderObjects, &renderObject));
      mERROR_CHECK(mSpriteBatch_Internal_RenderObject_Destroy(&renderObject));
    }
  }

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

template<typename ...Args>
struct mSpriteBatch_GenerateShader;

template<>
struct mSpriteBatch_GenerateShader <>
{
  static mFUNCTION(UnpackParams, mSpriteBatch_ShaderParams *)
  {
    mFUNCTION_SETUP();
    mRETURN_SUCCESS();
  }
};

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
      mERROR_IF(pParams->colour != 0, mR_InvalidParameter); // Colour parameter cannot be set twice.
      pParams->colour = 1;
      break;

    case mSBE_T_TextureCrop:
      mERROR_IF(pParams->textureCrop != 0, mR_InvalidParameter); // TextureCrop parameter cannot be set twice.
      pParams->textureCrop = 1;
      break;

    case mSBE_T_Rotation:
      mERROR_IF(pParams->rotation != 0, mR_InvalidParameter); // Rotation parameter cannot be set twice.
      pParams->rotation = 1;
      break;

    case mSBE_T_MatrixTransform:
      mERROR_IF(pParams->matrixTransform != 0, mR_InvalidParameter); // MatrixTransform parameter cannot be set twice.
      pParams->matrixTransform = 1;
      break;

    case mSBE_T_TextureFlip:
      mERROR_IF(pParams->textureFlip != 0, mR_InvalidParameter); // TextureFlip parameter cannot be set twice.
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
inline mFUNCTION(mSpriteBatch_Create_Internal, IN_OUT mSpriteBatch<Args...> *pSpriteBatch)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSpriteBatch == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mQueue_Create(&pSpriteBatch->enqueuedRenderObjects, nullptr));

  pSpriteBatch->isStarted = false;
  pSpriteBatch->alphaMode = mSpriteBatch_AlphaMode::mSB_AM_AlphaBlend;
  pSpriteBatch->spriteSortMode = mSpriteBatch_SpriteSortMode::mSB_SSM_BackToFront;
  pSpriteBatch->textureSampleMode = mSpriteBatch_TextureSampleMode::mSB_TSM_LinearFiltering;

  pSpriteBatch->shaderParams = { 0 };
  mERROR_CHECK(mSpriteBatch_GenerateShader<Args...>::UnpackParams(&pSpriteBatch->shaderParams));

  // Vertex Shader.
  char vertexShader[1024] = "";

  mERROR_IF(0 > sprintf_s(vertexShader, "#version 150 core\n\nin vec2 position0;\nin vec2 texCoord0;\nout vec2 _texCoord0;\n\nuniform vec2 screenSize0;\nuniform vec3 startOffset0;\nuniform vec2 scale0;\n"), mR_InternalError);
  
  if (pSpriteBatch->shaderParams.rotation)
    mERROR_IF(0 > sprintf_s(vertexShader, "%s\nuniform float " mSBERotation_UniformName ";", vertexShader), mR_InternalError);
  
  if (pSpriteBatch->shaderParams.matrixTransform)
    mERROR_IF(0 > sprintf_s(vertexShader, "%s\nuniform mat4 " mSBEMatrixTransform_UniformName ";", vertexShader), mR_InternalError);
  
  mERROR_IF(0 > sprintf_s(vertexShader, "%s\n\nvoid main()\n{\n\t_texCoord0 = texCoord0;\n\tvec4 position = vec4(position0, 0.0, 1.0);\n", vertexShader), mR_InternalError);
  
  mERROR_IF(0 > sprintf_s(vertexShader, "%s\n\tposition.xy *= scale0;", vertexShader), mR_InternalError);
  
  if (pSpriteBatch->shaderParams.rotation)
    mERROR_IF(0 > sprintf_s(vertexShader, "%s\n\tvec2 oldPos = position.xy;\n\tposition.x = cos(" mSBERotation_UniformName ") * oldPos.x - sin(" mSBERotation_UniformName ") * oldPos.y;\n\tposition.y = sin(" mSBERotation_UniformName ") * oldPos.x + cos(" mSBERotation_UniformName ") * oldPos.y;", vertexShader), mR_InternalError);

  mERROR_IF(0 > sprintf_s(vertexShader, "%s\n\tposition.xy = (position.xy + startOffset0.xy * 2 + scale0) / (screenSize0 * 2);", vertexShader), mR_InternalError);
  
  if (pSpriteBatch->shaderParams.matrixTransform)
    mERROR_IF(0 > sprintf_s(vertexShader, "%s\n\tposition *= " mSBEMatrixTransform_UniformName ";", vertexShader), mR_InternalError);
  
  mERROR_IF(0 > sprintf_s(vertexShader, "%s\n\tposition.y = 1 - position.y;\n\tposition.xy = position.xy * 2 - 1;\n\tposition.z = startOffset0.z;\n\n\tgl_Position = position;\n}\n", vertexShader), mR_InternalError);

  // Fragment Shader.
  char fragmentShaderGetPosition[1024] = "";
  char fragmentShader[1024] = "";
  char fragmentShaderMS[1024] = "";

  mERROR_IF(0 > sprintf_s(fragmentShader, "#version 150 core\n\nout vec4 fragColour0;\n\nin vec2 _texCoord0;\nuniform sampler2D texture0;\n"), mR_InternalError);
  mERROR_IF(0 > sprintf_s(fragmentShaderMS, "#version 150 core\n\nout vec4 fragColour0;\n\nin vec2 _texCoord0;\nuniform sampler2DMS texture0;\nuniform int " mSB_MultisampleCountUniformName ";\n"), mR_InternalError);
  
  if (pSpriteBatch->shaderParams.colour)
    mERROR_IF(0 > sprintf_s(fragmentShaderGetPosition, "%s\nuniform vec4 " mSBEColour_UniformName ";", fragmentShaderGetPosition), mR_InternalError);
  
  if (pSpriteBatch->shaderParams.textureFlip)
    mERROR_IF(0 > sprintf_s(fragmentShaderGetPosition, "%s\nuniform vec2 " mSBETextureFlip_UniformName ";", fragmentShaderGetPosition), mR_InternalError);
  
  if (pSpriteBatch->shaderParams.textureCrop)
    mERROR_IF(0 > sprintf_s(fragmentShaderGetPosition, "%s\nuniform vec4 " mSBETextureCrop_UniformName ";", fragmentShaderGetPosition), mR_InternalError);
  
  mERROR_IF(0 > sprintf_s(fragmentShaderGetPosition, "%s\n\nvoid main()\n{\n\tvec2 texturePosition = vec2(1) - _texCoord0;", fragmentShaderGetPosition), mR_InternalError);
  
  if (pSpriteBatch->shaderParams.textureFlip)
    mERROR_IF(0 > sprintf_s(fragmentShaderGetPosition, "%s\n\ttexturePosition = texturePosition * (1 - 2 * " mSBETextureFlip_UniformName ") + " mSBETextureFlip_UniformName ";", fragmentShaderGetPosition), mR_InternalError);
  
  if (pSpriteBatch->shaderParams.textureCrop)
    mERROR_IF(0 > sprintf_s(fragmentShaderGetPosition, "%s\n\ttexturePosition *= (" mSBETextureCrop_UniformName ".zw - " mSBETextureCrop_UniformName ".xy);\n\ttexturePosition += " mSBETextureCrop_UniformName ".xy;", fragmentShaderGetPosition), mR_InternalError);

  mERROR_IF(0 > sprintf_s(fragmentShader, "%s%s", fragmentShader, fragmentShaderGetPosition), mR_InternalError);
  mERROR_IF(0 > sprintf_s(fragmentShaderMS, "%s%s", fragmentShaderMS, fragmentShaderGetPosition), mR_InternalError);
  
  mERROR_IF(0 > sprintf_s(fragmentShader, "%s\n\tfragColour0 = texture(texture0, texturePosition);", fragmentShader), mR_InternalError);
  mERROR_IF(0 > sprintf_s(fragmentShaderMS, "%s\n\tvec2 textureSize0 = textureSize(texture0);\n\tfragColour0 = vec4(0);\n\t\n\tfor (int i = 0; i < " mSB_MultisampleCountUniformName "; i++)\n\t{\n\t\tfragColour0 += texelFetch(texture0, ivec2(texturePosition * textureSize0), i);\n\t}\n\t\n\tfragColour0 /= float(" mSB_MultisampleCountUniformName ");", fragmentShaderMS), mR_InternalError);
  
  if (pSpriteBatch->shaderParams.colour)
  {
    mERROR_IF(0 > sprintf_s(fragmentShader, "%s\n\tfragColour0 *= " mSBEColour_UniformName ";", fragmentShader), mR_InternalError);
    mERROR_IF(0 > sprintf_s(fragmentShaderMS, "%s\n\tfragColour0 *= " mSBEColour_UniformName ";", fragmentShaderMS), mR_InternalError);
  }

  mERROR_IF(0 > sprintf_s(fragmentShader, "%s\n}\n", fragmentShader), mR_InternalError);
  mERROR_IF(0 > sprintf_s(fragmentShaderMS, "%s\n}\n", fragmentShaderMS), mR_InternalError);

  mDEFER_CALL_ON_ERROR(&pSpriteBatch->shader, mSharedPointer_Destroy);
  mDEFER_CALL_ON_ERROR(&pSpriteBatch->multisampleShader, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Allocate(&pSpriteBatch->shader, nullptr, (std::function<void(mShader *)>)[](mShader *pData) {mShader_Destroy(pData);}, 1));
  mERROR_CHECK(mSharedPointer_Allocate(&pSpriteBatch->multisampleShader, nullptr, (std::function<void(mShader *)>)[](mShader *pData) {mShader_Destroy(pData); }, 1));
  mERROR_CHECK(mShader_Create(pSpriteBatch->shader.GetPointer(), vertexShader, fragmentShader, "fragColour0"));
  mERROR_CHECK(mShader_Create(pSpriteBatch->multisampleShader.GetPointer(), vertexShader, fragmentShaderMS, "fragColour0"));

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mSpriteBatch_Destroy_Internal, IN_OUT mSpriteBatch<Args...> *pSpriteBatch)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSpriteBatch == nullptr, mR_ArgumentNull);

  if (pSpriteBatch->vbo != 0)
  {
    glDeleteBuffers(1, &pSpriteBatch->vbo);
    pSpriteBatch->vbo = 0;
  }

  if (pSpriteBatch->vao != 0)
  {
    glDeleteVertexArrays(1, &pSpriteBatch->vao);
    pSpriteBatch->vao = 0;
  }

  size_t queueSize = 0;
  mERROR_CHECK(mQueue_GetCount(pSpriteBatch->enqueuedRenderObjects, &queueSize));

  for (size_t i = 0; i < queueSize; ++i)
  {
    mSpriteBatch_Internal_RenderObject<Args...> renderObject;
    mERROR_CHECK(mQueue_PopFront(pSpriteBatch->enqueuedRenderObjects, &renderObject));
    mERROR_CHECK(mSpriteBatch_Internal_RenderObject_Destroy(&renderObject));
  }

  mERROR_CHECK(mQueue_Destroy(&pSpriteBatch->enqueuedRenderObjects));
  mERROR_CHECK(mSharedPointer_Destroy(&pSpriteBatch->shader));
  mERROR_CHECK(mSharedPointer_Destroy(&pSpriteBatch->multisampleShader));

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mSpriteBatch_Internal_SetAlphaBlending, mPtr<mSpriteBatch<Args...>> &spriteBatch)
{
  mFUNCTION_SETUP();

  switch (spriteBatch->alphaMode)
  {
  case mSB_AM_NoAlpha:
    mERROR_CHECK(mRenderParams_SetBlendingEnabled(false));
    break;

  case mSB_AM_Additive:
    mERROR_CHECK(mRenderParams_SetBlendFunc(mRP_BF_Additive));
    break;

  case mSB_AM_AlphaBlend:
    mERROR_CHECK(mRenderParams_SetBlendFunc(mRP_BF_AlphaBlend));
    break;

  case mSB_AM_Premultiplied:
    mERROR_CHECK(mRenderParams_SetBlendFunc(mRP_BF_Premultiplied));
    break;

  default:
    mRETURN_RESULT(mR_OperationNotSupported);
    break;
  }

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mSpriteBatch_Internal_SetDrawOrder, mPtr<mSpriteBatch<Args...>> &spriteBatch)
{
  mFUNCTION_SETUP();

  switch (spriteBatch->spriteSortMode)
  {
  case mSB_SSM_None:
    glDisable(GL_DEPTH_TEST);
    glDepthFunc(GL_ALWAYS);
    break;

  case mSB_SSM_BackToFront:
    mERROR_CHECK(mRenderParams_ClearDepth());
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_GREATER);
    break;

  case mSB_SSM_FrontToBack:
    mERROR_CHECK(mRenderParams_ClearDepth());
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    break;

  default:
    mRETURN_RESULT(mR_OperationNotSupported);
    break;
  }

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mSpriteBatch_Internal_SetTextureFilterMode, mPtr<mSpriteBatch<Args...>> &spriteBatch)
{
  mFUNCTION_SETUP();

  switch (spriteBatch->textureSampleMode)
  {
  case mSB_TSM_NearestNeighbor:
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    break;

  case mSB_TSM_LinearFiltering:
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    break;

  default:
    mRETURN_RESULT(mR_OperationNotSupported);
    break;
  }

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mSpriteBatch_Internal_InitializeMesh, mPtr<mSpriteBatch<Args...>> &spriteBatch)
{
  mFUNCTION_SETUP();

  mVec2f buffer[8] = { {-1, -1}, {1, 1}, {-1, 1}, {1, 0}, {1, -1}, {0, 1}, {1, 1}, {0, 0} };

  glGenVertexArrays(1, &spriteBatch->vao);
  glBindVertexArray(spriteBatch->vao);

  glGenBuffers(1, &spriteBatch->vbo);
  glBindBuffer(GL_ARRAY_BUFFER, spriteBatch->vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(buffer), buffer, GL_STATIC_DRAW);

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}

template<typename ...Args>
inline mFUNCTION(mSpriteBatch_Internal_BindMesh, mPtr<mSpriteBatch<Args...>> &spriteBatch, mPtr<mShader> &shader)
{
  mFUNCTION_SETUP();
  mUnused(spriteBatch);

  mERROR_CHECK(mShader_Bind(shader));

  glBindVertexArray(spriteBatch->vao);
  glBindBuffer(GL_ARRAY_BUFFER, spriteBatch->vbo);

  const mShaderAttributeIndex_t vertexPositiondAttributeIndex = mShader_GetAttributeIndex(shader, "position0");
  mGL_DEBUG_ERROR_CHECK();
  const mShaderAttributeIndex_t textureCoordAttributeIndex = mShader_GetAttributeIndex(shader, "texCoord0");
  mGL_DEBUG_ERROR_CHECK();

  glEnableVertexAttribArray(vertexPositiondAttributeIndex);
  glVertexAttribPointer(vertexPositiondAttributeIndex, (GLint)2, GL_FLOAT, GL_FALSE, (GLsizei)sizeof(mVec2f) * 2, (const void *)0);

  glEnableVertexAttribArray(textureCoordAttributeIndex);
  glVertexAttribPointer(textureCoordAttributeIndex, (GLint)2, GL_FLOAT, GL_FALSE, (GLsizei)sizeof(mVec2f) * 2, (const void *)sizeof(mVec2f));

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

template<typename ...Args>
inline mFUNCTION(mSpriteBatch_Internal_RenderObject_Create, OUT mSpriteBatch_Internal_RenderObject<Args...> *pRenderObject, const mPtr<mTexture> &texture, const mVec2f position, const mVec2f size, const float_t depth, Args &&...args)
{
  mFUNCTION_SETUP();

  mERROR_IF(pRenderObject == nullptr, mR_ArgumentNull);

  pRenderObject->texture = texture;
  pRenderObject->position = mVec3f(position, depth);
  pRenderObject->size = size;
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
struct mSpriteBatch_Internal_RenderObject_Render_Unpacker;

template<>
struct mSpriteBatch_Internal_RenderObject_Render_Unpacker<>
{
  static mFUNCTION(Unpack, mShader &)
  {
    mFUNCTION_SETUP();
    mRETURN_SUCCESS();
  }
};

template<typename T>
struct mSpriteBatch_Internal_RenderObject_Render_Unpacker<T>
{
  static mFUNCTION(Unpack, mShader &shader, T t)
  {
    mFUNCTION_SETUP();

    switch (T::type)
    {
    case mSBE_T_Colour:
      mERROR_CHECK(mShader_SetUniform(shader, mSBEColour_UniformName, t));
      break;

    case mSBE_T_TextureCrop:
      mERROR_CHECK(mShader_SetUniform(shader, mSBETextureCrop_UniformName, t));
      break;

    case mSBE_T_Rotation:
      mERROR_CHECK(mShader_SetUniform(shader, mSBERotation_UniformName, t));
      break;

    case mSBE_T_MatrixTransform:
      mERROR_CHECK(mShader_SetUniform(shader, mSBEMatrixTransform_UniformName, t));
      break;

    case mSBE_T_TextureFlip:
      mERROR_CHECK(mShader_SetUniform(shader, mSBETextureFlip_UniformName, t));
      break;

    default:
      mRETURN_RESULT(mR_OperationNotSupported);
      break;
    }

    mRETURN_SUCCESS();
  }
};

template<typename T, typename ...Args>
struct mSpriteBatch_Internal_RenderObject_Render_Unpacker<T, Args...>
{
  static mFUNCTION(Unpack, mShader &shader, T t, Args... args)
  {
    mFUNCTION_SETUP();

    mERROR_CHECK(mSpriteBatch_Internal_RenderObject_Render_Unpacker<T>::Unpack(shader, t));
    mERROR_CHECK(mSpriteBatch_Internal_RenderObject_Render_Unpacker<Args...>::Unpack(shader, std::forward<Args>(args)...));

    mRETURN_SUCCESS();
  }
};

template<typename ...Args>
inline mFUNCTION(mSpriteBatch_Internal_RenderObject_Render, mSpriteBatch_Internal_RenderObject<Args...> &renderObject, mPtr<mSpriteBatch<Args...>> &spriteBatch)
{
  mFUNCTION_SETUP();

  const bool isMultisampleTexture = renderObject.texture->sampleCount > 0;
  mPtr<mShader> &shader = isMultisampleTexture ? spriteBatch->multisampleShader : spriteBatch->shader;

  if (spriteBatch->lastWasMultisampleTexture != isMultisampleTexture)
  {
    mERROR_CHECK(mSpriteBatch_Internal_BindMesh(spriteBatch, shader));
    spriteBatch->lastWasMultisampleTexture = isMultisampleTexture;
  }

  mERROR_CHECK(mSpriteBatch_Internal_RenderObject_Render_Raw(renderObject, spriteBatch, *renderObject.texture, shader));

  mRETURN_SUCCESS();
}

template <typename ...Args>
mFUNCTION(mSpriteBatch_Internal_RenderObject_Render_Raw, mSpriteBatch_Internal_RenderObject<Args...> &renderObject, mPtr<mSpriteBatch<Args...>> &spriteBatch, mTexture &texture, mPtr<mShader> &shader)
{
  mFUNCTION_SETUP();

  mUnused(spriteBatch);

  mERROR_CHECK(mTexture_Bind(texture));
  mERROR_CHECK(mShader_SetUniform(shader, "texture0", texture));
  mERROR_CHECK(mShader_SetUniform(shader, "screenSize0", mRenderParams_CurrentRenderResolutionF));
  mERROR_CHECK(mShader_SetUniform(shader, "scale0", renderObject.size));
  mERROR_CHECK(mShader_SetUniform(shader, "startOffset0", renderObject.position));

  // Set uniforms.
  mERROR_CHECK(mForwardTuple(mSpriteBatch_Internal_RenderObject_Render_Unpacker<Args...>::Unpack, *shader, renderObject.args));

  if (texture.sampleCount > 0)
    mERROR_CHECK(mShader_SetUniform(shader, mSB_MultisampleCountUniformName, (int32_t)texture.sampleCount));

  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

  mRETURN_SUCCESS();
}
