#include "mFontRenderer.h"

#include "mQueue.h"
#include "mShader.h"
#include "mIndexedRenderDataBuffer.h"
#include "mBinaryChunk.h"

#pragma warning(push, 0)
#include "freetype-gl/include/freetype-gl.h"
#include "freetype-gl/include/texture-atlas.h"
#include "freetype-gl/include/texture-font.h"
#pragma warning(pop)

const char *mFontRenderer_VertexShader = R"__SHADER__(#version 150 core
/* Freetype GL - A C OpenGL Freetype engine
 *
 * Distributed under the OSI-approved BSD 2-Clause License.  See accompanying
 * file `LICENSE` for more details.
 */

uniform mat4 matrix;

attribute vec3 vertex;
attribute vec2 tex_coord;
attribute vec4 color;

out vec2 _texCoord;
out vec4 _colour;

void main()
{
    _texCoord = tex_coord.xy;
    _colour = color;

    gl_Position = matrix * vec4(vertex, 1.0);
}
)__SHADER__";

const char *mFontRenderer_FragmentShader = R"__SHADER__(#version 150 core
/* Freetype GL - A C OpenGL Freetype engine
 *
 * Distributed under the OSI-approved BSD 2-Clause License.  See accompanying
 * file `LICENSE` for more details.
 */

uniform sampler2D texture;

out vec4 colour;

in vec2 _texCoord;
in vec4 _colour;

void main()
{
    float a = texture2D(texture, _texCoord).r;
    colour = vec4(_colour.rgb, _colour.a * a);
}
)__SHADER__";

extern const char mFontRenderer_PositionAttribute[] = "vertex";
extern const char mFontRenderer_TextureCoordAttribute[] = "tex_coord";
extern const char mFontRenderer_ColourAttribute[] = "color";

struct mFontDescription_Internal
{
  enum
  {
    HashMapSize = 64, // has to be power of two.
    FontSizeFactor = 2
  };

  mInplaceString<255> fontFileName;
  uint64_t fontSize;
  texture_font_t *pTextureFont = nullptr;
  uint64_t glyphRange0;
  uint64_t glyphRange1;
  mPtr<mQueue<mPtr<mQueue<mchar_t>>>> additionalGlyphRanges;

  bool operator==(const mFontDescription &description)
  {
    return fontSize == (uint64_t)roundf(description.fontSize * FontSizeFactor) && description.fontFileName == fontFileName;
  }
};

mFUNCTION(mFontRenderer_Destroy_Internal, mFontRenderer *pFontRenderer);
mFUNCTION(mFontRenderer_UpdateFontAtlas_Internal, mPtr<mFontRenderer> &fontRenderer);
mFUNCTION(mFontRenderer_DrawEnqueuedText_Internal, mPtr<mFontRenderer> &fontRenderer);
mFUNCTION(mFontRenderer_AddFont_Internal, mPtr<mFontRenderer> &fontRenderer, const mFontDescription &fontDescription, OUT size_t *pIndex);
mFUNCTION(mFontRenderer_LoadGlyph_Internal, mPtr<mFontRenderer> &fontRenderer, mFontDescription_Internal *pFontDescription, const mchar_t codePoint, const char *character, const size_t bytes, OUT bool *pUnloadedChar);
mFUNCTION(mFontRenderer_LoadGlyphs_Internal, mPtr<mFontRenderer> &fontRenderer, mFontDescription_Internal *pFontDescription, const mString &string, OUT bool *pUnloadedChars);
mFUNCTION(mFontRenderer_ClearGlyphs_Internal, mPtr<mFontRenderer> &fontRenderer);

struct mFontTextureAtlas
{
  mPtr<mQueue<mFontDescription_Internal>> fonts;
  texture_atlas_t *pTextureAtlas;
  mTexture texture;
  size_t lastSize;
};

mFUNCTION(mFontTextureAtlas_Create, OUT mPtr<mFontTextureAtlas> *pAtlas, mAllocator *pAllocator, const size_t width, const size_t height);
mFUNCTION(mFontTextureAtlas_Destroy, IN mFontTextureAtlas *pAtlas);
mFUNCTION(mFontTextureAtlas_RequiresUpload, mPtr<mFontTextureAtlas> &atlas, OUT bool *pNeedsUpload);

mFUNCTION(mFontRenderable_Destroy_Internal, IN mFontRenderable *pFontRenderable);

struct mFontRenderable
{
  mPtr<mFontTextureAtlas> textureAtlas;
  mIndexedRenderDataBuffer<mRDB_FloatAttribute<3, mFontRenderer_PositionAttribute>, mRDB_FloatAttribute<2, mFontRenderer_TextureCoordAttribute>, mRDB_FloatAttribute<4, mFontRenderer_ColourAttribute>> indexDataBuffer;
};

//////////////////////////////////////////////////////////////////////////

struct mFontRenderer
{
  mPtr<mFontTextureAtlas> textureAtlas;
  mPtr<mBinaryChunk> enqueuedText;
  mIndexedRenderDataBuffer<mRDB_FloatAttribute<3, mFontRenderer_PositionAttribute>, mRDB_FloatAttribute<2, mFontRenderer_TextureCoordAttribute>, mRDB_FloatAttribute<4, mFontRenderer_ColourAttribute>> indexDataBuffer;
  mPtr<mQueue<mPtr<mFontTextureAtlas>>> currentFrameTextureAtlasses;
  mPtr<mQueue<mPtr<mFontTextureAtlas>>> availableTextureAtlasses;
  mMatrix matrix;
  mAllocator *pAllocator;
  bool started;
  bool startedRenderable;
  mVec2f position, resetPosition;
  uint32_t *pIndexBuffer;
  size_t indexBufferCapacity;
  bool addedGlyph;
};

mFUNCTION(mFontRenderer_Create, OUT mPtr<mFontRenderer> *pFontRenderer, IN mAllocator *pAllocator, const size_t width /* = 2048 */, const size_t height /* = 2048 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pFontRenderer == nullptr || pAllocator == nullptr, mR_ArgumentNull);

  mDEFER_CALL_ON_ERROR(pFontRenderer, mFontRenderer_Destroy);
  mERROR_CHECK(mSharedPointer_Allocate(pFontRenderer, pAllocator, (std::function<void(mFontRenderer *)>)[](mFontRenderer *pData) {mFontRenderer_Destroy_Internal(pData);}, 1));

  (*pFontRenderer)->pAllocator = pAllocator;
  (*pFontRenderer)->started = false;
  (*pFontRenderer)->startedRenderable = false;
  (*pFontRenderer)->addedGlyph = false;

  mERROR_CHECK(mBinaryChunk_Create(&(*pFontRenderer)->enqueuedText, pAllocator));

  mERROR_CHECK(mFontTextureAtlas_Create(&(*pFontRenderer)->textureAtlas, pAllocator, width, height));

  mERROR_CHECK(mIndexedRenderDataBuffer_Create(&(*pFontRenderer)->indexDataBuffer, pAllocator, mFontRenderer_VertexShader, mFontRenderer_FragmentShader, true, false));
  mERROR_CHECK(mQueue_Create(&(*pFontRenderer)->currentFrameTextureAtlasses, pAllocator));
  mERROR_CHECK(mQueue_Create(&(*pFontRenderer)->availableTextureAtlasses, pAllocator));

  mRETURN_SUCCESS();
}

mFUNCTION(mFontRenderer_Destroy, IN_OUT mPtr<mFontRenderer> *pFontRenderer)
{
  mFUNCTION_SETUP();

  mERROR_IF(pFontRenderer == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Destroy(pFontRenderer));

  mRETURN_SUCCESS();
}

mFUNCTION(mFontRenderer_AddFont, mPtr<mFontRenderer> &fontRenderer, const mFontDescription &fontDescription)
{
  mFUNCTION_SETUP();

  size_t unused;
  mERROR_CHECK(mFontRenderer_AddFont_Internal(fontRenderer, fontDescription, &unused));

  mRETURN_SUCCESS();
}

mFUNCTION(mFontRenderer_SetPosition, mPtr<mFontRenderer> &fontRenderer, const mVec2f position)
{
  mFUNCTION_SETUP();

  mERROR_IF(fontRenderer == nullptr, mR_ArgumentNull);

  fontRenderer->resetPosition = fontRenderer->position = position;

  mRETURN_SUCCESS();
}

mFUNCTION(mFontRenderer_SetDisplayPosition, mPtr<mFontRenderer> &fontRenderer, const mVec2f position)
{
  mFUNCTION_SETUP();

  mERROR_IF(fontRenderer == nullptr, mR_ArgumentNull);

  mVec2f _position = position;
  _position.y = mRenderParams_CurrentRenderResolutionF.y - position.y;

  fontRenderer->resetPosition = fontRenderer->position = _position;

  mRETURN_SUCCESS();
}

mFUNCTION(mFontRenderer_GetCurrentPosition, mPtr<mFontRenderer> &fontRenderer, OUT mVec2f *pPosition)
{
  mFUNCTION_SETUP();

  mERROR_IF(fontRenderer == nullptr || pPosition == nullptr, mR_ArgumentNull);

  *pPosition = fontRenderer->position;

  mRETURN_SUCCESS();
}

mFUNCTION(mFontRenderer_GetCurrentDisplayPosition, mPtr<mFontRenderer> &fontRenderer, OUT mVec2f *pPosition)
{
  mFUNCTION_SETUP();

  mERROR_IF(fontRenderer == nullptr || pPosition == nullptr, mR_ArgumentNull);

  mVec2f _position = fontRenderer->position;
  _position.y = mRenderParams_CurrentRenderResolutionF.y - _position.y;

  *pPosition = _position;

  mRETURN_SUCCESS();
}

mFUNCTION(mFontRenderer_Begin, mPtr<mFontRenderer> &fontRenderer, const mMatrix &matrix /* = mMatrix::Scale(2.0f / mRenderParams_CurrentRenderResolutionF.x, 2.0f / mRenderParams_CurrentRenderResolutionF.y, 1) * mMatrix::Translation(-1, -1, 0) */)
{
  mFUNCTION_SETUP();

  mERROR_IF(fontRenderer == nullptr, mR_ArgumentNull);
  mERROR_IF(fontRenderer->started, mR_ResourceStateInvalid);

  fontRenderer->started = true;
  fontRenderer->matrix = matrix;
  fontRenderer->position = fontRenderer->resetPosition;

  mRETURN_SUCCESS();
}

mFUNCTION(mFontRenderer_BeginRenderable, mPtr<mFontRenderer> &fontRenderer)
{
  mFUNCTION_SETUP();

  mERROR_IF(fontRenderer == nullptr, mR_ArgumentNull);
  mERROR_IF(fontRenderer->startedRenderable, mR_ResourceStateInvalid);

  mERROR_CHECK(mFontRenderer_Begin(fontRenderer));

  fontRenderer->startedRenderable = true;

  mRETURN_SUCCESS();
}

mFUNCTION(mFontRenderer_End, mPtr<mFontRenderer> &fontRenderer)
{
  mFUNCTION_SETUP();

  mERROR_IF(fontRenderer == nullptr, mR_ArgumentNull);
  mERROR_IF(!fontRenderer->started, mR_ResourceStateInvalid);
  mERROR_IF(fontRenderer->startedRenderable, mR_ResourceStateInvalid);

  mERROR_CHECK(mFontRenderer_DrawEnqueuedText_Internal(fontRenderer));
  
  size_t textureAtlasCount = 0;
  mERROR_CHECK(mQueue_GetCount(fontRenderer->currentFrameTextureAtlasses, &textureAtlasCount));

  // Move all current frame texture atlases into available texture atlases.
  for (size_t i = 0; i < textureAtlasCount; i++)
  {
    mPtr<mFontTextureAtlas> atlas;
    mDEFER_CALL(&atlas, mSharedPointer_Destroy);
    mERROR_CHECK(mQueue_PopFront(fontRenderer->currentFrameTextureAtlasses, &atlas));
    mERROR_CHECK(mQueue_PushBack(fontRenderer->availableTextureAtlasses, &atlas));
  }

  fontRenderer->started = false;

  mRETURN_SUCCESS();
}

mFUNCTION(mFontRenderer_EndRenderable, mPtr<mFontRenderer> &fontRenderer, OUT mPtr<mFontRenderable> *pRenderable, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(fontRenderer == nullptr || pRenderable == nullptr, mR_ArgumentNull);
  mERROR_IF(!fontRenderer->started, mR_ResourceStateInvalid);
  mERROR_IF(!fontRenderer->startedRenderable, mR_ResourceStateInvalid);

  fontRenderer->started = false;
  fontRenderer->startedRenderable = false;

  mERROR_CHECK(mSharedPointer_Allocate(pRenderable, pAllocator, (std::function<void(mFontRenderable *)>)[](mFontRenderable *pData) {mFontRenderable_Destroy_Internal(pData);}, 1));

  (*pRenderable)->textureAtlas = fontRenderer->textureAtlas;

  mERROR_CHECK(mIndexedRenderDataBuffer_Create(&(*pRenderable)->indexDataBuffer, fontRenderer->indexDataBuffer.shader));

#if defined(mRENDERER_OPENGL)
  size_t count = 0;
  mERROR_CHECK(mBinaryChunk_GetWriteBytes(fontRenderer->enqueuedText, &count));

  if (count == 0)
    mRETURN_SUCCESS();

  uint8_t *pFirstIndex = fontRenderer->enqueuedText->pData;

  mERROR_CHECK(mIndexedRenderDataBuffer_SetVertexBuffer((*pRenderable)->indexDataBuffer, pFirstIndex, count));

  // Get the number of indices required to render the vertices as rectangles.
  // `count` is the size of the data of all vertices. 
  // one vertex is: {mVec3f position, mVec2f textureCoordinate, mVec4f colour}.
  // 4 vertices make a rectangle.
  // 6 indices are required to draw a quad.
  const size_t length = count / ((sizeof(mVec3f) + sizeof(mVec2f) + sizeof(mVec4f)) * 4) * 6;

  if (fontRenderer->indexBufferCapacity < length)
  {
    const size_t oldCapacity = fontRenderer->indexBufferCapacity;
    const size_t newLength = mMax(length, fontRenderer->indexBufferCapacity * 2 + 1 * 6);

    mERROR_CHECK(mAllocator_Reallocate(fontRenderer->pAllocator, &fontRenderer->pIndexBuffer, newLength));
    fontRenderer->indexBufferCapacity = newLength;

    for (size_t i6 = oldCapacity, i4 = oldCapacity / 6 * 4; i6 < newLength; i6 += 6, i4 += 4)
    {
      // 0, 1, 2, 0, 2, 3
      fontRenderer->pIndexBuffer[i6 + 0] = (uint32_t)i4 + 0;
      fontRenderer->pIndexBuffer[i6 + 1] = (uint32_t)i4 + 1;
      fontRenderer->pIndexBuffer[i6 + 2] = (uint32_t)i4 + 2;
      fontRenderer->pIndexBuffer[i6 + 3] = (uint32_t)i4 + 0;
      fontRenderer->pIndexBuffer[i6 + 4] = (uint32_t)i4 + 2;
      fontRenderer->pIndexBuffer[i6 + 5] = (uint32_t)i4 + 3;
    }

    mERROR_CHECK(mIndexedRenderDataBuffer_SetIndexBuffer(fontRenderer->indexDataBuffer, fontRenderer->pIndexBuffer, length)); // yes this is required. otherwise owned indexDataBuffer of the fontRenderer might get invalidated.
  }

  mERROR_CHECK(mIndexedRenderDataBuffer_SetIndexBuffer((*pRenderable)->indexDataBuffer, fontRenderer->pIndexBuffer, length));

  mERROR_CHECK(mFontRenderer_UpdateFontAtlas_Internal(fontRenderer));
  mERROR_CHECK(mIndexedRenderDataBuffer_SetRenderCount((*pRenderable)->indexDataBuffer, length));
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mERROR_CHECK(mBinaryChunk_ResetWrite(fontRenderer->enqueuedText));

  mRETURN_SUCCESS();
}

mFUNCTION(mFontRenderer_Draw, mPtr<mFontRenderer> &fontRenderer, const mFontDescription &fontDescription, const mString &string, const mVector colour)
{
  mFUNCTION_SETUP();

  mERROR_IF(fontRenderer == nullptr, mR_ArgumentNull);
  mERROR_IF(!fontRenderer->started, mR_ResourceStateInvalid);

  size_t fontIndex;
  mERROR_CHECK(mFontRenderer_AddFont_Internal(fontRenderer, fontDescription, &fontIndex));

  mFontDescription_Internal *pFontDescription = nullptr;
  mERROR_CHECK(mQueue_PointerAt(fontRenderer->textureAtlas->fonts, fontIndex, &pFontDescription));

  size_t index = 0;
  char *previousChar = nullptr;

  for (auto &&_char : string)
  {
    if (*_char.character == '\n')
    {
      previousChar = _char.character;
      fontRenderer->position.x = fontRenderer->resetPosition.x;
      fontRenderer->position.y -= fontDescription.lineHeightRatio * fontDescription.fontSize;
    }
    else
    {
      bool failedToLoadChars = false;
      bool unloadedChars = false;
      mERROR_CHECK(mFontRenderer_LoadGlyph_Internal(fontRenderer, pFontDescription, _char.codePoint, _char.character, _char.characterSize, &unloadedChars));

      if (unloadedChars)
      {
        dealWithUnloadedChars:
        failedToLoadChars = true;

        mERROR_IF(fontRenderer->startedRenderable, mR_ArgumentOutOfBounds);
        mERROR_CHECK(mFontRenderer_DrawEnqueuedText_Internal(fontRenderer));

        // Get new texture atlas.
        {
          // Enqueue current texture atlas in the currentFrameTextureAtlasses if it's not used by others.
          if (fontRenderer->textureAtlas.GetReferenceCount() == 1)
            mERROR_CHECK(mQueue_PushBack(fontRenderer->currentFrameTextureAtlasses, fontRenderer->textureAtlas));

          size_t textureAtlasCount = 0;
          mERROR_CHECK(mQueue_GetCount(fontRenderer->availableTextureAtlasses, &textureAtlasCount));

          if (textureAtlasCount > 0)
          {
            // Get available texture atlas if existent, clear it, make it the current one, continue.
            mERROR_CHECK(mQueue_PopFront(fontRenderer->availableTextureAtlasses, &fontRenderer->textureAtlas));
            
            texture_atlas_clear(fontRenderer->textureAtlas->pTextureAtlas);
            fontRenderer->textureAtlas->lastSize = 0;
            mERROR_CHECK(mFontRenderer_ClearGlyphs_Internal(fontRenderer));
          }
          else
          {
            // If no texture atlas available: create a new one.
            mERROR_CHECK(mFontTextureAtlas_Create(&fontRenderer->textureAtlas, fontRenderer->pAllocator, fontRenderer->textureAtlas->pTextureAtlas->width, fontRenderer->textureAtlas->pTextureAtlas->height));
          }

          mERROR_CHECK(mFontRenderer_AddFont_Internal(fontRenderer, fontDescription, &fontIndex));
          mERROR_CHECK(mQueue_PointerAt(fontRenderer->textureAtlas->fonts, fontIndex, &pFontDescription));
        }

        mERROR_CHECK(mFontRenderer_LoadGlyph_Internal(fontRenderer, pFontDescription, _char.codePoint, _char.character, _char.characterSize, &unloadedChars));

        mERROR_IF(unloadedChars, mR_ArgumentOutOfBounds);
      }

      texture_glyph_t *pGlyph = texture_font_find_glyph(pFontDescription->pTextureFont, _char.character);

      if (pGlyph == nullptr)
      {
        if (!failedToLoadChars && !fontRenderer->startedRenderable)
          goto dealWithUnloadedChars;

        pGlyph = texture_font_find_glyph(pFontDescription->pTextureFont, nullptr);

        if (pGlyph == nullptr)
          continue;
      }

      float kerning = 0.0f;

      if (index > 0)
        kerning = texture_glyph_get_kerning(pGlyph, previousChar);

      previousChar = _char.character;

      index++;

      fontRenderer->position.x += kerning;

      int64_t x0 = (int64_t)(fontRenderer->position.x + pGlyph->offset_x);
      int64_t y0 = (int64_t)(fontRenderer->position.y + pGlyph->offset_y);
      int64_t x1 = (int64_t)(x0 + pGlyph->width);
      int64_t y1 = (int64_t)(y0 - pGlyph->height);

      float_t s0 = pGlyph->s0;
      float_t t0 = pGlyph->t0;
      float_t s1 = pGlyph->s1;
      float_t t1 = pGlyph->t1;

      mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, mVec3f((float_t)x0, (float_t)y0, 0)));
      mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, mVec2f(s0, t0)));
      mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, (mVec4f)colour));

      mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, mVec3f((float_t)x0, (float_t)y1, 0)));
      mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, mVec2f(s0, t1)));
      mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, (mVec4f)colour));

      mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, mVec3f((float_t)x1, (float_t)y1, 0)));
      mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, mVec2f(s1, t1)));
      mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, (mVec4f)colour));

      mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, mVec3f((float_t)x1, (float_t)y0, 0)));
      mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, mVec2f(s1, t0)));
      mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, (mVec4f)colour));

      fontRenderer->position.x += pGlyph->advance_x * fontDescription.glyphSpacingRatio;
    }
  }

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mFontRenderer_Destroy_Internal, mFontRenderer *pFontRenderer)
{
  mFUNCTION_SETUP();

  mERROR_IF(pFontRenderer == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Destroy(&pFontRenderer->textureAtlas));

  mERROR_CHECK(mBinaryChunk_Destroy(&pFontRenderer->enqueuedText));

  mERROR_CHECK(mIndexedRenderDataBuffer_Destroy(&pFontRenderer->indexDataBuffer));
  mERROR_CHECK(mAllocator_FreePtr(pFontRenderer->pAllocator, &pFontRenderer->pIndexBuffer));
  mERROR_CHECK(mQueue_Destroy(&pFontRenderer->currentFrameTextureAtlasses));
  mERROR_CHECK(mQueue_Destroy(&pFontRenderer->availableTextureAtlasses));

  mRETURN_SUCCESS();
}

mFUNCTION(mFontRenderer_UpdateFontAtlas_Internal, mPtr<mFontRenderer> &fontRenderer)
{
  mFUNCTION_SETUP();

  bool requiresUpload = true;
  mERROR_CHECK(mFontTextureAtlas_RequiresUpload(fontRenderer->textureAtlas, &requiresUpload));

#if defined(mRENDERER_OPENGL)

  glBindTexture(GL_TEXTURE_2D, fontRenderer->textureAtlas->pTextureAtlas->id);

  if (requiresUpload || fontRenderer->addedGlyph)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, (GLsizei)fontRenderer->textureAtlas->pTextureAtlas->width, (GLsizei)fontRenderer->textureAtlas->pTextureAtlas->height, 0, GL_RED, GL_UNSIGNED_BYTE, fontRenderer->textureAtlas->pTextureAtlas->data);

  fontRenderer->addedGlyph = false;
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mFontRenderer_DrawEnqueuedText_Internal, mPtr<mFontRenderer> &fontRenderer)
{
  mFUNCTION_SETUP();

#if defined(mRENDERER_OPENGL)

  size_t count = 0;
  mERROR_CHECK(mBinaryChunk_GetWriteBytes(fontRenderer->enqueuedText, &count));

  if (count == 0)
    mRETURN_SUCCESS();

  uint8_t *pFirstIndex = fontRenderer->enqueuedText->pData;

  mERROR_CHECK(mIndexedRenderDataBuffer_SetVertexBuffer(fontRenderer->indexDataBuffer, pFirstIndex, count));

  // Get the number of indices required to render the vertices as rectangles.
  // `count` is the size of the data of all vertices. 
  // one vertex is: {mVec3f position, mVec2f textureCoordinate, mVec4f colour}.
  // 4 vertices make a rectangle.
  // 6 indices are required to draw a quad.
  const size_t length = count / ((sizeof(mVec3f) + sizeof(mVec2f) + sizeof(mVec4f)) * 4) * 6;

  if (fontRenderer->indexBufferCapacity < length)
  {
    const size_t oldCapacity = fontRenderer->indexBufferCapacity;
    const size_t newLength = mMax(length, fontRenderer->indexBufferCapacity * 2 + 1 * 6);

    mERROR_CHECK(mAllocator_Reallocate(fontRenderer->pAllocator, &fontRenderer->pIndexBuffer, newLength));
    fontRenderer->indexBufferCapacity = newLength;
    
    for (size_t i6 = oldCapacity, i4 = oldCapacity / 6 * 4; i6 < newLength; i6 += 6, i4 += 4)
    {
      // 0, 1, 2, 0, 2, 3
      fontRenderer->pIndexBuffer[i6 + 0] = (uint32_t)i4 + 0;
      fontRenderer->pIndexBuffer[i6 + 1] = (uint32_t)i4 + 1;
      fontRenderer->pIndexBuffer[i6 + 2] = (uint32_t)i4 + 2;
      fontRenderer->pIndexBuffer[i6 + 3] = (uint32_t)i4 + 0;
      fontRenderer->pIndexBuffer[i6 + 4] = (uint32_t)i4 + 2;
      fontRenderer->pIndexBuffer[i6 + 5] = (uint32_t)i4 + 3;
    }

    mERROR_CHECK(mIndexedRenderDataBuffer_SetIndexBuffer(fontRenderer->indexDataBuffer, fontRenderer->pIndexBuffer, newLength));
  }

  mERROR_CHECK(mFontRenderer_UpdateFontAtlas_Internal(fontRenderer));

  mERROR_CHECK(mRenderParams_SetBlendingEnabled(true));
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  mERROR_CHECK(mTexture_Bind(fontRenderer->textureAtlas->texture));
  mERROR_CHECK(mShader_Bind(*fontRenderer->indexDataBuffer.shader));
  mERROR_CHECK(mShader_SetUniform(fontRenderer->indexDataBuffer.shader, "matrix", fontRenderer->matrix));
  mERROR_CHECK(mShader_SetUniform(fontRenderer->indexDataBuffer.shader, "texture", fontRenderer->textureAtlas->texture));
  mERROR_CHECK(mIndexedRenderDataBuffer_SetRenderCount(fontRenderer->indexDataBuffer, length));
  mERROR_CHECK(mIndexedRenderDataBuffer_Draw(fontRenderer->indexDataBuffer));

  mERROR_CHECK(mBinaryChunk_ResetWrite(fontRenderer->enqueuedText));

#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mFontRenderer_AddFont_Internal, mPtr<mFontRenderer> &fontRenderer, const mFontDescription &fontDescription, OUT size_t *pIndex)
{
  mFUNCTION_SETUP();

  mERROR_IF(fontRenderer == nullptr, mR_ArgumentNull);

  size_t fontCount = 0;
  mERROR_CHECK(mQueue_GetCount(fontRenderer->textureAtlas->fonts, &fontCount));

  for (size_t i = 0; i < fontCount; i++)
  {
    mFontDescription_Internal *pDesc = nullptr;
    mERROR_CHECK(mQueue_PointerAt(fontRenderer->textureAtlas->fonts, i, &pDesc));

    if (*pDesc == fontDescription)
    {
      *pIndex = i;
      mRETURN_SUCCESS();
    }
  }

  mFontDescription_Internal fontDesc;
  mZeroMemory(&fontDesc, 1);

  mERROR_CHECK(mInplaceString_Create(&fontDesc.fontFileName, fontDescription.fontFileName));
  fontDesc.fontSize = (uint64_t)roundf(fontDescription.fontSize * mFontDescription_Internal::FontSizeFactor);
  fontDesc.pTextureFont = texture_font_new_from_file(fontRenderer->textureAtlas->pTextureAtlas, fontDesc.fontSize / (float_t)mFontDescription_Internal::FontSizeFactor, fontDesc.fontFileName.c_str());

  mERROR_IF(fontDesc.pTextureFont == nullptr, mR_InternalError);
  mERROR_CHECK(mQueue_PushBack(fontRenderer->textureAtlas->fonts, &fontDesc));

  *pIndex = fontCount;

  mRETURN_SUCCESS();
}

mFUNCTION(mFontRenderer_LoadGlyph_Internal, mPtr<mFontRenderer> &fontRenderer, mFontDescription_Internal *pFontDescription, const mchar_t codePoint, const char *character, const size_t /* bytes */, OUT bool *pUnloadedChar)
{
  mFUNCTION_SETUP();

  *pUnloadedChar = false;

  if (codePoint >= 0 && codePoint < 64)
  {
    if (pFontDescription->glyphRange0 & (1ULL << codePoint))
      mRETURN_SUCCESS();
  }
  else if (codePoint >= 64 && codePoint < 128)
  {
    if (pFontDescription->glyphRange1 & (1ULL << (codePoint - 64)))
      mRETURN_SUCCESS();
  }
  else
  {
    if (pFontDescription->additionalGlyphRanges == nullptr)
    {
      mERROR_CHECK(mQueue_Create(&pFontDescription->additionalGlyphRanges, fontRenderer->pAllocator));
      mERROR_CHECK(mQueue_Reserve(pFontDescription->additionalGlyphRanges, mFontDescription_Internal::HashMapSize));

      for (size_t i = 0; i < mFontDescription_Internal::HashMapSize; i++)
      {
        mPtr<mQueue<mchar_t>> queue;
        mDEFER_CALL(&queue, mQueue_Destroy);
        mERROR_CHECK(mQueue_Create(&queue, fontRenderer->pAllocator));

        mERROR_CHECK(mQueue_PushBack(pFontDescription->additionalGlyphRanges, std::move(queue)));
      }
    }

    const size_t index = codePoint & (mFontDescription_Internal::HashMapSize - 1);

    mPtr<mQueue<mchar_t>> *pQueue = nullptr;
    mERROR_CHECK(mQueue_PointerAt(pFontDescription->additionalGlyphRanges, index, &pQueue));

    for (mchar_t c : **pQueue)
    {
      if (c == codePoint)
        mRETURN_SUCCESS();
    }
  }

  // Load glyph.
  const size_t glyphLoaded = texture_font_load_glyph(pFontDescription->pTextureFont, character);
  *pUnloadedChar = 0 == glyphLoaded;

  if (!*pUnloadedChar)
  {
    fontRenderer->addedGlyph = true;

    if (codePoint >= 0 && codePoint < 64)
    {
      pFontDescription->glyphRange0 |= (1ULL << (codePoint));
    }
    else if (codePoint >= 64 && codePoint < 128)
    {
      pFontDescription->glyphRange1 |= (1ULL << (codePoint - 64));
    }
    else
    {
      const size_t index = codePoint & (mFontDescription_Internal::HashMapSize - 1);

      mPtr<mQueue<mchar_t>> *pQueue = nullptr;
      mERROR_CHECK(mQueue_PointerAt(pFontDescription->additionalGlyphRanges, index, &pQueue));

      mERROR_CHECK(mQueue_PushBack(*pQueue, codePoint));
    }
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mFontRenderer_LoadGlyphs_Internal, mPtr<mFontRenderer> &fontRenderer, mFontDescription_Internal *pFontDescription, const mString &string, OUT bool *pUnloadedChars)
{
  mFUNCTION_SETUP();

  *pUnloadedChars = false;

  for (auto &&_char : string)
  {
    mFUNCTION_SETUP();

    bool unloadedChar = false;
    mERROR_CHECK(mFontRenderer_LoadGlyph_Internal(fontRenderer, pFontDescription, _char.codePoint, _char.character, _char.characterSize, &unloadedChar));

    if (unloadedChar)
    {
      *pUnloadedChars = true;
      mRETURN_SUCCESS();
    }
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mFontRenderer_ClearGlyphs_Internal, mPtr<mFontRenderer> &fontRenderer)
{
  mFUNCTION_SETUP();

  fontRenderer->addedGlyph = false;

  size_t fontCount = 0;
  mERROR_CHECK(mQueue_GetCount(fontRenderer->textureAtlas->fonts, &fontCount));

  for (size_t i = 0; i < fontCount; i++)
  {
    mFontDescription_Internal *pFontDesc = nullptr;
    mERROR_CHECK(mQueue_PointerAt(fontRenderer->textureAtlas->fonts, i, &pFontDesc));

    pFontDesc->glyphRange0 = 0;
    pFontDesc->glyphRange1 = 0;

    if (pFontDesc->additionalGlyphRanges != nullptr)
    {
      for (size_t j = 0; j < mFontDescription_Internal::HashMapSize; j++)
      {
        mPtr<mQueue<mchar_t>> *pCharQueue;
        mERROR_CHECK(mQueue_PointerAt(pFontDesc->additionalGlyphRanges, j, &pCharQueue));
        mERROR_CHECK(mQueue_Clear(*pCharQueue));
      }
    }

    vector_clear(pFontDesc->pTextureFont->glyphs);
  }

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mFontTextureAtlas_Create, OUT mPtr<mFontTextureAtlas> *pAtlas, mAllocator *pAllocator, const size_t width, const size_t height)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAtlas == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Allocate(pAtlas, pAllocator, (std::function<void(mFontTextureAtlas *)>)[](mFontTextureAtlas *pData) { mFontTextureAtlas_Destroy(pData); }, 1));

  mERROR_CHECK(mQueue_Create(&(*pAtlas)->fonts, pAllocator));
  
  (*pAtlas)->pTextureAtlas = texture_atlas_new(width, height, 1);

  mERROR_IF((*pAtlas)->pTextureAtlas == nullptr, mR_InternalError);

  mERROR_CHECK(mTexture_Allocate(&(*pAtlas)->texture, mVec2s(width, height), mPF_Monochrome8));
  (*pAtlas)->pTextureAtlas->id = (*pAtlas)->texture.textureId;

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}

mFUNCTION(mFontTextureAtlas_Destroy, IN mFontTextureAtlas *pAtlas)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAtlas == nullptr, mR_ArgumentNull);

  mGL_DEBUG_ERROR_CHECK();

  if (pAtlas->fonts != nullptr)
  {
    size_t fontCount = 0;
    mERROR_CHECK(mQueue_GetCount(pAtlas->fonts, &fontCount));

    for (size_t i = 0; i < fontCount; i++)
    {
      mFontDescription_Internal fontDesc;
      mERROR_CHECK(mQueue_PopFront(pAtlas->fonts, &fontDesc));

      if (fontDesc.pTextureFont != nullptr)
      {
        texture_font_delete(fontDesc.pTextureFont);
        fontDesc.pTextureFont = nullptr;
      }

      if (fontDesc.additionalGlyphRanges)
        mERROR_CHECK(mQueue_Destroy(&fontDesc.additionalGlyphRanges));
    }

    mERROR_CHECK(mQueue_Destroy(&pAtlas->fonts));
  }

  if (pAtlas->pTextureAtlas != nullptr)
  {
    texture_atlas_delete(pAtlas->pTextureAtlas);
    pAtlas->pTextureAtlas = nullptr;
  }

  mERROR_CHECK(mTexture_Destroy(&pAtlas->texture));
  
  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}

mFUNCTION(mFontTextureAtlas_RequiresUpload, mPtr<mFontTextureAtlas> &atlas, OUT bool *pNeedsUpload)
{
  mFUNCTION_SETUP();

  mERROR_IF(atlas == nullptr || pNeedsUpload == nullptr, mR_ArgumentNull);

  if (atlas->pTextureAtlas->nodes->size != atlas->lastSize)
  {
    atlas->lastSize = atlas->pTextureAtlas->nodes->size;
    *pNeedsUpload = true;
  }
  else
  {
    *pNeedsUpload = false;
  }

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mFontRenderable_Draw, mPtr<mFontRenderable> &fontRenderable, const mMatrix &matrix /* = mMatrix::Scale(2.0f / mRenderParams_CurrentRenderResolutionF.x, 2.0f / mRenderParams_CurrentRenderResolutionF.y, 1) * mMatrix::Translation(-1, -1, 0) */)
{
  mFUNCTION_SETUP();

  mERROR_IF(fontRenderable == nullptr, mR_ArgumentNull);

#if defined (mRENDERER_OPENGL)
  mERROR_CHECK(mRenderParams_SetBlendingEnabled(true));
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  mERROR_CHECK(mTexture_Bind(fontRenderable->textureAtlas->texture));
  mERROR_CHECK(mShader_Bind(*fontRenderable->indexDataBuffer.shader));
  mERROR_CHECK(mShader_SetUniform(fontRenderable->indexDataBuffer.shader, "matrix", matrix));
  mERROR_CHECK(mShader_SetUniform(fontRenderable->indexDataBuffer.shader, "texture", fontRenderable->textureAtlas->texture));
  mERROR_CHECK(mIndexedRenderDataBuffer_Draw(fontRenderable->indexDataBuffer));
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mFontRenderable_Destroy, IN_OUT mPtr<mFontRenderable> *pFontRenderable)
{
  mFUNCTION_SETUP();

  mERROR_IF(pFontRenderable == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Destroy(pFontRenderable));

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mFontRenderable_Destroy_Internal, IN mFontRenderable *pFontRenderable)
{
  mFUNCTION_SETUP();

  mERROR_IF(pFontRenderable == nullptr, mR_ArgumentNull);
  
  mGL_DEBUG_ERROR_CHECK();

  mERROR_CHECK(mSharedPointer_Destroy(&pFontRenderable->textureAtlas));
  mERROR_CHECK(mIndexedRenderDataBuffer_Destroy(&pFontRenderable->indexDataBuffer));

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}
