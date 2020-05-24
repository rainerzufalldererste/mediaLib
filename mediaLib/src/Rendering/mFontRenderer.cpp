#include "mFontRenderer.h"

#include "mQueue.h"
#include "mShader.h"
#include "mIndexedRenderDataBuffer.h"
#include "mBinaryChunk.h"

#include "mProfiler.h"

#pragma warning(push, 0)
#include "freetype-gl/include/freetype-gl.h"
#include "freetype-gl/include/texture-atlas.h"
#include "freetype-gl/include/texture-font.h"
#pragma warning(pop)

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "0mafD6biWOrUwDnuaLbPBkRB6DkLy6BAIB5ppCm6PYfTSTIl0Wl+YBdyGVPiceHCRzixbwP+/9sxNEzM"
#endif

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

  mInplaceString<MAX_PATH> fontFileName;
  uint64_t fontSize;
  texture_font_t *pTextureFont = nullptr;
  uint64_t glyphRange0;
  uint64_t glyphRange1;
  mPtr<mQueue<mPtr<mQueue<mchar_t>>>> additionalGlyphRanges;

  bool operator==(const mFontDescription &description) const
  {
    return fontSize == (uint64_t)roundf(description.fontSize * FontSizeFactor) && description.fontFileName == fontFileName;
  }
};

mFUNCTION(mFontRenderer_Destroy_Internal, mFontRenderer *pFontRenderer);
mFUNCTION(mFontRenderer_UpdateFontAtlas_Internal, mPtr<mFontRenderer> &fontRenderer);
mFUNCTION(mFontRenderer_DrawEnqueuedText_Internal, mPtr<mFontRenderer> &fontRenderer);
mFUNCTION(mFontRenderer_AddFont_Internal, mPtr<mFontRenderer> &fontRenderer, const mFontDescription &fontDescription, OUT size_t *pIndex);
mFUNCTION(mFontRenderer_FontDescriptionAlreadyContainsChar, mPtr<mFontRenderer> &fontRenderer, mFontDescription_Internal *pFontDescription, const float_t originalFontSize, const mchar_t codePoint, const char *character, OUT bool *pContainsChar);
mFUNCTION(mFontRenderer_LoadGlyph_Internal, mPtr<mFontRenderer> &fontRenderer, mFontDescription_Internal *pFontDescription, const size_t queueFontIndex, const float_t originalFontSize, const mchar_t codePoint, const char *character, const size_t bytes, OUT bool *pFailedToLoadChar, const bool attemptToLoadFromBackupFont = true);
mFUNCTION(mFontRenderer_AddCharToFontDescription, mPtr<mFontRenderer> &fontRenderer, mFontDescription_Internal *pFontDescription, const mchar_t codePoint, const char *character);
mFUNCTION(mFontRenderer_LoadGlyphs_Internal, mPtr<mFontRenderer> &fontRenderer, mFontDescription_Internal *pFontDescription, const size_t queueFontIndex, const float_t originalFontSize, const mString &string, OUT bool *pFailedToLoadChars);
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
  mVec3f spacialOrigin;
  mVec3f right, up;
  uint32_t *pIndexBuffer;
  size_t indexBufferCapacity;
  bool addedGlyph;
  mRectangle2D<float_t> renderedArea;
  mPtr<mQueue<mString>> backupFonts;
};

//////////////////////////////////////////////////////////////////////////

mRectangle2D<float_t> mFontDescription_GetDisplayBounds(const mRectangle2D<float_t> &displayBounds)
{
  mVec2f position = displayBounds.position;
  position.y = mRenderParams_CurrentRenderResolutionF.y - displayBounds.position.y - displayBounds.h;

  return mRectangle2D<float_t>(position, displayBounds.size);
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mFontRenderer_Create, OUT mPtr<mFontRenderer> *pFontRenderer, IN mAllocator *pAllocator, const size_t width /* = 2048 */, const size_t height /* = 2048 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pFontRenderer == nullptr, mR_ArgumentNull);

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
  mERROR_CHECK(mQueue_Create(&(*pFontRenderer)->backupFonts, pAllocator));

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

mFUNCTION(mFontRenderer_SetOrigin, mPtr<mFontRenderer> &fontRenderer, const mVec3f position)
{
  mFUNCTION_SETUP();

  mERROR_IF(fontRenderer == nullptr, mR_ArgumentNull);

  fontRenderer->spacialOrigin = position;

  mRETURN_SUCCESS();
}

mFUNCTION(mFontRenderer_SetPosition, mPtr<mFontRenderer> &fontRenderer, const mVec2f position)
{
  mFUNCTION_SETUP();

  mERROR_IF(fontRenderer == nullptr, mR_ArgumentNull);

  fontRenderer->resetPosition = fontRenderer->position = position;

  mERROR_CHECK(mFontRenderer_ResetRenderedRect(fontRenderer));

  mRETURN_SUCCESS();
}

mFUNCTION(mFontRenderer_SetDisplayPosition, mPtr<mFontRenderer> &fontRenderer, const mVec2f position)
{
  mFUNCTION_SETUP();

  mERROR_IF(fontRenderer == nullptr, mR_ArgumentNull);

  mVec2f _position = position;
  _position.y = mRenderParams_CurrentRenderResolutionF.y - position.y;

  fontRenderer->resetPosition = fontRenderer->position = _position;

  mERROR_CHECK(mFontRenderer_ResetRenderedRect(fontRenderer));

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

mFUNCTION(mFontRenderer_ResetRenderedRect, mPtr<mFontRenderer> &fontRenderer)
{
  mFUNCTION_SETUP();

  mERROR_IF(fontRenderer == nullptr, mR_ArgumentNull);

  fontRenderer->renderedArea = mRectangle2D<float_t>(fontRenderer->position, mVec2f(0));

  mRETURN_SUCCESS();
}

mFUNCTION(mFontRenderer_GetRenderedRect, mPtr<mFontRenderer> &fontRenderer, OUT mRectangle2D<float_t> *pRenderedArea)
{
  mFUNCTION_SETUP();

  mERROR_IF(fontRenderer == nullptr || pRenderedArea == nullptr, mR_ArgumentNull);

  *pRenderedArea = fontRenderer->renderedArea;

  mRETURN_SUCCESS();
}

mFUNCTION(mFontRenderer_GetRenderedDisplayRect, mPtr<mFontRenderer> &fontRenderer, OUT mRectangle2D<float_t> *pRenderedArea)
{
  mFUNCTION_SETUP();

  mERROR_IF(fontRenderer == nullptr || pRenderedArea == nullptr, mR_ArgumentNull);

  mRectangle2D<float_t> area = fontRenderer->renderedArea;

  area.position.y = mRenderParams_CurrentRenderResolutionF.y - area.y - area.height;

  *pRenderedArea = area;

  mRETURN_SUCCESS();
}

mFUNCTION(mFontRenderer_AddBackupFont, mPtr<mFontRenderer> &fontRenderer, const mString &backupFont)
{
  mFUNCTION_SETUP();

  mERROR_IF(fontRenderer == nullptr, mR_ArgumentNull);
  mERROR_IF(backupFont.bytes <= 1, mR_InvalidParameter);

  mERROR_CHECK(mQueue_PushBack(fontRenderer->backupFonts, backupFont));

  mRETURN_SUCCESS();
}

mFUNCTION(mFontRenderer_ClearBackupFonts, mPtr<mFontRenderer> &fontRenderer)
{
  mFUNCTION_SETUP();

  mERROR_IF(fontRenderer == nullptr, mR_ArgumentNull);
  
  mERROR_CHECK(mQueue_Clear(fontRenderer->backupFonts));

  mRETURN_SUCCESS();
}

mFUNCTION(mFontRenderer_Begin, mPtr<mFontRenderer> &fontRenderer, const mMatrix &matrix /* = mMatrix::Scale(2.0f / mRenderParams_CurrentRenderResolutionF.x, 2.0f / mRenderParams_CurrentRenderResolutionF.y, 1) * mMatrix::Translation(-1, -1, 0) */, const mVec3f right /* = mVec3f(1, 0, 0) */, const mVec3f up /* = mVec3f(0, 1, 0) */)
{
  mFUNCTION_SETUP();

  mERROR_IF(fontRenderer == nullptr, mR_ArgumentNull);
  mERROR_IF(fontRenderer->started, mR_ResourceStateInvalid);

  fontRenderer->started = true;
  fontRenderer->matrix = matrix;
  fontRenderer->position = fontRenderer->resetPosition;
  fontRenderer->spacialOrigin = mVec3f(0);
  fontRenderer->right = right;
  fontRenderer->up = up;

  mERROR_CHECK(mFontRenderer_ResetRenderedRect(fontRenderer));

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
      mERROR_CHECK(mFontRenderer_LoadGlyph_Internal(fontRenderer, pFontDescription, fontIndex, fontDescription.fontSize, _char.codePoint, _char.character, _char.characterSize, &unloadedChars, !fontDescription.ignoreBackupFonts));

      mERROR_CHECK(mQueue_PointerAt(fontRenderer->textureAtlas->fonts, fontIndex, &pFontDescription)); // Might have moved due to the fontQueue growing if we loaded a character from a newly loaded backup font.

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

        mERROR_CHECK(mFontRenderer_LoadGlyph_Internal(fontRenderer, pFontDescription, fontIndex, fontDescription.fontSize, _char.codePoint, _char.character, _char.characterSize, &unloadedChars, !fontDescription.ignoreBackupFonts));

        mERROR_IF(unloadedChars, mR_ArgumentOutOfBounds);
      }

      texture_glyph_t *pGlyph = texture_font_find_glyph(pFontDescription->pTextureFont, _char.character);

      if (pGlyph == nullptr)
      {
        bool checkBackupFonts = !fontDescription.ignoreBackupFonts;

        if (checkBackupFonts)
        {
          mERROR_CHECK(mQueue_Any(fontRenderer->backupFonts, &checkBackupFonts));

          if (checkBackupFonts)
          {
            for (const mString &_backupFont : fontRenderer->backupFonts->Iterate())
            {
              mFontDescription backupFontDescription;
              backupFontDescription.fontSize = fontDescription.fontSize;
              mERROR_CHECK(mInplaceString_Create(&backupFontDescription.fontFileName, _backupFont));

              size_t backupFontIndex;
              mERROR_CHECK(mFontRenderer_AddFont_Internal(fontRenderer, backupFontDescription, &backupFontIndex));

              mFontDescription_Internal *pBackupFontDescription = nullptr;
              mERROR_CHECK(mQueue_PointerAt(fontRenderer->textureAtlas->fonts, backupFontIndex, &pBackupFontDescription));

              pGlyph = texture_font_find_glyph(pBackupFontDescription->pTextureFont, _char.character);

              if (pGlyph != nullptr)
              {
                if (pGlyph->glyph_index == 0)
                  continue;
                else
                  break;
              }
            }
          }
        }

        if (pGlyph == nullptr)
        {
          if (!failedToLoadChars && !fontRenderer->startedRenderable)
            goto dealWithUnloadedChars;

          pGlyph = texture_font_find_glyph(pFontDescription->pTextureFont, nullptr);

          if (pGlyph == nullptr)
            continue;
        }
      }

      float kerning = 0.0f;

      const char *actualPreviousChar = previousChar;
      bool charFailed = false;

    retry_render:
      if (index > 0)
        kerning = texture_glyph_get_kerning(pGlyph, actualPreviousChar);

      previousChar = _char.character;

      index++;

      fontRenderer->position.x += kerning;

      const float_t x0 = (fontRenderer->position.x + pGlyph->offset_x) * fontDescription.scale;
      const float_t y0 = (fontRenderer->position.y + pGlyph->offset_y) * fontDescription.scale;
      const float_t x1 = (x0 + pGlyph->width * fontDescription.scale);
      const float_t y1 = (y0 - pGlyph->height * fontDescription.scale);

      mRectangle2D<float_t> glyphBounds(x0, y1, x1 - x0, y0 - y1); // Flipped because it's in OpenGL space.

      if (fontDescription.hasBounds && !fontDescription.bounds.Contains(glyphBounds))
      {
        switch (fontDescription.boundMode)
        {
        case mFD_BM_Stop:
        {
          mRETURN_SUCCESS();

          break;
        }

        case mFD_BM_BreakLineOnBound:
        {
          if (charFailed)
            mRETURN_SUCCESS();

          if (fontDescription.bounds.position.y + fontDescription.bounds.size.y > fontRenderer->position.y + fontDescription.lineHeightRatio * fontDescription.fontSize)
          {
            actualPreviousChar = nullptr;
            fontRenderer->position.x = fontRenderer->resetPosition.x;
            fontRenderer->position.y -= fontDescription.lineHeightRatio * fontDescription.fontSize;
            charFailed = true;
            
            goto retry_render;
          }
          else
          {
            mRETURN_SUCCESS();
          }

          break;
        }

        case mFD_BM_StopWithTrippleDot:
        {
          mFontDescription description = fontDescription;
          description.hasBounds = false;

          mERROR_CHECK(mFontRenderer_Draw(fontRenderer, description, "...", colour));

          mRETURN_SUCCESS();
        }

        default:
          break;
        }
      }

      fontRenderer->renderedArea.GrowToContain(glyphBounds);

      const mVec2f halfPixel = mVec2f(.5f) / fontRenderer->textureAtlas->texture.resolutionF;

      const float_t s0 = pGlyph->s0;
      const float_t t0 = pGlyph->t0;
      const float_t s1 = pGlyph->s1 - halfPixel.x;
      const float_t t1 = pGlyph->t1 - halfPixel.y;

      mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, x0 * fontRenderer->right + y0 * fontRenderer->up + fontRenderer->spacialOrigin));
      mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, mVec2f(s0, t0)));
      mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, (mVec4f)colour));

      mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, x0 * fontRenderer->right + y1 * fontRenderer->up + fontRenderer->spacialOrigin));
      mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, mVec2f(s0, t1)));
      mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, (mVec4f)colour));

      mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, x1 * fontRenderer->right + y1 * fontRenderer->up + fontRenderer->spacialOrigin));
      mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, mVec2f(s1, t1)));
      mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, (mVec4f)colour));

      mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, x1 * fontRenderer->right + y0 * fontRenderer->up + fontRenderer->spacialOrigin));
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
  mERROR_CHECK(mQueue_Destroy(&pFontRenderer->backupFonts));

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

  mPROFILE_SCOPED("mFontRenderer_DrawEnqueuedText");

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

  size_t index = (size_t)-1;

  for (const mFontDescription_Internal &_fontDescription : fontRenderer->textureAtlas->fonts->Iterate())
  {
    ++index;

    if (_fontDescription == fontDescription)
    {
      *pIndex = index;
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

mFUNCTION(mFontRenderer_FontDescriptionAlreadyContainsChar, mPtr<mFontRenderer> & /* fontRenderer */, mFontDescription_Internal *pFontDescription, const float_t /* originalFontSize */, const mchar_t codePoint, const char * /* character */, OUT bool *pContainsChar)
{
  mFUNCTION_SETUP();

  *pContainsChar = true;

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
  else if (pFontDescription->additionalGlyphRanges != nullptr)
  {
    const size_t index = codePoint & (mFontDescription_Internal::HashMapSize - 1);

    mPtr<mQueue<mchar_t>> *pQueue = nullptr;
    mERROR_CHECK(mQueue_PointerAt(pFontDescription->additionalGlyphRanges, index, &pQueue));

    for (const mchar_t c : (*pQueue)->Iterate())
    {
      if (c == codePoint)
        mRETURN_SUCCESS();
    }
  }

  *pContainsChar = false;

  mRETURN_SUCCESS();
}

mFUNCTION(mFontRenderer_LoadGlyph_Internal, mPtr<mFontRenderer> &fontRenderer, mFontDescription_Internal *pFontDescription, const size_t queueFontIndex, const float_t originalFontSize, const mchar_t codePoint, const char *character, const size_t /* bytes */, OUT bool *pFailedToLoadChar, const bool attemptToLoadFromBackupFont /* = true */)
{
  mFUNCTION_SETUP();

  *pFailedToLoadChar = false;

  bool containsChar = false;
  mERROR_CHECK(mFontRenderer_FontDescriptionAlreadyContainsChar(fontRenderer, pFontDescription, originalFontSize, codePoint, character, &containsChar));

  if (containsChar)
    mRETURN_SUCCESS();

  bool checkBackupFonts = attemptToLoadFromBackupFont;

  if (checkBackupFonts)
  {
    bool hasBackupFonts = false;
    mERROR_CHECK(mQueue_Any(fontRenderer->backupFonts, &hasBackupFonts));

    // Does nothing if !hasBackupFonts.
    for (const auto &_backupFont : fontRenderer->backupFonts->Iterate())
    {
      if (_backupFont != pFontDescription->fontFileName.c_str())
      {
        mFontDescription fontDescription;
        fontDescription.fontSize = originalFontSize;
        mERROR_CHECK(mInplaceString_Create(&fontDescription.fontFileName, _backupFont));

        size_t fontIndex;
        mERROR_CHECK(mFontRenderer_AddFont_Internal(fontRenderer, fontDescription, &fontIndex));

        mFontDescription_Internal *pBackupFontDescription = nullptr;
        mERROR_CHECK(mQueue_PointerAt(fontRenderer->textureAtlas->fonts, fontIndex, &pBackupFontDescription));
        mERROR_CHECK(mQueue_PointerAt(fontRenderer->textureAtlas->fonts, queueFontIndex, &pFontDescription)); // This pointer might have changed due to the queue growing.

        bool backupFontContainsChar = false;
        mERROR_CHECK(mFontRenderer_FontDescriptionAlreadyContainsChar(fontRenderer, pBackupFontDescription, originalFontSize, codePoint, character, &backupFontContainsChar));

        if (backupFontContainsChar)
          mRETURN_SUCCESS();
      }
    }
  }

  mPROFILE_SCOPED("mFontRenderer_LoadGlyph");

  // Load glyph.
  size_t glyphLoaded = texture_font_load_glyph(pFontDescription->pTextureFont, character, checkBackupFonts ? TRUE : FALSE);

  if (glyphLoaded != 0)
  {
    texture_glyph_t *pGlyph = texture_font_find_glyph(pFontDescription->pTextureFont, character);
    
    if (pGlyph->glyph_index == 0)
      glyphLoaded = 0;
  }

  if (checkBackupFonts && glyphLoaded == 0)
  {
    for (const auto &_backupFont : fontRenderer->backupFonts->Iterate())
    {
      if (_backupFont != pFontDescription->fontFileName.c_str())
      {
        // We've already checked earlier: This font does not contain the glyph.

        mFontDescription fontDescription;
        fontDescription.fontSize = originalFontSize;
        mERROR_CHECK(mInplaceString_Create(&fontDescription.fontFileName, _backupFont));

        size_t fontIndex;
        mERROR_CHECK(mFontRenderer_AddFont_Internal(fontRenderer, fontDescription, &fontIndex));

        mFontDescription_Internal *pBackupFontDescription = nullptr;
        mERROR_CHECK(mQueue_PointerAt(fontRenderer->textureAtlas->fonts, fontIndex, &pBackupFontDescription));
        mERROR_CHECK(mQueue_PointerAt(fontRenderer->textureAtlas->fonts, queueFontIndex, &pFontDescription)); // This pointer might have changed due to the queue growing.
        
        // Load glyph.
        size_t backupGlyphLoaded = texture_font_load_glyph(pBackupFontDescription->pTextureFont, character, TRUE);

        if (backupGlyphLoaded != 0)
        {
          texture_glyph_t *pGlyph = texture_font_find_glyph(pBackupFontDescription->pTextureFont, character);

          if (pGlyph->glyph_index == 0)
            backupGlyphLoaded = 0;
        }

        if (backupGlyphLoaded != 0)
        {
          mERROR_CHECK(mFontRenderer_AddCharToFontDescription(fontRenderer, pBackupFontDescription, codePoint, character));

          *pFailedToLoadChar = false;

          mRETURN_SUCCESS();
        }
      }
    }

    glyphLoaded = texture_font_load_glyph(pFontDescription->pTextureFont, character, FALSE);
  }

  *pFailedToLoadChar = (0 == glyphLoaded);

  if (!*pFailedToLoadChar)
    mERROR_CHECK(mFontRenderer_AddCharToFontDescription(fontRenderer, pFontDescription, codePoint, character));

  mRETURN_SUCCESS();
}

mFUNCTION(mFontRenderer_AddCharToFontDescription, mPtr<mFontRenderer> &fontRenderer, mFontDescription_Internal *pFontDescription, const mchar_t codePoint, const char * /* character */)
{
  mFUNCTION_SETUP();

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

    mERROR_CHECK(mQueue_PushBack(*pQueue, codePoint));
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mFontRenderer_SaveAtlasTo, mPtr<mFontRenderer> &fontRenderer, const mString &directory)
{
  mFUNCTION_SETUP();

  mERROR_IF(fontRenderer == nullptr, mR_ArgumentNull);
  mERROR_IF(directory.hasFailed, mR_InvalidParameter);

  mPtr<mImageBuffer> imageBuffer;
  mString filename;

  if (fontRenderer->textureAtlas != nullptr)
  {
    mERROR_CHECK(mTexture_Download(fontRenderer->textureAtlas->texture, &imageBuffer, &mDefaultTempAllocator, mPF_Monochrome8));

    mERROR_CHECK(mString_Create(&filename, directory, &mDefaultTempAllocator));
    mERROR_CHECK(mString_Append(filename, "/atlas_active"));
    mERROR_CHECK(mString_Append(filename, ".png"));

    mERROR_CHECK(mImageBuffer_SaveAsPng(imageBuffer, filename));
  }

  size_t index = 0;

  for (auto &_atlas : fontRenderer->currentFrameTextureAtlasses->Iterate())
  {
    mERROR_CHECK(mTexture_Download(_atlas->texture, &imageBuffer, &mDefaultTempAllocator, mPF_Monochrome8));

    mERROR_CHECK(mString_Create(&filename, directory, &mDefaultTempAllocator));
    mERROR_CHECK(mString_Append(filename, "/atlas"));
    mERROR_CHECK(mString_AppendUnsignedInteger(filename, index++));
    mERROR_CHECK(mString_Append(filename, ".png"));

    mERROR_CHECK(mImageBuffer_SaveAsPng(imageBuffer, filename));
  }

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mFontRenderer_LoadGlyphs_Internal, mPtr<mFontRenderer> &fontRenderer, mFontDescription_Internal *pFontDescription, const size_t queueFontIndex, const float_t originalFontSize, const mString &string, OUT bool *pFailedToLoadChars)
{
  mFUNCTION_SETUP();

  *pFailedToLoadChars = false;

  for (auto &&_char : string)
  {
    mFUNCTION_SETUP();

    bool unloadedChar = false;
    mERROR_CHECK(mFontRenderer_LoadGlyph_Internal(fontRenderer, pFontDescription, queueFontIndex, originalFontSize, _char.codePoint, _char.character, _char.characterSize, &unloadedChar));

    if (unloadedChar)
    {
      *pFailedToLoadChars = true;
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
