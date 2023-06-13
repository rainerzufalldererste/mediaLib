#include "mFontRenderer.h"

#include "mFontRenderer_Internal.h"

#include "mQueue.h"
#include "mShader.h"
#include "mIndexedRenderDataBuffer.h"
#include "mBinaryChunk.h"
#include "mProfiler.h"

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

// This looks absolutely gorgeous with larger text, however with very small text it seems like non-sdf is more legible.
// there's also the issue that we're not entirely sure if we're rendering to an LCD anyways and the subpixel size will be horribly wrong when rendering to a projected texture.
// Inspired By: https://drewcassidy.me/2020/06/26/sdf-antialiasing/
const char *mFontRenderer_FragmentShaderSDF = R"__SHADER__(#version 150 core

uniform sampler2D texture;
uniform vec2 subPixel;

out vec4 colour;

in vec2 _texCoord;
in vec4 _colour;

const float cutoff = 0.475;
const float icutoff = 1 / cutoff;
const float inv3 = 1.0 / 3.0;
const float upDownFac = 0.85;

float getAlphaAtOffset(vec2 offset)
{
  float dist = max(0, cutoff - texture2D(texture, _texCoord + offset).r);
  float alpha = dist * icutoff;
  alpha = 1 - alpha * alpha;

  return alpha;
}

void main()
{
  float alphaR = max(max(getAlphaAtOffset(vec2(-subPixel)) * upDownFac, getAlphaAtOffset(vec2(-subPixel.x, 0))), getAlphaAtOffset(vec2(-subPixel.x, subPixel.y)) * upDownFac);
  float alphaG = max(max(getAlphaAtOffset(vec2(0, -subPixel.y)) * upDownFac, getAlphaAtOffset(vec2(0))), getAlphaAtOffset(vec2(0, subPixel.y)) * upDownFac);
  float alphaB = max(max(getAlphaAtOffset(vec2(subPixel.x, -subPixel.y)) * upDownFac, getAlphaAtOffset(vec2(subPixel.x, 0))), getAlphaAtOffset(vec2(subPixel)) * upDownFac);

  colour.a = _colour.a * (alphaR + alphaG + alphaB) / 3.0;
  colour.r = mix(1 - _colour.r, _colour.r, alphaR);
  colour.g = mix(1 - _colour.g, _colour.g, alphaG);
  colour.b = mix(1 - _colour.b, _colour.b, alphaB);
}
)__SHADER__";

extern const char mFontRenderer_PositionAttribute[] = "vertex";
extern const char mFontRenderer_TextureCoordAttribute[] = "tex_coord";
extern const char mFontRenderer_ColourAttribute[] = "color";
const char mFontRenderer_MatrixUniform[] = "matrix";
const char mFontRenderer_TextureUniform[] = "texture";
const char mFontRenderer_SubPixelUniform[] = "subPixel";

struct mFontDescription_Internal
{
  constexpr static float_t FontSizeLog = 0.07038932789139789; // log2(1.05)
  constexpr static float_t FontSizeInvLog = 1.f / FontSizeLog;

  mInplaceString<MAX_PATH> fontFileName;
  float_t fontSizePx;
  uint32_t fontSizeIndex;
  texture_font_t *pTextureFont = nullptr;
  uint64_t glyphRange0;
  uint64_t glyphRange1;
  size_t fontInfoIndex;
  mUniqueContainer<mQueue<mchar_t>> additionalGlyphRanges;

  inline static uint32_t GetFontSizeIndex(const float_t fontSize)
  {
    if (fontSize <= 14)
      return (uint32_t)fontSize;
    else
      return (uint32_t)(mLog2(fontSize) * FontSizeInvLog) - 54 + 14; // floor(log2(9) * log2(1.1)) = 54.
  }

  inline bool Matches(const mFontDescription &description, const uint32_t calculatedFontSizeIndex) const
  {
    return fontSizeIndex == calculatedFontSizeIndex && description.fontFileName == fontFileName;
  }
};

static mFUNCTION(mDestruct, mFontDescription_Internal *pFontDescription)
{
  texture_font_delete(pFontDescription->pTextureFont);
  pFontDescription->pTextureFont = nullptr;

  pFontDescription->~mFontDescription_Internal();

  return mR_Success;
}

struct mFontRenderer_KerningInfo
{
  float_t value;
  uint32_t uses;
};

struct mFontRenderer_GlyphInfo
{
  mchar_t glyph;
  size_t glyphUses, totalKerningUses;
  float_t approxOffsetX, approxOffsetY; // the larger the font size when rendering those chars, the more accurate this will become.
  float_t advanceX, advanceY, sizeX, sizeY;
  mUniqueContainer<mQueue<mKeyValuePair<mchar_t, mFontRenderer_KerningInfo>>> kerningInfo;

  bool operator==(const mchar_t codePoint) const
  {
    return glyph == codePoint;
  }
};

struct mFontRenderer_FontInfo
{
  mInplaceString<MAX_PATH> fontFileName;
  mUniqueContainer<mQueue<mKeyValuePair<mchar_t, mFontRenderer_GlyphInfo>>> glyphInfo;
  mUniqueContainer<mQueue<mchar_t>> missingGlyphs;
  size_t totalUses;

  bool operator==(const mInplaceString<MAX_PATH> &fileName) const
  {
    return fontFileName == fileName;
  }
};

struct mFontTextureAtlas
{
  mUniqueContainer<mQueue<mFontDescription_Internal>> fonts;
  texture_atlas_t *pTextureAtlas;
  mTexture texture;
  size_t lastSize;
};

enum mFontRenderer_PhraseRenderGlyphInfo_Type
{
  mFR_PRGI_CT_GlyphInfo,
  mFR_PRGI_CT_SwitchTextureAtlas,
  mFR_PRGI_CT_AdvanceOnly,
};

struct mFontRenderer_PhraseRenderGlyphInfo
{
  mFontRenderer_PhraseRenderGlyphInfo_Type commandType;

  float_t s0, t0, s1, t1, advanceX, advanceY, kerning;
  float_t offsetX, offsetY;
  float_t sizeX, sizeY;

  mPtr<mFontTextureAtlas> nextTextureAtlas;
};

static mFUNCTION(mFontRenderer_Destroy_Internal, mFontRenderer *pFontRenderer);
static mFUNCTION(mFontRenderer_UpdateFontAtlas_Internal, mPtr<mFontRenderer> &fontRenderer);
static mFUNCTION(mFontRenderer_DrawEnqueuedText_Internal, mPtr<mFontRenderer> &fontRenderer);
static mFUNCTION(mFontRenderer_AddFont_Internal, mPtr<mFontRenderer> &fontRenderer, mPtr<mFontTextureAtlas> &fontTextureAtlas, const mFontDescription &fontDescription, OUT size_t *pIndex, OUT bool *pAdded);
static mFUNCTION(mFontRenderer_FontDescriptionAlreadyContainsChar, mPtr<mFontRenderer> &fontRenderer, mFontDescription_Internal *pFontDescription, const float_t originalFontSize, const mchar_t codePoint, OUT bool *pContainsChar);
static mFUNCTION(mFontRenderer_LoadGlyph_Internal, mPtr<mFontRenderer> &fontRenderer, mPtr<mFontTextureAtlas> &fontTextureAtlas, mFontDescription_Internal *pFontDescription, const size_t queueFontIndex, const float_t originalFontSize, const mchar_t codePoint, OUT texture_font_t **ppFont, OUT texture_glyph_t **ppGlyph, const bool attemptToLoadFromBackupFont /* = true */);
static mFUNCTION(mFontRenderer_AddCharToFontDescription, mPtr<mFontRenderer> &fontRenderer, mFontDescription_Internal *pFontDescription, const mchar_t codePoint);
static mFUNCTION(mFontRenderer_AddCharNotFoundToFontDescription, mPtr<mFontRenderer> &fontRenderer, mFontDescription_Internal *pFontDescription, const mchar_t codePoint, texture_glyph_t *pGlyph);
static mFUNCTION(mFontRenderer_GetGlyphInfo_Internal, mPtr<mFontRenderer> &fontRenderer, mFontRenderer_FontInfo **ppFontInfo, const size_t fontInfoIndex, const mFontDescription &fontDescription, mFontDescription_Internal **ppFontDescription, size_t *pQueueFontIndex, const mchar_t codePoint, const char *character, const size_t bytes, const bool checkBackupFonts, OUT mFontRenderer_GlyphInfo **ppGlyphInfo, OUT bool *pRequireAtlasFlush, OUT OPTIONAL bool *pCharNotFound_NullptrIfNotInternalCall = nullptr);
static mFUNCTION(mFontRenderer_TryRenderPhrase, mPtr<mFontRenderer> &fontRenderer, const mString &phrase, const mFontDescription &fontDescription, OUT mPtr<mQueue<mFontRenderer_PhraseRenderGlyphInfo>> &renderInfo, OUT mVec2f *pSize);

static mFUNCTION(mFontTextureAtlas_Create, OUT mPtr<mFontTextureAtlas> *pAtlas, mAllocator *pAllocator, const size_t width, const size_t height);
static mFUNCTION(mFontTextureAtlas_Destroy, IN mFontTextureAtlas *pAtlas);
static mFUNCTION(mFontTextureAtlas_RequiresUpload, mPtr<mFontTextureAtlas> &atlas, OUT bool *pNeedsUpload);
static mFUNCTION(mFontTextureAtlas_ClearGlyphs_Internal, mPtr<mFontTextureAtlas> &textureAtlas);

static mFUNCTION(mFontRenderable_Destroy_Internal, IN mFontRenderable *pFontRenderable);

struct mFontRenderable
{
  mPtr<mFontTextureAtlas> textureAtlas;
  mIndexedRenderDataBuffer<mRDB_FloatAttribute<3, mFontRenderer_PositionAttribute>, mRDB_FloatAttribute<2, mFontRenderer_TextureCoordAttribute>, mRDB_FloatAttribute<4, mFontRenderer_ColourAttribute>> indexDataBuffer;
};

//////////////////////////////////////////////////////////////////////////

static texture_font_library_t *mFontRenderer_pFontLibrary = nullptr;

//////////////////////////////////////////////////////////////////////////

struct mFontRenderer
{
  mUniqueContainer<mQueue<mFontRenderer_FontInfo>> fontInfo;
  mPtr<mFontTextureAtlas> textureAtlas;
  mPtr<mFontTextureAtlas> phraseRenderingTextureAtlas;
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
  bool stoppedAtStartX, stoppedAtStartY, stoppedAtEndX, stoppedAtEndY;
  mchar_t previousChar;
  bool signedDistanceFieldRendering;
  mUniqueContainer<mQueue<mFontRenderer_PhraseRenderGlyphInfo>> phraseRenderInfo, entirePhraseRenderInfo;
  mString phraseString;
};

//////////////////////////////////////////////////////////////////////////

mRectangle2D<float_t> mFontDescription_GetDisplayBounds(const mRectangle2D<float_t> &displayBounds)
{
  mVec2f position = displayBounds.position;
  position.y = mRenderParams_CurrentRenderResolutionF.y - displayBounds.position.y - displayBounds.h;

  return mRectangle2D<float_t>(position, displayBounds.size);
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mFontRenderer_Create, OUT mPtr<mFontRenderer> *pFontRenderer, IN mAllocator *pAllocator, const size_t width /* = 2048 */, const size_t height /* = 2048 */, const bool useSignedDistanceFieldRendering /* = false */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pFontRenderer == nullptr, mR_ArgumentNull);

  if (mFontRenderer_pFontLibrary == nullptr)
    mERROR_CHECK(texture_library_new(&mFontRenderer_pFontLibrary, pAllocator));

  mERROR_IF(mFontRenderer_pFontLibrary == nullptr, mR_InternalError);

  mDEFER_CALL_ON_ERROR(pFontRenderer, mFontRenderer_Destroy);
  mERROR_CHECK(mSharedPointer_Allocate(pFontRenderer, pAllocator, (std::function<void(mFontRenderer *)>)[](mFontRenderer *pData) {mFontRenderer_Destroy_Internal(pData);}, 1));

  (*pFontRenderer)->pAllocator = pAllocator;
  (*pFontRenderer)->started = false;
  (*pFontRenderer)->startedRenderable = false;
  (*pFontRenderer)->addedGlyph = false;
  (*pFontRenderer)->signedDistanceFieldRendering = useSignedDistanceFieldRendering;

  mERROR_CHECK(mString_Create(&(*pFontRenderer)->phraseString, "", pAllocator));
  mERROR_CHECK(mQueue_Create(&(*pFontRenderer)->phraseRenderInfo, pAllocator));
  mERROR_CHECK(mQueue_Create(&(*pFontRenderer)->entirePhraseRenderInfo, pAllocator));

  mERROR_CHECK(mBinaryChunk_Create(&(*pFontRenderer)->enqueuedText, pAllocator));

  mERROR_CHECK(mFontTextureAtlas_Create(&(*pFontRenderer)->textureAtlas, pAllocator, width, height));

  mERROR_CHECK(mIndexedRenderDataBuffer_Create(&(*pFontRenderer)->indexDataBuffer, pAllocator, mFontRenderer_VertexShader, useSignedDistanceFieldRendering ? mFontRenderer_FragmentShaderSDF : mFontRenderer_FragmentShader, true, false));
  mERROR_CHECK(mQueue_Create(&(*pFontRenderer)->currentFrameTextureAtlasses, pAllocator));
  mERROR_CHECK(mQueue_Create(&(*pFontRenderer)->availableTextureAtlasses, pAllocator));
  mERROR_CHECK(mQueue_Create(&(*pFontRenderer)->backupFonts, pAllocator));
  mERROR_CHECK(mQueue_Create(&(*pFontRenderer)->fontInfo, pAllocator));

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

  bool added;
  size_t unused;
  mERROR_CHECK(mFontRenderer_AddFont_Internal(fontRenderer, fontRenderer->textureAtlas, fontDescription, &unused, &added));

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

mFUNCTION(mFontRenderer_IsStarted, mPtr<mFontRenderer> &fontRenderer, OUT bool *pStarted)
{
  mFUNCTION_SETUP();

  mERROR_IF(fontRenderer == nullptr || pStarted == nullptr, mR_ArgumentNull);

  *pStarted = fontRenderer->started;

  mRETURN_SUCCESS();
}

mFUNCTION(mFontRenderer_ResetRenderedRect, mPtr<mFontRenderer> &fontRenderer)
{
  mFUNCTION_SETUP();

  mERROR_IF(fontRenderer == nullptr, mR_ArgumentNull);

  fontRenderer->renderedArea = mRectangle2D<float_t>(fontRenderer->position, mVec2f(0));
  fontRenderer->stoppedAtStartX = fontRenderer->stoppedAtStartY = fontRenderer->stoppedAtEndX = fontRenderer->stoppedAtEndY = false;

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

mFUNCTION(mFontRenderer_LastDrawCallStoppedAtBounds, mPtr<mFontRenderer> &fontRenderer, OUT OPTIONAL bool *pStopped, OUT OPTIONAL bool *pStoppedAtStartX /* = nullptr */, OUT OPTIONAL bool *pStoppedAtStartY /* = nullptr */, OUT OPTIONAL bool *pStoppedAtEndX /* = nullptr */, OUT OPTIONAL bool *pStoppedAtEndY /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_IF(fontRenderer == nullptr, mR_ArgumentNull);

  if (pStopped != nullptr)
    *pStopped = fontRenderer->stoppedAtStartX | fontRenderer->stoppedAtStartY | fontRenderer->stoppedAtEndX | fontRenderer->stoppedAtEndY;

  if (pStoppedAtStartX != nullptr)
    *pStoppedAtStartX = fontRenderer->stoppedAtStartX;

  if (pStoppedAtStartY != nullptr)
    *pStoppedAtStartY = fontRenderer->stoppedAtStartY;

  if (pStoppedAtEndX != nullptr)
    *pStoppedAtEndX = fontRenderer->stoppedAtEndX;

  if (pStoppedAtEndY != nullptr)
    *pStoppedAtEndY = fontRenderer->stoppedAtEndY;

  mRETURN_SUCCESS();
}

mFUNCTION(mFontRenderer_LastDrawCallStoppedAtDisplayBounds, mPtr<mFontRenderer> &fontRenderer, OUT OPTIONAL bool *pStopped, OUT OPTIONAL bool *pStoppedAtStartX /* = nullptr */, OUT OPTIONAL bool *pStoppedAtStartY /* = nullptr */, OUT OPTIONAL bool *pStoppedAtEndX /* = nullptr */, OUT OPTIONAL bool *pStoppedAtEndY /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_IF(fontRenderer == nullptr, mR_ArgumentNull);

  if (pStopped != nullptr)
    *pStopped = fontRenderer->stoppedAtStartX | fontRenderer->stoppedAtStartY | fontRenderer->stoppedAtEndX | fontRenderer->stoppedAtEndY;

  if (pStoppedAtStartX != nullptr)
    *pStoppedAtStartX = fontRenderer->stoppedAtEndX;

  if (pStoppedAtStartY != nullptr)
    *pStoppedAtStartY = fontRenderer->stoppedAtEndY;

  if (pStoppedAtEndX != nullptr)
    *pStoppedAtEndX = fontRenderer->stoppedAtStartX;

  if (pStoppedAtEndY != nullptr)
    *pStoppedAtEndY = fontRenderer->stoppedAtStartY;

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

  mDEFER_CALL_ON_ERROR(pRenderable, mSharedPointer_Destroy);
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

  fontRenderer->previousChar = 0;

  return mFontRenderer_DrawContinue(fontRenderer, fontDescription, string, colour);
}

mFUNCTION(mFontRenderer_DrawContinue, mPtr<mFontRenderer> &fontRenderer, const mFontDescription &fontDescription, const mString &string, const mVector colour)
{
  mFUNCTION_SETUP();

  mERROR_IF(fontRenderer == nullptr, mR_ArgumentNull);
  mERROR_IF(!fontRenderer->started, mR_ResourceStateInvalid);

  bool added;
  size_t fontIndex;
  mERROR_CHECK(mFontRenderer_AddFont_Internal(fontRenderer, fontRenderer->textureAtlas, fontDescription, &fontIndex, &added));

  mFontDescription_Internal *pFontDescription = nullptr;
  mERROR_CHECK(mQueue_PointerAt(fontRenderer->textureAtlas->fonts, fontIndex, &pFontDescription));

  const float_t sizeFactor = fontDescription.fontSize / pFontDescription->fontSizePx;

  size_t index = 0;

  for (auto &&_char : string)
  {
    if (*_char.character == '\n')
    {
      fontRenderer->previousChar = _char.codePoint;
      fontRenderer->position.x = fontRenderer->resetPosition.x;
      fontRenderer->position.y -= fontDescription.lineHeightRatio * fontDescription.fontSize;
    }
    else
    {
      texture_font_t *pFont = nullptr;
      texture_glyph_t *pGlyph = nullptr;

      // Load Glyph.
      {
        const mResult result = mFontRenderer_LoadGlyph_Internal(fontRenderer, fontRenderer->textureAtlas, pFontDescription, fontIndex, fontDescription.fontSize, _char.codePoint, &pFont, &pGlyph, !fontDescription.ignoreBackupFonts);

        if (mFAILED(result))
        {
          mSILENT_ERROR_IF(result != mR_ResourceBusy, result);

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

              mERROR_CHECK(mFontTextureAtlas_ClearGlyphs_Internal(fontRenderer->textureAtlas));
            }
            else
            {
              // If no texture atlas available: create a new one.
              mERROR_CHECK(mFontTextureAtlas_Create(&fontRenderer->textureAtlas, fontRenderer->pAllocator, fontRenderer->textureAtlas->pTextureAtlas->width, fontRenderer->textureAtlas->pTextureAtlas->height));
            }

            mERROR_CHECK(mFontRenderer_AddFont_Internal(fontRenderer, fontRenderer->textureAtlas, fontDescription, &fontIndex, &added));
            mERROR_CHECK(mQueue_PointerAt(fontRenderer->textureAtlas->fonts, fontIndex, &pFontDescription));
          }

          mERROR_CHECK(mFontRenderer_LoadGlyph_Internal(fontRenderer, fontRenderer->textureAtlas, pFontDescription, fontIndex, fontDescription.fontSize, _char.codePoint, &pFont, &pGlyph, !fontDescription.ignoreBackupFonts));
        }
      }

      mASSERT_DEBUG(pGlyph != nullptr, "Glyph should've been retrieved in `mFontRenderer_LoadGlyph_Internal`.");

      mERROR_CHECK(mQueue_PointerAt(fontRenderer->textureAtlas->fonts, fontIndex, &pFontDescription)); // Might have moved due to the fontQueue growing if we loaded a character from a newly loaded backup font.

      float_t kerning = 0.0f;

      mchar_t actualPreviousChar = fontRenderer->previousChar;
      bool charFailed = false;

    retry_render:
      if (index > 0)
        /* Let's ignore errors for now. */ mSILENCE_ERROR(texture_glyph_generate_kerning_on_demand(pFont, pGlyph, actualPreviousChar, &kerning));

      fontRenderer->previousChar = _char.codePoint;

      index++;

      fontRenderer->position.x += kerning * sizeFactor;

      const float_t x0 = (fontRenderer->position.x + pGlyph->offset_x * sizeFactor) * fontDescription.scale;
      const float_t y0 = (fontRenderer->position.y + pGlyph->offset_y * sizeFactor) * fontDescription.scale;
      const float_t x1 = (x0 + pGlyph->width * sizeFactor * fontDescription.scale);
      const float_t y1 = (y0 - pGlyph->height * sizeFactor * fontDescription.scale);

      mRectangle2D<float_t> glyphBounds(x0, y1, x1 - x0, y0 - y1); // Flipped because it's in OpenGL space.

      if (fontDescription.hasBounds && !fontDescription.bounds.Contains(glyphBounds))
      {
        fontRenderer->stoppedAtStartX = glyphBounds.x < fontDescription.bounds.x;
        fontRenderer->stoppedAtStartY = glyphBounds.y < fontDescription.bounds.y;
        fontRenderer->stoppedAtEndX = glyphBounds.x + glyphBounds.w > fontDescription.bounds.x;
        fontRenderer->stoppedAtEndY = glyphBounds.y + glyphBounds.h > fontDescription.bounds.y;

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
            actualPreviousChar = 0;
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

      const mVec2f halfPixel = mVec2f(.75f) / fontRenderer->textureAtlas->texture.resolutionF; // not entirely sure why, but without this being nudged to 0.75 we're getting weird off-by-half-a-texel rendering errors.

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

mFUNCTION(mFontRenderer_DrawWithLayout, mPtr<mFontRenderer> &fontRenderer, const mFontDescription &fontDescription, const mTextLayout layout, const mString &text, const bool setPosition /* = false */, const mVector colour /* = mVector(1, 1, 1, 1) */)
{
  mFUNCTION_SETUP();

  mERROR_IF(fontRenderer == nullptr, mR_ArgumentNull);
  mERROR_IF(!fontRenderer->started, mR_ResourceStateInvalid);

  mDEFER_CALL(&fontRenderer->phraseRenderingTextureAtlas, mSharedPointer_Destroy);
  fontRenderer->phraseRenderingTextureAtlas = fontRenderer->textureAtlas;

  fontRenderer->previousChar = 0;

  const char *startText = text.c_str();
  bool lastWasEndPhrase = true;
  mVec2f phraseSize;
  float_t size = 0;
  bool firstPhrase = fontDescription.hasBounds;

  mERROR_CHECK(mQueue_Clear(fontRenderer->entirePhraseRenderInfo));

  mFontRenderer_PhraseRenderGlyphInfo spaceGlyphInfo;
  const bool phraseCanEndAtSpace = fontDescription.hasBounds;

  {
    mERROR_CHECK(mString_Create(&fontRenderer->phraseString, " ", fontRenderer->pAllocator));
    mERROR_CHECK(mFontRenderer_TryRenderPhrase(fontRenderer, fontRenderer->phraseString, fontDescription, fontRenderer->phraseRenderInfo, &phraseSize));

    mERROR_IF(fontRenderer->phraseRenderInfo->count != 1, mR_Failure);
    mERROR_CHECK(mQueue_PopFront(fontRenderer->phraseRenderInfo, &spaceGlyphInfo));
  }

  if (fontDescription.hasBounds && setPosition)
  {
    fontRenderer->position = fontDescription.bounds.position + mVec2f(0, fontDescription.bounds.size.y - fontDescription.fontSize);
    
    if (layout.alignment == mTL_A_Centered)
      fontRenderer->position.x += fontDescription.bounds.width * .5f;
    else if (layout.alignment == mTL_A_Right)
      fontRenderer->position.x += fontDescription.bounds.width;

    fontRenderer->resetPosition = fontRenderer->position;

    mERROR_CHECK(mFontRenderer_ResetRenderedRect(fontRenderer));
  }

  const mchar_t space = mToChar<2>(" ");
  const mchar_t tab = mToChar<2>("\t");
  const mchar_t newLine = mToChar<2>("\n");

  size_t index = (size_t)-1;

  bool justWrapped = false;
  float_t hideableSize = 0;

  for (const auto &_char : text)
  {
    ++index;

    const bool isLast = (index == (text.count - 2));
    bool endPhrase = false;

    if (phraseCanEndAtSpace)
      endPhrase |= (_char.codePoint == space || _char.codePoint == tab);

    const bool isNewLine = (_char.codePoint == newLine);

    endPhrase |= isNewLine;

    if (endPhrase || isLast)
    {
      if (lastWasEndPhrase && endPhrase)
      {
        if (isNewLine)
        {
          fontRenderer->position.x = fontRenderer->resetPosition.x;
          fontRenderer->position.y -= fontDescription.lineHeightRatio * fontDescription.fontSize;
          size = 0;
        }
        else if (!justWrapped)
        {
          const float_t sizeFactor = _char.codePoint == tab ? 2.f : 1.f;

          hideableSize += sizeFactor * spaceGlyphInfo.advanceX;
        }
      }
      else
      {
        // Get Phrase.
        if (!isLast)
          mERROR_CHECK(mString_Create(&fontRenderer->phraseString, startText, _char.character - startText, fontRenderer->pAllocator));
        else
          mERROR_CHECK(mString_Create(&fontRenderer->phraseString, startText, fontRenderer->pAllocator));

        mERROR_CHECK(mFontRenderer_TryRenderPhrase(fontRenderer, fontRenderer->phraseString, fontDescription, fontRenderer->phraseRenderInfo, &phraseSize));

        bool charsAdded = false;
        bool doesntFit = fontDescription.hasBounds && fontDescription.bounds.width <= hideableSize + size + phraseSize.x;

        if (isLast || isNewLine || doesntFit)
        {
          if (!doesntFit)
          {
            for (const auto &_item : fontRenderer->phraseRenderInfo->Iterate())
              mERROR_CHECK(mQueue_PushBack(fontRenderer->entirePhraseRenderInfo, _item));

            size += phraseSize.x;
            hideableSize = 0;
            charsAdded = true;
          }

          justWrapped = true;

          if (layout.alignment == mTL_A_Centered)
            fontRenderer->position.x = fontRenderer->resetPosition.x - (size / fontDescription.scale + spaceGlyphInfo.advanceX) * .5f;
          else if (layout.alignment == mTL_A_Right)
            fontRenderer->position.x = fontRenderer->resetPosition.x - (size / fontDescription.scale + spaceGlyphInfo.advanceX);

          bool first = firstPhrase;
          firstPhrase = false;

          // Render.
          for (const auto &_item : fontRenderer->entirePhraseRenderInfo->Iterate())
          {
            if (_item.commandType == mFR_PRGI_CT_SwitchTextureAtlas)
            {
              mERROR_CHECK(mFontRenderer_DrawEnqueuedText_Internal(fontRenderer));
              mERROR_CHECK(mQueue_PushBack(fontRenderer->currentFrameTextureAtlasses, std::move(fontRenderer->textureAtlas)));
              fontRenderer->textureAtlas = _item.nextTextureAtlas;
            }
            else if (_item.commandType == mFR_PRGI_CT_AdvanceOnly)
            {
              fontRenderer->position.x += _item.advanceX;
            }
            else
            {
              fontRenderer->position.x += _item.kerning;

              const float_t x0 = (fontRenderer->position.x + _item.offsetX) * fontDescription.scale;
              const float_t y0 = (fontRenderer->position.y + _item.offsetY) * fontDescription.scale;
              const float_t x1 = (x0 + _item.sizeX * fontDescription.scale);
              const float_t y1 = (y0 - _item.sizeY * fontDescription.scale);

              mRectangle2D<float_t> glyphBounds(x0, y1, x1 - x0, y0 - y1); // Flipped because it's in OpenGL space.

              if (first)
              {
                if (fontDescription.hasBounds && !fontDescription.bounds.Contains(glyphBounds))
                {
                  // HACK: Sometimes it seems like we're a tiny amount of pixels off, when rendering center- (and probably also right-) aligned. (But make sure it doesn't render one line before or after the bounds)
                  if ((glyphBounds.y + glyphBounds.height > fontDescription.bounds.y + fontDescription.bounds.height || glyphBounds.y < fontDescription.bounds.y) || (glyphBounds.size.LengthSquared() - fontDescription.bounds.Intersect(glyphBounds).size.LengthSquared() > mVec2f(spaceGlyphInfo.advanceX * .5f, glyphBounds.height).LengthSquared() && fontDescription.bounds.position.y <= glyphBounds.y && fontDescription.bounds.position.y + fontDescription.bounds.height >= glyphBounds.y + glyphBounds.height))
                  {
                    fontRenderer->stoppedAtStartX = glyphBounds.x < fontDescription.bounds.x;
                    fontRenderer->stoppedAtStartY = glyphBounds.y < fontDescription.bounds.y;
                    fontRenderer->stoppedAtEndX = glyphBounds.x + glyphBounds.w > fontDescription.bounds.x;
                    fontRenderer->stoppedAtEndY = glyphBounds.y + glyphBounds.h > fontDescription.bounds.y;

                    if (fontDescription.boundMode == mFD_BM_StopWithTrippleDot)
                    {
                      mFontDescription description = fontDescription;
                      description.hasBounds = false;

                      mERROR_CHECK(mFontRenderer_Draw(fontRenderer, description, "...", colour));
                    }

                    mRETURN_SUCCESS();
                  }
                }

                first = false;
              }

              fontRenderer->renderedArea.GrowToContain(glyphBounds);

              mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, x0 * fontRenderer->right + y0 * fontRenderer->up + fontRenderer->spacialOrigin));
              mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, mVec2f(_item.s0, _item.t0)));
              mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, (mVec4f)colour));

              mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, x0 * fontRenderer->right + y1 * fontRenderer->up + fontRenderer->spacialOrigin));
              mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, mVec2f(_item.s0, _item.t1)));
              mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, (mVec4f)colour));

              mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, x1 * fontRenderer->right + y1 * fontRenderer->up + fontRenderer->spacialOrigin));
              mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, mVec2f(_item.s1, _item.t1)));
              mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, (mVec4f)colour));

              mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, x1 * fontRenderer->right + y0 * fontRenderer->up + fontRenderer->spacialOrigin));
              mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, mVec2f(_item.s1, _item.t0)));
              mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, (mVec4f)colour));

              fontRenderer->position.x += _item.advanceX;
            }
          }

          mERROR_CHECK(mQueue_Clear(fontRenderer->entirePhraseRenderInfo));

          if ((isNewLine || isLast) && doesntFit)
          {
            if (fontDescription.hasBounds)
            {
              if (fontRenderer->position.y - fontDescription.lineHeightRatio * fontDescription.fontSize < fontDescription.bounds.y)
              {
                switch (fontDescription.boundMode)
                {
                default:
                case mFD_BM_Stop:
                case mFD_BM_BreakLineOnBound:
                  break;

                case mFD_BM_StopWithTrippleDot:
                  mFontDescription description = fontDescription;
                  description.hasBounds = false;

                  mERROR_CHECK(mFontRenderer_Draw(fontRenderer, description, "...", colour));
                  break;
                }

                fontRenderer->stoppedAtStartY = true;
                break; // Stop Rendering, because we've hit the border.
              }
            }

            fontRenderer->position.x = fontRenderer->resetPosition.x;
            fontRenderer->position.y -= fontDescription.lineHeightRatio * fontDescription.fontSize;
            size = 0;

            if (layout.alignment == mTL_A_Centered)
              fontRenderer->position.x = fontRenderer->resetPosition.x - (phraseSize.x / fontDescription.scale + spaceGlyphInfo.advanceX) * .5f;
            else if (layout.alignment == mTL_A_Right)
              fontRenderer->position.x = fontRenderer->resetPosition.x - (phraseSize.x / fontDescription.scale + spaceGlyphInfo.advanceX);

            // Render.
            for (const auto &_item : fontRenderer->phraseRenderInfo->Iterate())
            {
              if (_item.commandType == mFR_PRGI_CT_SwitchTextureAtlas)
              {
                mERROR_CHECK(mFontRenderer_DrawEnqueuedText_Internal(fontRenderer));
                mERROR_CHECK(mQueue_PushBack(fontRenderer->currentFrameTextureAtlasses, std::move(fontRenderer->textureAtlas)));
                fontRenderer->textureAtlas = _item.nextTextureAtlas;
              }
              else if (_item.commandType == mFR_PRGI_CT_AdvanceOnly)
              {
                fontRenderer->position.x += _item.advanceX;
              }
              else
              {
                fontRenderer->position.x += _item.kerning;

                const float_t x0 = (fontRenderer->position.x + _item.offsetX) * fontDescription.scale;
                const float_t y0 = (fontRenderer->position.y + _item.offsetY) * fontDescription.scale;
                const float_t x1 = (x0 + _item.sizeX * fontDescription.scale);
                const float_t y1 = (y0 - _item.sizeY * fontDescription.scale);

                mRectangle2D<float_t> glyphBounds(x0, y1, x1 - x0, y0 - y1); // Flipped because it's in OpenGL space.
                fontRenderer->renderedArea.GrowToContain(glyphBounds);

                mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, x0 * fontRenderer->right + y0 * fontRenderer->up + fontRenderer->spacialOrigin));
                mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, mVec2f(_item.s0, _item.t0)));
                mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, (mVec4f)colour));

                mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, x0 * fontRenderer->right + y1 * fontRenderer->up + fontRenderer->spacialOrigin));
                mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, mVec2f(_item.s0, _item.t1)));
                mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, (mVec4f)colour));

                mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, x1 * fontRenderer->right + y1 * fontRenderer->up + fontRenderer->spacialOrigin));
                mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, mVec2f(_item.s1, _item.t1)));
                mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, (mVec4f)colour));

                mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, x1 * fontRenderer->right + y0 * fontRenderer->up + fontRenderer->spacialOrigin));
                mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, mVec2f(_item.s1, _item.t0)));
                mERROR_CHECK(mBinaryChunk_WriteData(fontRenderer->enqueuedText, (mVec4f)colour));

                fontRenderer->position.x += _item.advanceX;
              }
            }

            if (isNewLine)
              justWrapped = true;

            charsAdded = true;
          }

          if (!isLast)
          {
            if (fontDescription.hasBounds)
            {
              if (fontRenderer->position.y - fontDescription.lineHeightRatio * fontDescription.fontSize < fontDescription.bounds.y)
              {
                switch (fontDescription.boundMode)
                {
                default:
                case mFD_BM_Stop:
                case mFD_BM_BreakLineOnBound:
                  break;

                case mFD_BM_StopWithTrippleDot:
                  mFontDescription description = fontDescription;
                  description.hasBounds = false;
                  fontRenderer->stoppedAtStartY = true;

                  mERROR_CHECK(mFontRenderer_Draw(fontRenderer, description, "...", colour));
                  break;
                }

                fontRenderer->stoppedAtStartX = true;
                break; // Stop Rendering, because we've hit the border.
              }
            }

            fontRenderer->position.x = fontRenderer->resetPosition.x;
            fontRenderer->position.y -= fontDescription.lineHeightRatio * fontDescription.fontSize;
            size = 0;
          }
        }
        
        if (!charsAdded)
        {
          for (const auto &_item : fontRenderer->phraseRenderInfo->Iterate())
            mERROR_CHECK(mQueue_PushBack(fontRenderer->entirePhraseRenderInfo, _item));

          size += hideableSize + phraseSize.x;
          hideableSize = 0;

          // Add current character.
          {
            const float_t sizeFactor = _char.codePoint == tab ? 2.f : 1.f;
            const float_t addedSize = sizeFactor * spaceGlyphInfo.advanceX * fontDescription.glyphSpacingRatio;

            hideableSize = addedSize;

            mFontRenderer_PhraseRenderGlyphInfo info;
            info.commandType = mFR_PRGI_CT_AdvanceOnly;
            info.advanceX = addedSize;

            mERROR_CHECK(mQueue_PushBack(fontRenderer->entirePhraseRenderInfo, std::move(info)));
          }
        }
      }

      startText = _char.character + _char.characterSize;
    }
    else
    {
      justWrapped = false;
    }

    lastWasEndPhrase = endPhrase;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mFontRenderer_GetRenderedSize, mPtr<mFontRenderer> &fontRenderer, const mFontDescription &fontDescription, const mString &string, OUT mVec2f *pRenderedSize)
{
  mFUNCTION_SETUP();

  mERROR_IF(fontRenderer == nullptr || pRenderedSize == nullptr, mR_ArgumentNull);

  *pRenderedSize = mVec2f(0);

  mFontDescription_Internal *pFontDescription = nullptr;
  size_t fontDescriptionIndex = 0;

  // Find Matching font description.
  {
    const uint32_t fontSizeIndex = mFontDescription_Internal::GetFontSizeIndex(fontDescription.fontSize);

    for (auto &_fontDescription : fontRenderer->textureAtlas->fonts->Iterate())
    {
      if (_fontDescription.Matches(fontDescription, fontSizeIndex))
      {
        pFontDescription = &_fontDescription;
        break;
      }

      ++fontDescriptionIndex;
    }
  }

  mFontRenderer_FontInfo *pFontInfo = nullptr;
  size_t fontInfoIndex = 0;

  if (pFontDescription == nullptr)
  {
    for (auto &_fontInfo : fontRenderer->fontInfo->Iterate())
    {
      if (_fontInfo == fontDescription.fontFileName)
      {
        pFontInfo = &_fontInfo;
        break;
      }

      ++fontInfoIndex;
    }

    if (pFontInfo == nullptr)
    {
      bool added;
      size_t index;
      mERROR_CHECK(mFontRenderer_AddFont_Internal(fontRenderer, fontRenderer->textureAtlas, fontDescription, &index, &added));
      mASSERT_DEBUG(added, "Font should've already been found earlier otherwise.");

      mERROR_CHECK(mQueue_PointerAt(fontRenderer->textureAtlas->fonts, index, &pFontDescription));
      mERROR_CHECK(mQueue_PointerAt(fontRenderer->fontInfo, pFontDescription->fontInfoIndex, &pFontInfo));
    }
  }
  else
  {
    mERROR_CHECK(mQueue_PointerAt(fontRenderer->fontInfo, pFontDescription->fontInfoIndex, &pFontInfo));

    fontInfoIndex = pFontDescription->fontInfoIndex;
  }

  mVec2f resetPosition(0);
  mVec2f position(0);
  mRectangle2D<float_t> renderedArea(mVec2f(0), mVec2f(0));

  mchar_t previousChar = 0;

  for (auto &&_char : string)
  {
    if (*_char.character == '\n')
    {
      previousChar = _char.codePoint;
      position.x = resetPosition.x;
      position.y -= fontDescription.lineHeightRatio * fontDescription.fontSize;
    }
    else
    {
      bool retried = false;
      mFontRenderer_GlyphInfo *pGlyphInfo = nullptr;
      bool needsAtlasFlush = false;

    retry:
      mERROR_CHECK(mFontRenderer_GetGlyphInfo_Internal(fontRenderer, &pFontInfo, fontInfoIndex, fontDescription, &pFontDescription, &fontDescriptionIndex, _char.codePoint, _char.character, _char.characterSize, !fontDescription.ignoreBackupFonts, &pGlyphInfo, &needsAtlasFlush));
      pGlyphInfo->glyphUses++;
      pFontInfo->totalUses++;

      if (needsAtlasFlush)
      {
        if (retried)
          mRETURN_RESULT(mR_Failure);

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

            mERROR_CHECK(mFontTextureAtlas_ClearGlyphs_Internal(fontRenderer->textureAtlas));
          }
          else
          {
            // If no texture atlas available: create a new one.
            mERROR_CHECK(mFontTextureAtlas_Create(&fontRenderer->textureAtlas, fontRenderer->pAllocator, fontRenderer->textureAtlas->pTextureAtlas->width, fontRenderer->textureAtlas->pTextureAtlas->height));
          }

          pFontDescription = nullptr;
        }

        retried = true;
        goto retry;
      }

      float_t kerning = 0.0f;
      
      if (previousChar != 0)
      {
        for (auto &_kerningInfo : pGlyphInfo->kerningInfo->Iterate())
        {
          if (_kerningInfo.key == previousChar)
          {
            _kerningInfo.value.uses++;
            kerning = _kerningInfo.value.value;
            break;
          }
        }
      }
      
      previousChar = _char.codePoint;
      
      position.x += kerning;
      
      const float_t x0 = (fontRenderer->position.x + pGlyphInfo->approxOffsetX * fontDescription.fontSize) * fontDescription.scale;
      const float_t y0 = (fontRenderer->position.y + pGlyphInfo->approxOffsetY * fontDescription.fontSize) * fontDescription.scale;
      const float_t x1 = (x0 + pGlyphInfo->sizeX * fontDescription.scale);
      const float_t y1 = (y0 - pGlyphInfo->sizeY * fontDescription.scale);
      
      mRectangle2D<float_t> glyphBounds(x0, y1, x1 - x0, y0 - y1); // Flipped because it's in OpenGL space.
      
      renderedArea.GrowToContain(glyphBounds);
      position.x += pGlyphInfo->advanceX * fontDescription.glyphSpacingRatio;
    }
  }

  *pRenderedSize = renderedArea.size;

  mRETURN_SUCCESS();
}

mFUNCTION(mFontRenderer_GetRenderedWidth, mPtr<mFontRenderer> &fontRenderer, const mFontDescription &fontDescription, const mString &string, OUT float_t *pRenderedWidth)
{
  mFUNCTION_SETUP();

  mERROR_IF(pRenderedWidth == nullptr, mR_ArgumentNull);

  mVec2f size;
  mERROR_CHECK(mFontRenderer_GetRenderedSize(fontRenderer, fontDescription, string, &size));

  *pRenderedWidth = size.x;

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

static mFUNCTION(mFontRenderer_Destroy_Internal, mFontRenderer *pFontRenderer)
{
  mFUNCTION_SETUP();

  mERROR_IF(pFontRenderer == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mQueue_Destroy(&pFontRenderer->backupFonts));
  mERROR_CHECK(mQueue_Destroy(&pFontRenderer->fontInfo));
  mERROR_CHECK(mQueue_Destroy(&pFontRenderer->phraseRenderInfo));
  mERROR_CHECK(mQueue_Destroy(&pFontRenderer->entirePhraseRenderInfo));

  mERROR_CHECK(mQueue_Destroy(&pFontRenderer->currentFrameTextureAtlasses));
  mERROR_CHECK(mQueue_Destroy(&pFontRenderer->availableTextureAtlasses));

  mERROR_CHECK(mSharedPointer_Destroy(&pFontRenderer->textureAtlas));

  mERROR_CHECK(mBinaryChunk_Destroy(&pFontRenderer->enqueuedText));

  mERROR_CHECK(mIndexedRenderDataBuffer_Destroy(&pFontRenderer->indexDataBuffer));
  mERROR_CHECK(mAllocator_FreePtr(pFontRenderer->pAllocator, &pFontRenderer->pIndexBuffer));
  mERROR_CHECK(mString_Destroy(&pFontRenderer->phraseString));

  mRETURN_SUCCESS();
}

static mFUNCTION(mFontRenderer_UpdateFontAtlas_Internal, mPtr<mFontRenderer> &fontRenderer)
{
  mFUNCTION_SETUP();

  bool requiresUpload = true;
  mERROR_CHECK(mFontTextureAtlas_RequiresUpload(fontRenderer->textureAtlas, &requiresUpload));

#if defined(mRENDERER_OPENGL)

  glBindTexture(GL_TEXTURE_2D, fontRenderer->textureAtlas->pTextureAtlas->id);

  if (requiresUpload || fontRenderer->addedGlyph)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, (GLsizei)fontRenderer->textureAtlas->pTextureAtlas->width, (GLsizei)fontRenderer->textureAtlas->pTextureAtlas->height, 0, GL_RED, GL_UNSIGNED_BYTE, fontRenderer->textureAtlas->pTextureAtlas->pData);

  fontRenderer->addedGlyph = false;
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

static mFUNCTION(mFontRenderer_DrawEnqueuedText_Internal, mPtr<mFontRenderer> &fontRenderer)
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
  mERROR_CHECK(mShader_SetUniform(fontRenderer->indexDataBuffer.shader, mFontRenderer_MatrixUniform, fontRenderer->matrix));
  mERROR_CHECK(mShader_SetUniform(fontRenderer->indexDataBuffer.shader, mFontRenderer_TextureUniform, fontRenderer->textureAtlas->texture));

  if (fontRenderer->signedDistanceFieldRendering)
    mERROR_CHECK(mShader_SetUniform(fontRenderer->indexDataBuffer.shader, mFontRenderer_SubPixelUniform, 1.f / (fontRenderer->textureAtlas->texture.resolutionF * 3.f)));

  mERROR_CHECK(mIndexedRenderDataBuffer_SetRenderCount(fontRenderer->indexDataBuffer, length));
  mERROR_CHECK(mIndexedRenderDataBuffer_Draw(fontRenderer->indexDataBuffer));

  mERROR_CHECK(mBinaryChunk_ResetWrite(fontRenderer->enqueuedText));

#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

static mFUNCTION(mFontRenderer_AddPersistentFontInfo_Internal, mPtr<mFontRenderer> &fontRenderer, const mFontDescription &fontDescription, OUT size_t *pIndex)
{
  mFUNCTION_SETUP();

  *pIndex = 0;

  for (const auto &_fontRenderable : fontRenderer->fontInfo->Iterate())
  {
    if (_fontRenderable == fontDescription.fontFileName)
      mRETURN_SUCCESS();

    ++(*pIndex);
  }

  // *pIndex is already set to the new index

  mFontRenderer_FontInfo persistentFontInfo;
  persistentFontInfo.fontFileName = fontDescription.fontFileName;
  mERROR_CHECK(mQueue_Create(&persistentFontInfo.glyphInfo, fontRenderer->pAllocator));
  mERROR_CHECK(mQueue_Create(&persistentFontInfo.missingGlyphs, fontRenderer->pAllocator));

  mERROR_CHECK(mQueue_PushBack(fontRenderer->fontInfo, std::move(persistentFontInfo)));

  mRETURN_SUCCESS();
}

static mFUNCTION(mFontRenderer_AddFont_Internal, mPtr<mFontRenderer> &fontRenderer, mPtr<mFontTextureAtlas> &fontTextureAtlas, const mFontDescription &fontDescription, OUT size_t *pIndex, OUT bool *pAdded)
{
  mFUNCTION_SETUP();

  mERROR_IF(fontRenderer == nullptr, mR_ArgumentNull);

  size_t fontCount = 0;
  mERROR_CHECK(mQueue_GetCount(fontTextureAtlas->fonts, &fontCount));
  
  *pIndex = 0;
  *pAdded = false;

  const uint32_t fontSizeIndex = mFontDescription_Internal::GetFontSizeIndex(fontDescription.fontSize);

  for (const mFontDescription_Internal &_fontDescription : fontTextureAtlas->fonts->Iterate())
  {
    if (_fontDescription.Matches(fontDescription, fontSizeIndex))
      mRETURN_SUCCESS();

    ++(*pIndex);
  }

  // *pIndex is already set to the new index

  mFontDescription_Internal internalFontDescription;
  mZeroMemory(&internalFontDescription, 1);

  mERROR_CHECK(mInplaceString_Create(&internalFontDescription.fontFileName, fontDescription.fontFileName));
  internalFontDescription.fontSizeIndex = fontSizeIndex;
  internalFontDescription.fontSizePx = fontDescription.fontSize;
  
  // Profiled.
  {
    mPROFILE_SCOPED("mFontRenderer_AddFont_Internal (texture_font_new_from_file)");

    mERROR_CHECK(texture_font_new_from_file(fontTextureAtlas->pTextureAtlas, &internalFontDescription.pTextureFont, fontRenderer->pAllocator, internalFontDescription.fontSizePx, internalFontDescription.fontFileName.c_str(), MODE_MANUAL_CLOSE, mFontRenderer_pFontLibrary));
  }

  if (fontRenderer->signedDistanceFieldRendering)
    internalFontDescription.pTextureFont->rendermode = RENDER_SIGNED_DISTANCE_FIELD;

  mERROR_CHECK(mFontRenderer_AddPersistentFontInfo_Internal(fontRenderer, fontDescription, &internalFontDescription.fontInfoIndex));

  mERROR_IF(internalFontDescription.pTextureFont == nullptr, mR_InternalError);
  mERROR_CHECK(mQueue_PushBack(fontTextureAtlas->fonts, std::move(internalFontDescription)));

  *pAdded = true;

  mRETURN_SUCCESS();
}

static mFUNCTION(mFontRenderer_FontDescriptionAlreadyContainsChar, mPtr<mFontRenderer> & /* fontRenderer */, mFontDescription_Internal *pFontDescription, const float_t /* originalFontSize */, const mchar_t codePoint, OUT bool *pContainsChar)
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
    for (const mchar_t c : pFontDescription->additionalGlyphRanges->Iterate())
    {
      if (c == codePoint)
        mRETURN_SUCCESS();
    }
  }

  *pContainsChar = false;

  mRETURN_SUCCESS();
}

mFUNCTION(mFontRenderer_FontDescriptionKnownToNotContainChar, mPtr<mFontRenderer> & /* fontRenderer */, const mFontRenderer_FontInfo &fontInfo, const mchar_t codePoint, OUT bool *pKnownToNotContainChar)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mQueue_Contains(fontInfo.missingGlyphs, codePoint, pKnownToNotContainChar));

  mRETURN_SUCCESS();
}

static mFUNCTION(mFontRenderer_AddGlyphToFontInfo, mPtr<mFontRenderer> &fontRenderer, const mFontDescription_Internal *pFontDescription, mFontRenderer_FontInfo *pFontInfo, texture_glyph_t *pGlyph, const mchar_t codePoint, OUT mFontRenderer_GlyphInfo **ppGlyphInfo = nullptr)
{
  mFUNCTION_SETUP();

  for (auto &_info : pFontInfo->glyphInfo->Iterate())
  {
    if (_info.key == codePoint)
    {
      if (ppGlyphInfo != nullptr)
        *ppGlyphInfo = &_info.value;

      mRETURN_SUCCESS();
    }
  }

  const float_t invFontSize = 1.f / pFontDescription->fontSizePx;

  mKeyValuePair<mchar_t, mFontRenderer_GlyphInfo> glyphInfo;
  mZeroMemory(&glyphInfo);

  glyphInfo.key = codePoint;
  glyphInfo.value.advanceX = pGlyph->advance_x * invFontSize;
  glyphInfo.value.advanceY = pGlyph->advance_y * invFontSize;
  glyphInfo.value.approxOffsetX = pGlyph->offset_x * invFontSize;
  glyphInfo.value.approxOffsetY = pGlyph->offset_y * invFontSize;
  glyphInfo.value.sizeX = pGlyph->width * invFontSize;
  glyphInfo.value.sizeY = pGlyph->height * invFontSize;
  mERROR_CHECK(mQueue_Create(&glyphInfo.value.kerningInfo, fontRenderer->pAllocator));

  for (size_t i = 0; i < pGlyph->kerning.count; i++)
  {
    const float_t *pKerning = pGlyph->kerning[i];

    if (pKerning == nullptr)
      continue;

    for (size_t j = 0; j < 0x100; j++)
    {
      if (pKerning[j] == 0) // The kerning generation has been modified to never generate zero values, so we can *actually* rely on this now.
        continue;

      mKeyValuePair<mchar_t, mFontRenderer_KerningInfo> kerningInfo;
      kerningInfo.key = (mchar_t)((i << 8) | j);
      kerningInfo.value.value = pKerning[j] * invFontSize;
      kerningInfo.value.uses = 0;

      mERROR_CHECK(mQueue_PushBack(glyphInfo.value.kerningInfo, std::move(kerningInfo)));
    }
  }

  mERROR_CHECK(mQueue_PushBack(pFontInfo->glyphInfo, std::move(glyphInfo)));

  if (ppGlyphInfo != nullptr)
  {
    mKeyValuePair<mchar_t, mFontRenderer_GlyphInfo> *pGlyphInfo;
    mERROR_CHECK(mQueue_PointerAt(pFontInfo->glyphInfo, pFontInfo->glyphInfo->count - 1, &pGlyphInfo));

    *ppGlyphInfo = &pGlyphInfo->value;
  }

  mRETURN_SUCCESS();
}

static mFUNCTION(mFontRenderer_LoadGlyph_Internal, mPtr<mFontRenderer> &fontRenderer, mPtr<mFontTextureAtlas> &fontTextureAtlas, mFontDescription_Internal *pFontDescription, const size_t queueFontIndex, const float_t originalFontSize, const mchar_t codePoint, OUT texture_font_t **ppFont, OUT texture_glyph_t **ppGlyph, const bool attemptToLoadFromBackupFont /* = true */)
{
  mFUNCTION_SETUP();

  *ppGlyph = nullptr;
  *ppFont = nullptr;

  bool containsChar = false;
  mERROR_CHECK(mFontRenderer_FontDescriptionAlreadyContainsChar(fontRenderer, pFontDescription, originalFontSize, codePoint, &containsChar));

  // Is this codepoint already loaded?
  if (containsChar)
  {
    texture_glyph_t *pGlyph = texture_font_find_glyph_gi(pFontDescription->pTextureFont, codePoint);

    mASSERT_DEBUG(pGlyph != nullptr, "Invalid Description Information Loaded.");

    *ppFont = pFontDescription->pTextureFont;
    *ppGlyph = pGlyph;

    mRETURN_SUCCESS();
  }

  mFontRenderer_FontInfo *pFontInfo = nullptr;
  mERROR_CHECK(mQueue_PointerAt(fontRenderer->fontInfo, pFontDescription->fontInfoIndex, &pFontInfo));

  bool knownNotToContainChar = false;
  mERROR_CHECK(mFontRenderer_FontDescriptionKnownToNotContainChar(fontRenderer, *pFontInfo, codePoint, &knownNotToContainChar));

  const bool checkBackupFonts = attemptToLoadFromBackupFont;

  // Should we add the backup fonts?
  if (knownNotToContainChar && checkBackupFonts)
  {
    bool anyFontsAdded = false;

    for (const auto &_backupFont : fontRenderer->backupFonts->Iterate())
    {
      if (_backupFont != pFontDescription->fontFileName)
      {
        mFontDescription fontDescription;
        fontDescription.fontSize = originalFontSize;
        mERROR_CHECK(mInplaceString_Create(&fontDescription.fontFileName, _backupFont));

        bool added = false;
        size_t fontIndex;
        mERROR_CHECK(mFontRenderer_AddFont_Internal(fontRenderer, fontTextureAtlas, fontDescription, &fontIndex, &added));

        anyFontsAdded |= added;

        mFontDescription_Internal *pBackupFontDescription = nullptr;
        mERROR_CHECK(mQueue_PointerAt(fontTextureAtlas->fonts, fontIndex, &pBackupFontDescription));

        bool backupFontContainsChar = false;
        mERROR_CHECK(mFontRenderer_FontDescriptionAlreadyContainsChar(fontRenderer, pBackupFontDescription, originalFontSize, codePoint, &backupFontContainsChar));

        if (backupFontContainsChar)
        {
          texture_glyph_t *pGlyph = texture_font_find_glyph_gi(pBackupFontDescription->pTextureFont, codePoint);

          mASSERT_DEBUG(pGlyph != nullptr, "Invalid Description Information Loaded.");

          *ppFont = pBackupFontDescription->pTextureFont;
          *ppGlyph = pGlyph;

          mRETURN_SUCCESS();
        }
      }
    }

    if (anyFontsAdded)
      mERROR_CHECK(mQueue_PointerAt(fontTextureAtlas->fonts, queueFontIndex, &pFontDescription));
  }

  mPROFILE_SCOPED("mFontRenderer_LoadGlyph");

  // Try to load the glyph of this codepoint.
  if (!knownNotToContainChar)
  {
    mPROFILE_SCOPED("mFontRenderer_LoadGlyph (Primary)");

    // Load glyph.
    mResult loadResult = texture_font_load_glyph(pFontDescription->pTextureFont, codePoint, checkBackupFonts);

    if (mFAILED(loadResult))
    {
      if (loadResult == mR_ResourceAlreadyExists)
        loadResult = mR_Success;
      else if (loadResult == mR_ResourceBusy)
        mSILENT_RETURN_RESULT(loadResult);
    }

    if (mSUCCEEDED(loadResult))
    {
      texture_glyph_t *pGlyph = texture_font_find_glyph_gi(pFontDescription->pTextureFont, codePoint);

      if (pGlyph->codepoint == 0)
        loadResult = mR_Failure;
      else
        mERROR_CHECK(mFontRenderer_AddGlyphToFontInfo(fontRenderer, pFontDescription, pFontInfo, pGlyph, codePoint));

      *ppFont = pFontDescription->pTextureFont;
      *ppGlyph = pGlyph;
    }

    if (mFAILED(loadResult))
    {
      if (pFontInfo == nullptr)
        mERROR_CHECK(mQueue_PointerAt(fontRenderer->fontInfo, pFontDescription->fontInfoIndex, &pFontInfo));

      mERROR_CHECK(mQueue_PushBack(pFontInfo->missingGlyphs, codePoint));
    }
    else
    {
      mERROR_CHECK(mFontRenderer_AddCharToFontDescription(fontRenderer, pFontDescription, codePoint));

      mRETURN_SUCCESS();
    }
  }

  // Do any of the backup fonts contain this codepoint?
  if (checkBackupFonts)
  {
    bool anyFontsAdded = false;

    for (const auto &_backupFont : fontRenderer->backupFonts->Iterate())
    {
      // We've already checked earlier: This font does not contain the glyph.
      if (_backupFont != pFontDescription->fontFileName)
      {
        mPROFILE_SCOPED("mFontRenderer_LoadGlyph (Backup)");

        mFontDescription fontDescription;
        fontDescription.fontSize = originalFontSize;
        mERROR_CHECK(mInplaceString_Create(&fontDescription.fontFileName, _backupFont));

        bool added;
        size_t fontIndex;
        mERROR_CHECK(mFontRenderer_AddFont_Internal(fontRenderer, fontTextureAtlas, fontDescription, &fontIndex, &added));

        anyFontsAdded |= added;

        mFontDescription_Internal *pBackupFontDescription = nullptr;
        mERROR_CHECK(mQueue_PointerAt(fontTextureAtlas->fonts, fontIndex, &pBackupFontDescription));

        mFontRenderer_FontInfo *pBackupFontInfo = nullptr;
        mERROR_CHECK(mQueue_PointerAt(fontRenderer->fontInfo, pBackupFontDescription->fontInfoIndex, &pBackupFontInfo));

        // Load glyph.
        mResult loadResult = texture_font_load_glyph(pBackupFontDescription->pTextureFont, codePoint, true);

        if (mFAILED(loadResult))
        {
          if (loadResult == mR_ResourceAlreadyExists)
            loadResult = mR_Success;
          else if (loadResult == mR_ResourceBusy)
            mSILENT_RETURN_RESULT(loadResult);
        }

        if (mSUCCEEDED(loadResult))
        {
          texture_glyph_t *pGlyph = texture_font_find_glyph_gi(pBackupFontDescription->pTextureFont, codePoint);

          if (pGlyph->codepoint == 0)
            loadResult = mR_Failure;
          else
            mERROR_CHECK(mFontRenderer_AddGlyphToFontInfo(fontRenderer, pBackupFontDescription, pBackupFontInfo, pGlyph, codePoint));

          *ppFont = pBackupFontDescription->pTextureFont;
          *ppGlyph = pGlyph;
        }

        if (mSUCCEEDED(loadResult))
        {
          mERROR_CHECK(mFontRenderer_AddCharToFontDescription(fontRenderer, pBackupFontDescription, codePoint));

          mRETURN_SUCCESS();
        }
        else
        {
          mERROR_CHECK(mQueue_PushBack(pBackupFontInfo->missingGlyphs, codePoint));
        }
      }
    }

    if (anyFontsAdded)
      mERROR_CHECK(mQueue_PointerAt(fontTextureAtlas->fonts, queueFontIndex, &pFontDescription));

    // oh well, none of the backup fonts contained this codepoint.
    mResult loadResult = texture_font_load_glyph(pFontDescription->pTextureFont, codePoint, false);

    if (mFAILED(loadResult))
    {
      if (loadResult == mR_ResourceAlreadyExists)
        loadResult = mR_Success;
      else if (loadResult == mR_ResourceBusy)
        mSILENT_RETURN_RESULT(loadResult);
    }

    if (mSUCCEEDED(loadResult))
    {
      texture_glyph_t *pGlyph = nullptr;
      const mResult result = texture_font_get_glyph_cp(pFontDescription->pTextureFont, codePoint, &pGlyph, false);

      if (mFAILED(result))
        if (result != mR_ResourceNotFound)
          mRETURN_RESULT(result);

      if (pGlyph != nullptr)
      {
        mERROR_CHECK(mFontRenderer_AddCharNotFoundToFontDescription(fontRenderer, pFontDescription, codePoint, pGlyph));

        *ppFont = pFontDescription->pTextureFont;
        *ppGlyph = pGlyph;

        mRETURN_SUCCESS();
      }
    }
  }

  mRETURN_RESULT(mR_ResourceNotFound);
}

static mFUNCTION(mFontRenderer_GetGlyphInfo_Internal, mPtr<mFontRenderer> &fontRenderer, mFontRenderer_FontInfo **ppFontInfo, const size_t fontInfoIndex, const mFontDescription &fontDescription, mFontDescription_Internal **ppFontDescription, size_t *pQueueFontIndex, const mchar_t codePoint, const char *character, const size_t bytes, const bool checkBackupFonts, OUT mFontRenderer_GlyphInfo **ppGlyphInfo, OUT bool *pRequireAtlasFlush, OUT OPTIONAL bool *pCharNotFound_NullptrIfNotInternalCall /* = nullptr */)
{
  mFUNCTION_SETUP();

  *pRequireAtlasFlush = false;
  *ppGlyphInfo = nullptr;

  for (auto &_char : (*ppFontInfo)->glyphInfo->Iterate())
  {
    if (_char.key == codePoint)
    {
      *ppGlyphInfo = &_char.value;
      mRETURN_SUCCESS();
    }
  }

  mPROFILE_SCOPED("mFontRenderer_GetGlyphInfo (Load Glyph)");

  bool knownNotToContainChar = false;
  mERROR_CHECK(mFontRenderer_FontDescriptionKnownToNotContainChar(fontRenderer, **ppFontInfo, codePoint, &knownNotToContainChar));

  // Does any backup font contain the char already?
  if (knownNotToContainChar && checkBackupFonts)
  {
    for (const auto &_backupFont : fontRenderer->backupFonts->Iterate())
    {
      for (auto &_fontInfo : fontRenderer->fontInfo->Iterate())
      {
        if (_fontInfo.fontFileName == _backupFont)
        {
          for (auto &_char : _fontInfo.glyphInfo->Iterate())
          {
            if (_char.key == codePoint)
            {
              *ppGlyphInfo = &_char.value;
              mRETURN_SUCCESS();
            }
          }

          break;
        }
      }
    }
  }

  int32_t glyphLoaded = 0;

  if (!knownNotToContainChar)
  {
    if (*ppFontDescription == nullptr)
    {
      size_t index;
      bool added;
      mERROR_CHECK(mFontRenderer_AddFont_Internal(fontRenderer, fontRenderer->textureAtlas, fontDescription, &index, &added));
      *pQueueFontIndex = index;
      mASSERT_DEBUG(added, "Should've been found already otherwise!");
      mERROR_CHECK(mQueue_PointerAt(fontRenderer->textureAtlas->fonts, index, ppFontDescription));
    }

    // Load glyph.
    glyphLoaded = texture_font_load_glyph((*ppFontDescription)->pTextureFont, codePoint, checkBackupFonts);

    if (glyphLoaded != 0)
    {
      texture_glyph_t *pGlyph = texture_font_find_glyph_gi((*ppFontDescription)->pTextureFont, codePoint);

      if (pGlyph->codepoint == 0)
        glyphLoaded = 0;
      else
        mERROR_CHECK(mFontRenderer_AddGlyphToFontInfo(fontRenderer, *ppFontDescription, *ppFontInfo, pGlyph, codePoint, ppGlyphInfo));
    }

    if (glyphLoaded == 0)
    {
      mERROR_CHECK(mQueue_PushBack((*ppFontInfo)->missingGlyphs, codePoint));
    }
    else
    {
      mERROR_CHECK(mFontRenderer_AddCharToFontDescription(fontRenderer, *ppFontDescription, codePoint));
      mRETURN_SUCCESS();
    }
  }

  if (checkBackupFonts)
  {
    size_t noFontsAddedSize = fontRenderer->textureAtlas->fonts->count;

    for (const auto &_backupFont : fontRenderer->backupFonts->Iterate())
    {
      if (_backupFont != fontDescription.fontFileName)
      {
        size_t backupFontInfoIndex = 0;
        mFontRenderer_FontInfo *pBackupFontInfo = nullptr;
        mFontDescription_Internal *pBackupFontDescription = nullptr;
        
        for (auto &_fontInfo : fontRenderer->fontInfo->Iterate())
        {
          if (_fontInfo.fontFileName == _backupFont)
          {
            pBackupFontInfo = &_fontInfo;
            break;
          }

          ++backupFontInfoIndex;
        }

        if (pBackupFontInfo == nullptr)
        {
          mFontDescription backupFontDescription = fontDescription;
          mERROR_CHECK(mInplaceString_Create(&backupFontDescription.fontFileName, _backupFont));

          size_t index;
          bool added;
          mERROR_CHECK(mFontRenderer_AddFont_Internal(fontRenderer, fontRenderer->textureAtlas, fontDescription, &index, &added));
          mERROR_CHECK(mQueue_PointerAt(fontRenderer->textureAtlas->fonts, index, &pBackupFontDescription));
          mASSERT_DEBUG(added, "Should've been found already otherwise!");
          mERROR_CHECK(mQueue_PointerAt(fontRenderer->fontInfo, pBackupFontDescription->fontInfoIndex, &pBackupFontInfo));

          //if (added) // _MUST_ have just been added.
          {
            noFontsAddedSize = fontRenderer->textureAtlas->fonts->count;
            mERROR_CHECK(mQueue_PointerAt(fontRenderer->fontInfo, fontInfoIndex, ppFontInfo));
            mERROR_CHECK(mQueue_PointerAt(fontRenderer->textureAtlas->fonts, *pQueueFontIndex, ppFontDescription));
          }
        }

        bool charNotFound = true;
        size_t backupFontDescriptionIndex;

        mERROR_CHECK(mFontRenderer_GetGlyphInfo_Internal(fontRenderer, &pBackupFontInfo, backupFontInfoIndex, fontDescription, &pBackupFontDescription, &backupFontDescriptionIndex, codePoint, character, bytes, false, ppGlyphInfo, pRequireAtlasFlush, &charNotFound));

        if (!charNotFound)
        {
          if (noFontsAddedSize != fontRenderer->textureAtlas->fonts->count)
          {
            mERROR_CHECK(mQueue_PointerAt(fontRenderer->fontInfo, fontInfoIndex, ppFontInfo));
            mERROR_CHECK(mQueue_PointerAt(fontRenderer->textureAtlas->fonts, *pQueueFontIndex, ppFontDescription));
          }

          mRETURN_SUCCESS();
        }
      }
    }

    if (noFontsAddedSize != fontRenderer->textureAtlas->fonts->count)
    {
      mERROR_CHECK(mQueue_PointerAt(fontRenderer->fontInfo, fontInfoIndex, ppFontInfo));
      mERROR_CHECK(mQueue_PointerAt(fontRenderer->textureAtlas->fonts, *pQueueFontIndex, ppFontDescription));
    }
  }

  // Forcefully load a char (may just be [?] or similar).
  if (pCharNotFound_NullptrIfNotInternalCall == nullptr)
  {
    glyphLoaded = texture_font_load_glyph((*ppFontDescription)->pTextureFont, codePoint, false);

    if (glyphLoaded != 0)
    {
      texture_glyph_t *pGlyph = nullptr;
      const mResult result = texture_font_get_glyph_cp((*ppFontDescription)->pTextureFont, codePoint, &pGlyph, false);

      if (mFAILED(result))
        if (result != mR_ResourceNotFound)
          mRETURN_RESULT(result);

      if (pGlyph != nullptr)
        mERROR_CHECK(mFontRenderer_AddCharNotFoundToFontDescription(fontRenderer, *ppFontDescription, codePoint, pGlyph));
    }

    *pRequireAtlasFlush = (0 == glyphLoaded);
  }
  else
  {
    *pCharNotFound_NullptrIfNotInternalCall = true;
  }

  mRETURN_SUCCESS();
}

static mFUNCTION(mFontRenderer_AddCharToFontDescription, mPtr<mFontRenderer> &fontRenderer, mFontDescription_Internal *pFontDescription, const mchar_t codePoint)
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
      mERROR_CHECK(mQueue_Create(&pFontDescription->additionalGlyphRanges, fontRenderer->pAllocator));

    mERROR_CHECK(mQueue_PushBack(pFontDescription->additionalGlyphRanges, codePoint));
  }

  mRETURN_SUCCESS();
}

static mFUNCTION(mFontRenderer_AddCharNotFoundToFontDescription, mPtr<mFontRenderer> &fontRenderer, mFontDescription_Internal *pFontDescription, const mchar_t codePoint, texture_glyph_t *pGlyph)
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
      mERROR_CHECK(mQueue_Create(&pFontDescription->additionalGlyphRanges, fontRenderer->pAllocator));

    mERROR_CHECK(mQueue_PushBack(pFontDescription->additionalGlyphRanges, codePoint));
  }

  mFontRenderer_FontInfo *pFontInfo = nullptr;
  mERROR_CHECK(mQueue_PointerAt(fontRenderer->fontInfo, pFontDescription->fontInfoIndex, &pFontInfo));

  bool contained = false;

  for (auto &_char : pFontInfo->glyphInfo->Iterate())
  {
    if (_char.key == (mchar_t)0)
    {
      ++_char.value.glyphUses;
      contained = true;
      break;
    }
  }

  if (!contained)
    mERROR_CHECK(mFontRenderer_AddGlyphToFontInfo(fontRenderer, pFontDescription, pFontInfo, pGlyph, codePoint));

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

mFUNCTION(mFontRenderer_TryRenderPhrase, mPtr<mFontRenderer> &fontRenderer, const mString &phrase, const mFontDescription &fontDescription, OUT mPtr<mQueue<mFontRenderer_PhraseRenderGlyphInfo>> &renderInfo, OUT mVec2f *pSize)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mQueue_Clear(renderInfo));

  mFontDescription_Internal *pFontDescription = nullptr;
  size_t fontDescriptionQueueIndex;
  bool added;
  mERROR_CHECK(mFontRenderer_AddFont_Internal(fontRenderer, fontRenderer->phraseRenderingTextureAtlas, fontDescription, &fontDescriptionQueueIndex, &added));
  mERROR_CHECK(mQueue_PointerAt(fontRenderer->textureAtlas->fonts, fontDescriptionQueueIndex, &pFontDescription));

  const float_t sizeFactor = fontDescription.fontSize / pFontDescription->fontSizePx;

  mFontRenderer_FontInfo *pFontInfo;
  mERROR_CHECK(mQueue_PointerAt(fontRenderer->fontInfo, pFontDescription->fontInfoIndex, &pFontInfo));

  mchar_t previousChar = 0;
  mVec2f position = fontRenderer->position;
  mRectangle2D<float_t> bounds(position, mVec2f(0));

  for (const auto &_char : phrase)
  {
    texture_font_t *pFont= nullptr;
    texture_glyph_t *pGlyph = nullptr;

    // Load Glyph.
    {
      const mResult result = mFontRenderer_LoadGlyph_Internal(fontRenderer, fontRenderer->phraseRenderingTextureAtlas, pFontDescription, fontDescriptionQueueIndex, fontDescription.fontSize, _char.codePoint, &pFont, &pGlyph, !fontDescription.ignoreBackupFonts);

      if (mFAILED(result))
      {
        mSILENT_ERROR_IF(result != mR_ResourceBusy, result);

        // Switch Font Atlas.
        {
          mERROR_IF(fontRenderer->startedRenderable, mR_ArgumentOutOfBounds);

          mFontRenderer_PhraseRenderGlyphInfo glyphInfo;
          glyphInfo.commandType = mFR_PRGI_CT_SwitchTextureAtlas;

          if (fontRenderer->availableTextureAtlasses->count > 0)
          {
            mERROR_CHECK(mQueue_PopFront(fontRenderer->availableTextureAtlasses, &glyphInfo.nextTextureAtlas));

            mERROR_CHECK(mFontTextureAtlas_ClearGlyphs_Internal(glyphInfo.nextTextureAtlas));

            fontRenderer->phraseRenderingTextureAtlas = glyphInfo.nextTextureAtlas;
          }
          else
          {
            mERROR_CHECK(mFontTextureAtlas_Create(&glyphInfo.nextTextureAtlas, fontRenderer->pAllocator, fontRenderer->textureAtlas->pTextureAtlas->width, fontRenderer->textureAtlas->pTextureAtlas->height));
            fontRenderer->phraseRenderingTextureAtlas = glyphInfo.nextTextureAtlas;
          }

          mERROR_CHECK(mQueue_PushBack(renderInfo, std::move(glyphInfo)));

          mERROR_CHECK(mFontRenderer_AddFont_Internal(fontRenderer, fontRenderer->phraseRenderingTextureAtlas, fontDescription, &fontDescriptionQueueIndex, &added));
          mERROR_CHECK(mQueue_PointerAt(fontRenderer->phraseRenderingTextureAtlas->fonts, fontDescriptionQueueIndex, &pFontDescription));
        }

        mERROR_CHECK(mFontRenderer_LoadGlyph_Internal(fontRenderer, fontRenderer->phraseRenderingTextureAtlas, pFontDescription, fontDescriptionQueueIndex, fontDescription.fontSize, _char.codePoint, &pFont, &pGlyph, !fontDescription.ignoreBackupFonts));
      }
    }

    mASSERT_DEBUG(pGlyph != nullptr, "Glyph should've been retrieved in `mFontRenderer_LoadGlyph_Internal`.");

    mERROR_CHECK(mQueue_PointerAt(fontRenderer->phraseRenderingTextureAtlas->fonts, fontDescriptionQueueIndex, &pFontDescription)); // Might have moved due to the fontQueue growing if we loaded a character from a newly loaded backup font.
    mERROR_CHECK(mQueue_PointerAt(fontRenderer->fontInfo, pFontDescription->fontInfoIndex, &pFontInfo));

    mFontRenderer_PhraseRenderGlyphInfo glyphInfo;
    glyphInfo.commandType = mFR_PRGI_CT_GlyphInfo;

    float_t kerning = 0.0f;

    if (previousChar != 0)
      /* Let's ignore errors for now. */ mSILENCE_ERROR(texture_glyph_generate_kerning_on_demand(pFont, pGlyph, previousChar, &kerning));

    previousChar = _char.codePoint;
    glyphInfo.kerning = kerning * sizeFactor;

    position.x += glyphInfo.kerning;

    glyphInfo.offsetX = pGlyph->offset_x * sizeFactor;
    glyphInfo.offsetY = pGlyph->offset_y * sizeFactor;
    glyphInfo.sizeX = pGlyph->width * sizeFactor;
    glyphInfo.sizeY = pGlyph->height * sizeFactor;
    glyphInfo.advanceX = pGlyph->advance_x * sizeFactor * fontDescription.glyphSpacingRatio;
    glyphInfo.advanceY = pGlyph->advance_y * sizeFactor * fontDescription.glyphSpacingRatio;

    const mVec2f halfPixel = mVec2f(.75f) / fontRenderer->textureAtlas->texture.resolutionF; // not entirely sure why, but without this being nudged to 0.75 we're getting weird off-by-half-a-texel rendering errors.

    glyphInfo.s0 = pGlyph->s0;
    glyphInfo.t0 = pGlyph->t0;
    glyphInfo.s1 = pGlyph->s1 - halfPixel.x;
    glyphInfo.t1 = pGlyph->t1 - halfPixel.y;

    const float_t x0 = (position.x + glyphInfo.offsetX) * fontDescription.scale;
    const float_t y0 = (position.y + glyphInfo.offsetY) * fontDescription.scale;
    const float_t x1 = (x0 + glyphInfo.sizeX * fontDescription.scale);
    const float_t y1 = (y0 - glyphInfo.sizeY * fontDescription.scale);

    mRectangle2D<float_t> glyphBounds(x0, y1, x1 - x0, y0 - y1); // Flipped because it's in OpenGL space.
    bounds.GrowToContain(glyphBounds);

    position.x += glyphInfo.advanceX;

    mERROR_CHECK(mQueue_PushBack(renderInfo, glyphInfo));
  }

  *pSize = bounds.size;

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mFontTextureAtlas_Create, OUT mPtr<mFontTextureAtlas> *pAtlas, mAllocator *pAllocator, const size_t width, const size_t height)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAtlas == nullptr, mR_ArgumentNull);

  mDEFER_CALL_ON_ERROR(pAtlas, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Allocate(pAtlas, pAllocator, (std::function<void(mFontTextureAtlas *)>)[](mFontTextureAtlas *pData) { mFontTextureAtlas_Destroy(pData); }, 1));

  mERROR_CHECK(mQueue_Create(&(*pAtlas)->fonts, pAllocator));
  
  mERROR_CHECK(texture_atlas_new(&(*pAtlas)->pTextureAtlas, pAllocator, width, height, 1));

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
    mERROR_CHECK(mQueue_Destroy(&pAtlas->fonts));

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

  if (atlas->pTextureAtlas->isModified)
  {
    atlas->pTextureAtlas->isModified = false;
    *pNeedsUpload = true;
  }
  else
  {
    *pNeedsUpload = false;
  }

  mRETURN_SUCCESS();
}

static mFUNCTION(mFontTextureAtlas_ClearGlyphs_Internal, mPtr<mFontTextureAtlas> &textureAtlas)
{
  mFUNCTION_SETUP();

  texture_atlas_clear(textureAtlas->pTextureAtlas);

  size_t fontCount = 0;
  mERROR_CHECK(mQueue_GetCount(textureAtlas->fonts, &fontCount));

  for (size_t i = 0; i < fontCount; i++)
  {
    mFontDescription_Internal *pFontDesc = nullptr;
    mERROR_CHECK(mQueue_PointerAt(textureAtlas->fonts, i, &pFontDesc));

    pFontDesc->glyphRange0 = 0;
    pFontDesc->glyphRange1 = 0;

    if (pFontDesc->additionalGlyphRanges != nullptr)
      mERROR_CHECK(mQueue_Clear(pFontDesc->additionalGlyphRanges));

    for (texture_glyph_t **ppGlyph : pFontDesc->pTextureFont->glyphs)
      if (ppGlyph != nullptr)
        for (size_t j = 0; j < 0x100; j++)
          texture_glyph_delete(&ppGlyph[j]);

    mERROR_CHECK(mList_Clear(pFontDesc->pTextureFont->glyphs));
  }

  textureAtlas->lastSize = 0;

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

static mFUNCTION(mFontRenderable_Destroy_Internal, IN mFontRenderable *pFontRenderable)
{
  mFUNCTION_SETUP();

  mERROR_IF(pFontRenderable == nullptr, mR_ArgumentNull);
  
  mGL_DEBUG_ERROR_CHECK();

  mERROR_CHECK(mSharedPointer_Destroy(&pFontRenderable->textureAtlas));
  mERROR_CHECK(mIndexedRenderDataBuffer_Destroy(&pFontRenderable->indexDataBuffer));

  mGL_DEBUG_ERROR_CHECK();

  mRETURN_SUCCESS();
}
