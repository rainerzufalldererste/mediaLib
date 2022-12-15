#include "mFontRenderer_Internal.h"

#include "mQueue.h"
#include "mProfiler.h"

#pragma warning(push, 0)
#include "ft2build.h"
#include FT_FREETYPE_H
#include FT_STROKER_H
#include FT_LCD_FILTER_H
#include FT_SIZES_H
#pragma warning(pop)

// Copyright 2011-2016 Nicolas P. Rougier
// Copyright 2013-2016 Marcel Metz
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
//  1. Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
// 
//  2. Redistributions in binary form must reproduce the above copyright
//     notice, this list of conditions and the following disclaimer in the
//     documentation and/or other materials provided with the distribution.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 
// The views and conclusions contained in the software and documentation are
// those of the authors and should not be interpreted as representing official
// policies, either expressed or implied, of the freetype-gl project.


#ifdef __APPLE__
# include <machine/endian.h>
# define __BIG_ENDIAN __ORDER_BIG_ENDIAN__
# define __LITTLE_ENDIAN __ORDER_LITTLE_ENDIAN__
# define __BYTE_ORDER __BYTE_ORDER__
#elif defined(_WIN32) || defined(_WIN64)
# define __LITTLE_ENDIAN 1234
# define __BIG_ENDIAN 4321
# define __BYTE_ORDER __LITTLE_ENDIAN
#else
# include <endian.h>
#endif

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "GkGehISV1s80k0PRvzpOkjyhTZAt8v/sjATmyU9ElumAkCf0uAEDKrYOP3zhz55h7RdBav5p60rou882"
#endif

//////////////////////////////////////////////////////////////////////////

mFUNCTION(texture_atlas_get_region, texture_atlas_t *pSelf, const size_t width, const size_t height, OUT mRectangle2D<int32_t> *pRegion);
mFUNCTION(texture_atlas_set_region, texture_atlas_t *pSelf, const size_t x, const size_t y, const size_t width, const size_t height, const uint8_t *data, const size_t stride);

mFUNCTION(texture_font_load_face, texture_font_t *pSelf, float_t size, texture_font_library_t *pLibrary);
mFUNCTION(texture_font_generate_kerning_range, texture_font_t *pSelf, FT_Face *pFace, const uint32_t codepointA, const uint32_t codepointB);

//////////////////////////////////////////////////////////////////////////

constexpr size_t _HRES = 64;
constexpr float_t _HRESf = (float_t)_HRES;
constexpr size_t _DPI = 72;

inline static void _FreetypeError(FT_Error error)
{
  mPRINT_ERROR("FreeType Error 0x", mFX()((int32_t)error), ".");
}

inline static float_t _F26Dot6ToFloat(FT_F26Dot6 value)
{
  return ((float_t)value) / 64.0f;
}

inline static FT_F26Dot6 _FloatToF26Dot6(float_t value)
{
  return (FT_F26Dot6)(value * 64.0);
}

inline static uint32_t _Utf8ToUtf32(const char *character)
{
  if (!character)
  {
    return (uint32_t)-1;
  }

  if ((character[0] & 0x80) == 0x0)
  {
    return character[0];
  }

  if ((character[0] & 0xE0) == 0xC0)
  {
    return ((character[0] & 0x3F) << 6) | (character[1] & 0x3F);
  }

  if ((character[0] & 0xF0) == 0xE0)
  {
    return ((character[0] & 0x1F) << (6 + 6)) | ((character[1] & 0x3F) << 6) | (character[2] & 0x3F);
  }

  if ((character[0] & 0xF8) == 0xF0)
  {
    return ((character[0] & 0x0F) << (6 + 6 + 6)) | ((character[1] & 0x3F) << (6 + 6)) | ((character[2] & 0x3F) << 6) | (character[3] & 0x3F);
  }

  if ((character[0] & 0xFC) == 0xF8)
  {
    return ((character[0] & 0x07) << (6 + 6 + 6 + 6)) | ((character[1] & 0x3F) << (6 + 6 + 6)) | ((character[2] & 0x3F) << (6 + 6)) | ((character[3] & 0x3F) << 6) | (character[4] & 0x3F);
  }

  return 0xFFFD; // invalid character
}

inline static uint32_t _SwapBytes(const uint32_t in)
{
  return ((in >> 24) & 0xFF) | ((in >> 8) & 0xFF00) | ((in & 0xFF00) << 8) | ((in & 0xFF) << 24);
}

inline static uint32_t _RollBits(const uint32_t in, const uint32_t x)
{
  return (in >> (32 - x)) | (in << x);
}

//////////////////////////////////////////////////////////////////////////

void texture_font_close(texture_font_t *pSelf, font_mode_t face_mode, font_mode_t library_mode)
{
  if (pSelf != nullptr && pSelf->face != nullptr && pSelf->mode <= face_mode)
  {
    FT_Done_Face(pSelf->face);
    pSelf->face = nullptr;
  }
  else
  {
    return; // never close the library when the face stays open.
  }

  if (pSelf != nullptr && pSelf->pLibrary != nullptr && pSelf->pLibrary->library != nullptr && pSelf->pLibrary->mode <= library_mode)
  {
    FT_Done_FreeType(pSelf->pLibrary->library);
    pSelf->pLibrary->library = nullptr;
  }
}

mFUNCTION(texture_glyph_new, OUT texture_glyph_t **ppGlyph, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppGlyph == nullptr, mR_ArgumentNull);

  texture_glyph_t *pSelf = nullptr;
  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &pSelf, 1));
  mDEFER_ON_ERROR(mAllocator_FreePtr(pAllocator, &pSelf));

  pSelf->pAllocator = pAllocator;

  pSelf->codepoint = (uint32_t)-1;
  pSelf->width = 0;
  pSelf->height = 0;

  // Attributes that can have different images for the same codepoint.
  pSelf->rendermode = RENDER_NORMAL;
  pSelf->outline_thickness = 0.0;
  pSelf->glyphmode = GLYPH_END;

  pSelf->offset_x = 0;
  pSelf->offset_y = 0;
  pSelf->advance_x = 0.0;
  pSelf->advance_y = 0.0;
  pSelf->s0 = 0.0;
  pSelf->t0 = 0.0;
  pSelf->s1 = 0.0;
  pSelf->t1 = 0.0;

  mERROR_CHECK(mList_Create(&pSelf->kerning, pAllocator));
  
  *ppGlyph = pSelf;

  mRETURN_SUCCESS();
}

void texture_glyph_delete(texture_glyph_t **ppGlyph)
{
  if (ppGlyph == nullptr || *ppGlyph == nullptr)
    return;

  for (float_t *pKerningRange : (*ppGlyph)->kerning)
    mFreePtr(&pKerningRange);

  mList_Destroy(&(*ppGlyph)->kerning);

  mAllocator_FreePtr((*ppGlyph)->pAllocator, ppGlyph);
}

float_t texture_glyph_get_kerning(const texture_glyph_t *pSelf, const uint32_t codepoint)
{
  if (pSelf == nullptr)
    return 0;

  const uint32_t i = codepoint >> 8;
  const uint32_t j = codepoint & 0xFF;

  if (codepoint == -1)
    return 0;

  if (pSelf->kerning.count <= i)
    return 0;

  const float_t *pKerningIndex = pSelf->kerning[i];

  if (!pKerningIndex)
    return 0;
  else
    return pKerningIndex[j];
}

mFUNCTION(texture_glyph_generate_kerning_on_demand, texture_font_t *pFont, texture_glyph_t *pGlyph, const uint32_t codepoint, OUT float_t *pKerning)
{
  mFUNCTION_SETUP();

  mERROR_IF(pFont == nullptr || pGlyph == nullptr || pKerning == nullptr, mR_ArgumentNull);

  const uint32_t i = codepoint >> 8;
  const uint32_t j = codepoint & 0xFF;

  if (codepoint == -1)
  {
    *pKerning = 0;
    mRETURN_SUCCESS();
  }

  if (pGlyph->kerning.count <= i || pGlyph->kerning[i] == nullptr || pGlyph->kerning[i][j] == 0)
  {
    mDEFER_CALL_3(texture_font_close, pFont, MODE_AUTO_CLOSE, MODE_AUTO_CLOSE);
    mERROR_CHECK(texture_font_load_face(pFont, pFont->size, pFont->pLibrary));

    mERROR_CHECK(texture_font_generate_kerning_range(pFont, &pFont->face, pGlyph->codepoint, codepoint));

    mERROR_IF(pGlyph->kerning.count <= i || pGlyph->kerning[i] == nullptr || pGlyph->kerning[i][j] == 0, mR_InternalError);
  }

  *pKerning = pGlyph->kerning[i][j];

  mRETURN_SUCCESS();
}

mFUNCTION(texture_library_new, OUT texture_font_library_t **pLibrary, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pLibrary == nullptr, mR_ArgumentNull);

  texture_font_library_t *pSelf;
  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &pSelf, 1));

  pSelf->pAllocator = pAllocator;
  pSelf->mode = MODE_MANUAL_CLOSE;

  *pLibrary = pSelf;

  mRETURN_SUCCESS();
}

mFUNCTION(texture_font_set_size, texture_font_t *pSelf, float_t size)
{
  mFUNCTION_SETUP();

  FT_Matrix matrix = 
  {
    (int32_t)((1.0 / _HRES) * 0x10000L),
    (int32_t)((0.0) * 0x10000L),
    (int32_t)((0.0) * 0x10000L),
    (int32_t)((1.0) * 0x10000L)
  };

  if (FT_HAS_FIXED_SIZES(pSelf->face))
  {
    // Select best size.
    mERROR_IF(pSelf->face->num_fixed_sizes == 0, mR_ResourceIncompatible);

    int32_t best_match = 0;
    float_t diff = 1e20;

    for (int32_t i = 0; i < pSelf->face->num_fixed_sizes; ++i)
    {
      const float_t new_size = _F26Dot6ToFloat(pSelf->face->available_sizes[i].size);
      const float_t ndiff = size > new_size ? size / new_size : new_size / size;
      
      mLOG("FontRenderer: Candiate: Size [", i, "] = ", new_size, " ", pSelf->face->available_sizes[i].width, " x ", pSelf->face->available_sizes[i].height, "\n");
      
      if (ndiff < diff)
      {
        best_match = i;
        diff = ndiff;
      }
    }

    mLOG("FontRenderer: Selected: Size [", best_match, "] for ", size, "\n");
    
    const FT_Error error = FT_Select_Size(pSelf->face, best_match);
    mERROR_IF(error, mR_InternalError);
    
    pSelf->scale = pSelf->size / _F26Dot6ToFloat(pSelf->face->available_sizes[best_match].size);
  }
  else
  {
    // Set char size.
    const FT_Error error = FT_Set_Char_Size(pSelf->face, _FloatToF26Dot6(size), 0, _DPI * _HRES, _DPI);
    mERROR_IF(error, mR_InternalError);
  }

  // Set transform matrix.
  FT_Set_Transform(pSelf->face, &matrix, nullptr);

  mRETURN_SUCCESS();
}

mFUNCTION(texture_font_load_face, texture_font_t *pSelf, float_t size, texture_font_library_t *pLibrary)
{
  mFUNCTION_SETUP();

  mERROR_IF(pLibrary == nullptr, mR_ArgumentNull);

  if (pSelf->pLibrary == nullptr)
    pSelf->pLibrary = pLibrary;

  if (pSelf->pLibrary->library == nullptr)
  {
    const FT_Error error = FT_Init_FreeType(&pSelf->pLibrary->library);
    mERROR_IF(error, mR_InternalError);
  }

  if (pSelf->face == nullptr)
  {
    switch (pSelf->location)
    {
    case TEXTURE_FONT_FILE:
    {
      const FT_Error error = FT_New_Face(pSelf->pLibrary->library, pSelf->filename.c_str(), 0, &pSelf->face);

      if (error)
      {
        _FreetypeError(error);
        texture_font_close(pSelf, MODE_ALWAYS_OPEN, MODE_ALWAYS_OPEN);
        mRETURN_RESULT(mR_ResourceNotFound);
      }

      break;
    }

    case TEXTURE_FONT_MEMORY:
    {
      const FT_Error error = FT_New_Memory_Face(pSelf->pLibrary->library, pSelf->memory.pBase, (FT_Long)pSelf->memory.size, 0, &pSelf->face);

      if (error)
      {
        _FreetypeError(error);
        texture_font_close(pSelf, MODE_ALWAYS_OPEN, MODE_ALWAYS_OPEN);
        mRETURN_RESULT(mR_ResourceInvalid);
      }

      break;
    }
    }

    // Select charmap.
    {
      mDEFER_ON_ERROR(texture_font_close(pSelf, MODE_ALWAYS_OPEN, MODE_FREE_CLOSE));

      FT_Error error = FT_Select_Charmap(pSelf->face, FT_ENCODING_UNICODE);
      mERROR_IF(error, mR_InternalError);

      error = FT_New_Size(pSelf->face, &pSelf->ft_size);
      mERROR_IF(error, mR_InternalError);

      error = FT_Activate_Size(pSelf->ft_size);
      mERROR_IF(error, mR_InternalError);

      mERROR_CHECK(texture_font_set_size(pSelf, size));
    }
  }

  mRETURN_SUCCESS();
}

void texture_font_init_size(texture_font_t *pSelf)
{
  FT_Size_Metrics metrics;

  pSelf->underline_position = pSelf->face->underline_position / (float_t)(_HRESf * _HRESf) * pSelf->size;
  pSelf->underline_position = mRound(pSelf->underline_position);

  if (pSelf->underline_position > -2)
    pSelf->underline_position = -2.0;

  pSelf->underline_thickness = pSelf->face->underline_thickness / (float_t)(_HRESf * _HRESf) * pSelf->size;
  pSelf->underline_thickness = mRound(pSelf->underline_thickness);

  if (pSelf->underline_thickness < 1)
    pSelf->underline_thickness = 1.0;

  metrics = pSelf->face->size->metrics;

  pSelf->ascender = (float_t)(metrics.ascender >> 6);
  pSelf->descender = (float_t)(metrics.descender >> 6);
  pSelf->height = (float_t)(metrics.height >> 6);
  pSelf->linegap = pSelf->height - pSelf->ascender + pSelf->descender;
}

static mFUNCTION(texture_font_init, texture_font_t *pSelf, texture_font_library_t *pLibrary)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSelf->pAtlas == nullptr, mR_InternalError);
  mERROR_IF(pSelf->size <= 0, mR_InternalError);
  mERROR_IF((pSelf->location == TEXTURE_FONT_FILE && pSelf->filename.bytes <= 1) || (pSelf->location == TEXTURE_FONT_MEMORY && pSelf->memory.pBase == nullptr && pSelf->memory.size == 0), mR_InternalError);
  
  pSelf->height = 0;
  pSelf->ascender = 0;
  pSelf->descender = 0;
  pSelf->linegap = 0;
  pSelf->rendermode = RENDER_NORMAL;
  pSelf->outline_thickness = 0.0;
  pSelf->useAutoHinting = true;
  pSelf->useKerning = true;
  pSelf->useLcdFiltering = true;
  pSelf->scaleTextureCoords = true;
  pSelf->scale = 1.0;

  // FT_LCD_FILTER_LIGHT   is (0x00, 0x55, 0x56, 0x55, 0x00)
  // FT_LCD_FILTER_DEFAULT is (0x10, 0x40, 0x70, 0x40, 0x10)
  pSelf->lcd_weights[0] = 0x10;
  pSelf->lcd_weights[1] = 0x40;
  pSelf->lcd_weights[2] = 0x70;
  pSelf->lcd_weights[3] = 0x40;
  pSelf->lcd_weights[4] = 0x10;

  mERROR_CHECK(texture_font_load_face(pSelf, pSelf->size, pLibrary));

  texture_font_init_size(pSelf);

  mERROR_CHECK(texture_font_set_size(pSelf, pSelf->size));

  // null is a special glyph.
  {
    texture_glyph_t *pGlyph = nullptr;

    mERROR_CHECK(texture_font_get_glyph(pSelf, nullptr, &pGlyph, false));
    
    mERROR_IF(pGlyph == nullptr, mR_InternalError);
  }

  mRETURN_SUCCESS();
}

mFUNCTION(texture_font_new_from_file, texture_atlas_t *pAtlas, OUT texture_font_t **ppFont, IN mAllocator *pAllocator, const float_t pt_size, const char *filename, const font_mode_t fontMode, texture_font_library_t *pLibrary)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAtlas == nullptr || filename == nullptr || ppFont == nullptr || pLibrary == nullptr, mR_ArgumentNull);

  texture_font_t *pSelf = nullptr;
  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &pSelf, 1));
  mDEFER_ON_ERROR(mAllocator_FreePtr(pAllocator, &pSelf));

  pSelf->pAllocator = pAllocator;

  mERROR_CHECK(mList_Create(&pSelf->glyphs, pAllocator));
  mDEFER_CALL_ON_ERROR(&pSelf->glyphs, mList_Destroy);

  pSelf->pAtlas = pAtlas;
  pSelf->size = pt_size;

  pSelf->location = TEXTURE_FONT_FILE;

  mERROR_CHECK(mInplaceString_Create(&pSelf->filename, filename));
  
  pSelf->mode = fontMode;

  mDEFER_CALL_ON_ERROR(pSelf, texture_font_delete);
  mERROR_CHECK(texture_font_init(pSelf, pLibrary));

  *ppFont = pSelf;
  
  mRETURN_SUCCESS();
}

void texture_font_delete(texture_font_t *pSelf)
{
  if (pSelf == nullptr)
    return;

  // This ends up crashing with use after free in `FT_Done_Size` because `face->driver` isn't `nullptr` but has already been freed somewhere else.
  //if (pSelf->ft_size)
  //{
  //  const FT_Error error = FT_Done_Size(pSelf->ft_size);
  //
  //  if (error)
  //    _FreetypeError(error);
  //
  //  pSelf->ft_size = nullptr;
  //}

  texture_font_close(pSelf, MODE_ALWAYS_OPEN, MODE_FREE_CLOSE);

  for (size_t i = 0; i < pSelf->glyphs.count; i++)
  {
    texture_glyph_t **ppGlyphs = pSelf->glyphs[i];

    if (ppGlyphs == nullptr)
      continue;

    for (size_t j = 0; j < 0x100; j++)
      texture_glyph_delete(&ppGlyphs[j]);

    mFreePtr(&ppGlyphs);
  }

  mList_Destroy(&pSelf->glyphs);

  mAllocator_Free(pSelf->pAllocator, pSelf);
}

mFUNCTION(texture_font_index_glyph, texture_font_t *pSelf, texture_glyph_t *pGlyph, uint32_t codepoint, OUT OPTIONAL bool *pFreeGlyph = nullptr)
{
  mFUNCTION_SETUP();

  const uint32_t i = codepoint >> 8;
  const uint32_t j = codepoint & 0xFF;

  if (pSelf->glyphs.count <= i)
    mERROR_CHECK(mList_ResizeWith(pSelf->glyphs, (size_t)i + 1, (texture_glyph_t **)nullptr));

  texture_glyph_t ***pppGlyphIndex1 = &pSelf->glyphs[i];

  if (*pppGlyphIndex1 == nullptr)
    mERROR_CHECK(mAllocZero(pppGlyphIndex1, 0x100));

  texture_glyph_t *pGlyphInsert = (*pppGlyphIndex1)[j];

  if (pGlyphInsert != nullptr)
  {
    size_t index = 0;

    while (pGlyphInsert[index].glyphmode != GLYPH_END)
      index++;
    
    pGlyphInsert[index].glyphmode = GLYPH_CONT;

    mERROR_CHECK(mRealloc(&pGlyphInsert, (index + 2)));
    (*pppGlyphIndex1)[j] = pGlyphInsert;

    mMemcpy(pGlyphInsert + (index + 1), pGlyph, 1);
    
    if (pFreeGlyph != nullptr)
      *pFreeGlyph = true;
  }
  else
  {
    (*pppGlyphIndex1)[j] = pGlyph;

    if (pFreeGlyph != nullptr)
      *pFreeGlyph = false;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(texture_glyph_clone, const texture_glyph_t *pSelf, texture_glyph_t **ppClone)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSelf == nullptr || ppClone == nullptr, mR_ArgumentNull);

  texture_glyph_t *pNewGlyph = nullptr;
  mERROR_CHECK(mAllocator_AllocateZero(pSelf->pAllocator, &pNewGlyph, 1));
  mDEFER_ON_ERROR(mAllocator_FreePtr(pSelf->pAllocator, &pNewGlyph));

  pNewGlyph->pAllocator = pSelf->pAllocator;

  pNewGlyph->codepoint = pSelf->codepoint;
  pNewGlyph->width = pSelf->width;
  pNewGlyph->height = pSelf->height;
  pNewGlyph->offset_x = pSelf->offset_x;
  pNewGlyph->offset_y = pSelf->offset_y;
  pNewGlyph->advance_x = pSelf->advance_x;
  pNewGlyph->advance_y = pSelf->advance_y;
  pNewGlyph->s0 = pSelf->s0;
  pNewGlyph->t0 = pSelf->t0;
  pNewGlyph->s1 = pSelf->s1;
  pNewGlyph->t1 = pSelf->t1;
  pNewGlyph->rendermode = pSelf->rendermode;
  pNewGlyph->outline_thickness = pSelf->outline_thickness;
  pNewGlyph->glyphmode = pSelf->glyphmode;
  
  mERROR_CHECK(mList_Create(&pNewGlyph->kerning, pNewGlyph->pAllocator));
  mDEFER_CALL_ON_ERROR(&pNewGlyph->kerning, mList_Destroy);

  mERROR_CHECK(mList_ResizeWith(pNewGlyph->kerning, pSelf->kerning.count, (float_t *)nullptr));

  mDEFER_ON_ERROR(
    for (size_t i = 0; i < pNewGlyph->kerning.count; i++)
      if (pNewGlyph->kerning[i] != nullptr)
        mFreePtr(&pNewGlyph->kerning[i]);
  );

  for (size_t i = 0; i < pSelf->kerning.count; i++)
  {
    const float_t *pSource = pSelf->kerning[i];

    if (pSource != nullptr)
    {
      float_t **ppTarget = &pNewGlyph->kerning[i];

      mERROR_CHECK(mAllocZero(ppTarget, 0x100));
      mMemcpy(*ppTarget, pSource, 0x100);
    }
  }

  *ppClone = pNewGlyph;

  mRETURN_SUCCESS();
}

mFUNCTION(texture_font_index_kerning, texture_glyph_t *pSelf, const uint32_t codepoint, const float_t kerning)
{
  mFUNCTION_SETUP();

  const uint32_t i = codepoint >> 8;
  const uint32_t j = codepoint & 0xFF;

  if (pSelf->kerning.count <= i)
    mERROR_CHECK(mList_ResizeWith(pSelf->kerning, i + 1, (float_t *)nullptr));

  float_t **ppKerningIndex = &pSelf->kerning[i];

  if (*ppKerningIndex == nullptr)
    mERROR_CHECK(mAllocZero(ppKerningIndex, 0x100));

  float_t kerningValue = kerning;

  if (kerningValue == 0)
    kerningValue = FLT_EPSILON;
  
  (*ppKerningIndex)[j] = kerningValue;

  mRETURN_SUCCESS();
}

bool texture_font_has_kerning_info(texture_glyph_t *pSelf, const uint32_t codepoint)
{
  const uint32_t i = codepoint >> 8;
  const uint32_t j = codepoint & 0xFF;

  if (pSelf->kerning.count <= i)
    return false;

  float_t **ppKerningIndex = &pSelf->kerning[i];

  if (*ppKerningIndex == nullptr)
    return false;

  if ((*ppKerningIndex)[j] == 0)
    return false;

  return true;
}

double_t edgedf(double_t gx, double_t gy, const double_t a)
{
  if ((gx == 0) || (gy == 0)) 
  {
    return 0.5 - a; // Either A) gu or gv are zero, or B) both. Linear approximation is A) correct or B) a fair guess.
  }
  else
  {
    double_t glength = mSqrt(gx * gx + gy * gy);

    if (glength > 0)
    {
      gx = gx / glength;
      gy = gy / glength;
    }

    // Everything is symmetric wrt sign and transposition, so move to first octant (pGx>=0, pGy>=0, pGx>=pGy) to avoid handling all possible edge directions.
    gx = mAbs(gx);
    gy = mAbs(gy);

    if (gx < gy)
    {
      const double_t temp = gx;
      gx = gy;
      gy = temp;
    }

    const double_t a1 = 0.5 * gy / gx;

    if (a < a1) // 0 <= a < a1
      return 0.5 * (gx + gy) - mSqrt(2.0 * gx * gy * a);
    else if (a < (1.0 - a1)) // a1 <= a <= 1-a1
      return (0.5 - a) * gx;
    else // 1-a1 < a <= 1
      return -0.5 * (gx + gy) + mSqrt(2.0 * gx * gy * (1.0 - a));
  }
}

double_t distaa3(const double_t *pImg, const double_t *pGxImg, const double_t *pGyImg, const int32_t w, const int32_t c, const int32_t xc, const int32_t yc, const int32_t xi, const int32_t yi)
{
  const int32_t closest = c - xc - yc * w; // Index to the edge pixel pointed to from c
  
  double_t a = pImg[closest]; // Grayscale value at the edge pixel.
  const double_t gx = pGxImg[closest]; // X gradient component at the edge pixel.
  const double_t gy = pGyImg[closest]; // Y gradient component at the edge pixel.

  if (a > 1.0)
    a = 1.0;

  if (a < 0.0)
    a = 0.0; // Clip grayscale values pOutside the range [0,1].

  if (a == 0.0)
    return 1000000.0; // Not an object pixel, return "very far" ("don't know yet").

  const double_t dx = (double_t)xi;
  const double_t dy = (double_t)yi;
  const double_t di = mSqrt(dx * dx + dy * dy); // Length of integer vector, like a traditional EDT.

  // Same metric as edtaa2, except at edges (where di=0).

  if (di == 0)
    return di + edgedf(gx, gy, a); // Use local gradient only at edges. Estimate based on local gradient only.
  else
    return di + edgedf(dx, dy, a); // Estimate gradient based on direction to edge (accurate for large di).
}

void edtaa3(double_t *pImg, double_t *pGx, double_t *pGy, int32_t w, int32_t h, int16_t *pDistX, int16_t *pDistY, double_t *dist)
{
  int32_t x, y, i, c;
  int32_t offset_u, offset_ur, offset_r, offset_rd, offset_d, offset_dl, offset_l, offset_lu;
  double_t olddist, newdist;
  int32_t cdistx, cdisty, newdistx, newdisty;
  int32_t changed;
  double_t epsilon = 1e-3;

  // Initialize index offsets for the current image width.
  offset_u = -w;
  offset_ur = -w + 1;
  offset_r = 1;
  offset_rd = w + 1;
  offset_d = w;
  offset_dl = w - 1;
  offset_l = -1;
  offset_lu = -w - 1;

  // Initialize the distance images.
  for (i = 0; i < w * h; i++)
  {
    pDistX[i] = 0; // At first, all pixels point to
    pDistY[i] = 0; // themselves as the closest known.

    if (pImg[i] <= 0.0)
      dist[i] = 1000000.0; // Big value, means "not set yet"
    else if (pImg[i] < 1.0)
      dist[i] = edgedf(pGx[i], pGy[i], pImg[i]); // Gradient-assisted estimate
    else
      dist[i] = 0.0; // Inside the object
  }

  /* Perform the transformation */
  do
  {
    changed = 0;

    /* Scan rows, except first row */
    for (y = 1; y < h; y++)
    {

      /* move index to leftmost pixel of current row */
      i = y * w;

      /* scan right, propagate distances from above & left */

      /* Leftmost pixel is special, has no left neighbors */
      olddist = dist[i];

      if (olddist > 0) // If non-zero distance or not set yet
      {
        c = i + offset_u; // Index of candidate for testing
        cdistx = pDistX[c];
        cdisty = pDistY[c];
        newdistx = cdistx;
        newdisty = cdisty + 1;
        newdist = distaa3(pImg, pGx, pGy, w, c, cdistx, cdisty, newdistx, newdisty);

        if (newdist < olddist - epsilon)
        {
          pDistX[i] = (int16_t)newdistx;
          pDistY[i] = (int16_t)newdisty;
          dist[i] = newdist;
          olddist = newdist;
          changed = 1;
        }

        c = i + offset_ur;
        cdistx = pDistX[c];
        cdisty = pDistY[c];
        newdistx = cdistx - 1;
        newdisty = cdisty + 1;
        newdist = distaa3(pImg, pGx, pGy, w, c, cdistx, cdisty, newdistx, newdisty);

        if (newdist < olddist - epsilon)
        {
          pDistX[i] = (int16_t)newdistx;
          pDistY[i] = (int16_t)newdisty;
          dist[i] = newdist;
          changed = 1;
        }
      }

      i++;

      /* Middle pixels have all neighbors */
      for (x = 1; x < w - 1; x++, i++)
      {
        olddist = dist[i];

        if (olddist <= 0)
          continue; // No need to update further

        c = i + offset_l;
        cdistx = pDistX[c];
        cdisty = pDistY[c];
        newdistx = cdistx + 1;
        newdisty = cdisty;
        newdist = distaa3(pImg, pGx, pGy, w, c, cdistx, cdisty, newdistx, newdisty);

        if (newdist < olddist - epsilon)
        {
          pDistX[i] = (int16_t)newdistx;
          pDistY[i] = (int16_t)newdisty;
          dist[i] = newdist;
          olddist = newdist;
          changed = 1;
        }

        c = i + offset_lu;
        cdistx = pDistX[c];
        cdisty = pDistY[c];
        newdistx = cdistx + 1;
        newdisty = cdisty + 1;
        newdist = distaa3(pImg, pGx, pGy, w, c, cdistx, cdisty, newdistx, newdisty);

        if (newdist < olddist - epsilon)
        {
          pDistX[i] = (int16_t)newdistx;
          pDistY[i] = (int16_t)newdisty;
          dist[i] = newdist;
          olddist = newdist;
          changed = 1;
        }

        c = i + offset_u;
        cdistx = pDistX[c];
        cdisty = pDistY[c];
        newdistx = cdistx;
        newdisty = cdisty + 1;
        newdist = distaa3(pImg, pGx, pGy, w, c, cdistx, cdisty, newdistx, newdisty);

        if (newdist < olddist - epsilon)
        {
          pDistX[i] = (int16_t)newdistx;
          pDistY[i] = (int16_t)newdisty;
          dist[i] = newdist;
          olddist = newdist;
          changed = 1;
        }

        c = i + offset_ur;
        cdistx = pDistX[c];
        cdisty = pDistY[c];
        newdistx = cdistx - 1;
        newdisty = cdisty + 1;
        newdist = distaa3(pImg, pGx, pGy, w, c, cdistx, cdisty, newdistx, newdisty);

        if (newdist < olddist - epsilon)
        {
          pDistX[i] = (int16_t)newdistx;
          pDistY[i] = (int16_t)newdisty;
          dist[i] = newdist;
          changed = 1;
        }
      }

      /* Rightmost pixel of row is special, has no right neighbors */
      olddist = dist[i];

      if (olddist > 0) // If not already zero distance
      {
        c = i + offset_l;
        cdistx = pDistX[c];
        cdisty = pDistY[c];
        newdistx = cdistx + 1;
        newdisty = cdisty;
        newdist = distaa3(pImg, pGx, pGy, w, c, cdistx, cdisty, newdistx, newdisty);

        if (newdist < olddist - epsilon)
        {
          pDistX[i] = (int16_t)newdistx;
          pDistY[i] = (int16_t)newdisty;
          dist[i] = newdist;
          olddist = newdist;
          changed = 1;
        }

        c = i + offset_lu;
        cdistx = pDistX[c];
        cdisty = pDistY[c];
        newdistx = cdistx + 1;
        newdisty = cdisty + 1;
        newdist = distaa3(pImg, pGx, pGy, w, c, cdistx, cdisty, newdistx, newdisty);

        if (newdist < olddist - epsilon)
        {
          pDistX[i] = (int16_t)newdistx;
          pDistY[i] = (int16_t)newdisty;
          dist[i] = newdist;
          olddist = newdist;
          changed = 1;
        }

        c = i + offset_u;
        cdistx = pDistX[c];
        cdisty = pDistY[c];
        newdistx = cdistx;
        newdisty = cdisty + 1;
        newdist = distaa3(pImg, pGx, pGy, w, c, cdistx, cdisty, newdistx, newdisty);

        if (newdist < olddist - epsilon)
        {
          pDistX[i] = (int16_t)newdistx;
          pDistY[i] = (int16_t)newdisty;
          dist[i] = newdist;
          changed = 1;
        }
      }

      /* Move index to second rightmost pixel of current row. */
      /* Rightmost pixel is skipped, it has no right neighbor. */
      i = y * w + w - 2;

      /* scan left, propagate distance from right */
      for (x = w - 2; x >= 0; x--, i--)
      {
        olddist = dist[i];

        if (olddist <= 0)
          continue; // Already zero distance

        c = i + offset_r;
        cdistx = pDistX[c];
        cdisty = pDistY[c];
        newdistx = cdistx - 1;
        newdisty = cdisty;
        newdist = distaa3(pImg, pGx, pGy, w, c, cdistx, cdisty, newdistx, newdisty);

        if (newdist < olddist - epsilon)
        {
          pDistX[i] = (int16_t)newdistx;
          pDistY[i] = (int16_t)newdisty;
          dist[i] = newdist;
          changed = 1;
        }
      }
    }

    /* Scan rows in reverse order, except last row */
    for (y = h - 2; y >= 0; y--)
    {
      /* move index to rightmost pixel of current row */
      i = y * w + w - 1;

      /* Scan left, propagate distances from below & right */

      /* Rightmost pixel is special, has no right neighbors */
      olddist = dist[i];
      if (olddist > 0) // If not already zero distance
      {
        c = i + offset_d;
        cdistx = pDistX[c];
        cdisty = pDistY[c];
        newdistx = cdistx;
        newdisty = cdisty - 1;
        newdist = distaa3(pImg, pGx, pGy, w, c, cdistx, cdisty, newdistx, newdisty);

        if (newdist < olddist - epsilon)
        {
          pDistX[i] = (int16_t)newdistx;
          pDistY[i] = (int16_t)newdisty;
          dist[i] = newdist;
          olddist = newdist;
          changed = 1;
        }

        c = i + offset_dl;
        cdistx = pDistX[c];
        cdisty = pDistY[c];
        newdistx = cdistx + 1;
        newdisty = cdisty - 1;
        newdist = distaa3(pImg, pGx, pGy, w, c, cdistx, cdisty, newdistx, newdisty);

        if (newdist < olddist - epsilon)
        {
          pDistX[i] = (int16_t)newdistx;
          pDistY[i] = (int16_t)newdisty;
          dist[i] = newdist;
          changed = 1;
        }
      }

      i--;

      /* Middle pixels have all neighbors */
      for (x = w - 2; x > 0; x--, i--)
      {
        olddist = dist[i];

        if (olddist <= 0)
          continue; // Already zero distance

        c = i + offset_r;
        cdistx = pDistX[c];
        cdisty = pDistY[c];
        newdistx = cdistx - 1;
        newdisty = cdisty;
        newdist = distaa3(pImg, pGx, pGy, w, c, cdistx, cdisty, newdistx, newdisty);

        if (newdist < olddist - epsilon)
        {
          pDistX[i] = (int16_t)newdistx;
          pDistY[i] = (int16_t)newdisty;
          dist[i] = newdist;
          olddist = newdist;
          changed = 1;
        }

        c = i + offset_rd;
        cdistx = pDistX[c];
        cdisty = pDistY[c];
        newdistx = cdistx - 1;
        newdisty = cdisty - 1;
        newdist = distaa3(pImg, pGx, pGy, w, c, cdistx, cdisty, newdistx, newdisty);

        if (newdist < olddist - epsilon)
        {
          pDistX[i] = (int16_t)newdistx;
          pDistY[i] = (int16_t)newdisty;
          dist[i] = newdist;
          olddist = newdist;
          changed = 1;
        }

        c = i + offset_d;
        cdistx = pDistX[c];
        cdisty = pDistY[c];
        newdistx = cdistx;
        newdisty = cdisty - 1;
        newdist = distaa3(pImg, pGx, pGy, w, c, cdistx, cdisty, newdistx, newdisty);

        if (newdist < olddist - epsilon)
        {
          pDistX[i] = (int16_t)newdistx;
          pDistY[i] = (int16_t)newdisty;
          dist[i] = newdist;
          olddist = newdist;
          changed = 1;
        }

        c = i + offset_dl;
        cdistx = pDistX[c];
        cdisty = pDistY[c];
        newdistx = cdistx + 1;
        newdisty = cdisty - 1;
        newdist = distaa3(pImg, pGx, pGy, w, c, cdistx, cdisty, newdistx, newdisty);

        if (newdist < olddist - epsilon)
        {
          pDistX[i] = (int16_t)newdistx;
          pDistY[i] = (int16_t)newdisty;
          dist[i] = newdist;
          changed = 1;
        }
      }
      /* Leftmost pixel is special, has no left neighbors */
      olddist = dist[i];

      if (olddist > 0) // If not already zero distance
      {
        c = i + offset_r;
        cdistx = pDistX[c];
        cdisty = pDistY[c];
        newdistx = cdistx - 1;
        newdisty = cdisty;
        newdist = distaa3(pImg, pGx, pGy, w, c, cdistx, cdisty, newdistx, newdisty);

        if (newdist < olddist - epsilon)
        {
          pDistX[i] = (int16_t)newdistx;
          pDistY[i] = (int16_t)newdisty;
          dist[i] = newdist;
          olddist = newdist;
          changed = 1;
        }

        c = i + offset_rd;
        cdistx = pDistX[c];
        cdisty = pDistY[c];
        newdistx = cdistx - 1;
        newdisty = cdisty - 1;
        newdist = distaa3(pImg, pGx, pGy, w, c, cdistx, cdisty, newdistx, newdisty);

        if (newdist < olddist - epsilon)
        {
          pDistX[i] = (int16_t)newdistx;
          pDistY[i] = (int16_t)newdisty;
          dist[i] = newdist;
          olddist = newdist;
          changed = 1;
        }

        c = i + offset_d;
        cdistx = pDistX[c];
        cdisty = pDistY[c];
        newdistx = cdistx;
        newdisty = cdisty - 1;
        newdist = distaa3(pImg, pGx, pGy, w, c, cdistx, cdisty, newdistx, newdisty);

        if (newdist < olddist - epsilon)
        {
          pDistX[i] = (int16_t)newdistx;
          pDistY[i] = (int16_t)newdisty;
          dist[i] = newdist;
          changed = 1;
        }
      }

      /* Move index to second leftmost pixel of current row. */
      /* Leftmost pixel is skipped, it has no left neighbor. */
      i = y * w + 1;

      for (x = 1; x < w; x++, i++)
      {
        /* scan right, propagate distance from left */
        olddist = dist[i];

        if (olddist <= 0)
          continue; // Already zero distance

        c = i + offset_l;
        cdistx = pDistX[c];
        cdisty = pDistY[c];
        newdistx = cdistx + 1;
        newdisty = cdisty;
        newdist = distaa3(pImg, pGx, pGy, w, c, cdistx, cdisty, newdistx, newdisty);

        if (newdist < olddist - epsilon)
        {
          pDistX[i] = (int16_t)newdistx;
          pDistY[i] = (int16_t)newdisty;
          dist[i] = newdist;
          changed = 1;
        }
      }
    }
  } while (changed); // Sweep until no more updates are made
  
  // The transformation is completed.
}

void computegradient(double_t *pImg, int32_t w, int32_t h, double_t *gx, double_t *gy)
{
  for (int32_t i = 1; i < h - 1; i++) // Avoid edges where the kernels would spill over
  {
    for (int32_t j = 1; j < w - 1; j++)
    {
      const int32_t k = i * w + j;

      if ((pImg[k] > 0.0) && (pImg[k] < 1.0)) // Compute gradient for edge pixels only
      {
        gx[k] = -pImg[k - w - 1] - mSQRT2 * pImg[k - 1] - pImg[k + w - 1] + pImg[k - w + 1] + mSQRT2 * pImg[k + 1] + pImg[k + w + 1];
        gy[k] = -pImg[k - w - 1] - mSQRT2 * pImg[k - w] - pImg[k - w + 1] + pImg[k + w - 1] + mSQRT2 * pImg[k + w] + pImg[k + w + 1];

        double_t glength = gx[k] * gx[k] + gy[k] * gy[k];

        if (glength > 0.0) // Avoid division by zero
        {
          glength = sqrt(glength);
          gx[k] = gx[k] / glength;
          gy[k] = gy[k] / glength;
        }
      }
    }
  }

  // TODO: Compute reasonable values for pGx, pGy also around the image edges.
  // (These are zero now, which reduces the accuracy for a 1-pixel wide region
  // around the image edge.) 2x2 kernels would be suitable for this.
}

mFUNCTION(make_distance_mapd, double_t *pData, const uint32_t width, const uint32_t height)
{
  mFUNCTION_SETUP();

  mERROR_IF(pData == nullptr, mR_ArgumentNull);

  int16_t *pDistX = nullptr;
  int16_t *pDistY = nullptr;
  double_t *pGx = nullptr;
  double_t *pGy = nullptr;
  double_t *pOutside = nullptr;
  double_t *pInside = nullptr;

  mERROR_CHECK(mAlloc(&pDistX, width * height));
  mDEFER_CALL(&pDistX, mFreePtr);

  mERROR_CHECK(mAlloc(&pDistY, width * height));
  mDEFER_CALL(&pDistY, mFreePtr);

  mERROR_CHECK(mAllocZero(&pGx, width * height));
  mDEFER_CALL(&pGx, mFreePtr);

  mERROR_CHECK(mAllocZero(&pGy, width * height));
  mDEFER_CALL(&pGy, mFreePtr);

  mERROR_CHECK(mAllocZero(&pOutside, width * height));
  mDEFER_CALL(&pOutside, mFreePtr);

  mERROR_CHECK(mAllocZero(&pInside, width * height));
  mDEFER_CALL(&pInside, mFreePtr);

  double_t vmin = DBL_MAX;

  // Compute pOutside = edtaa3(bitmap); % Transform background (0's)
  computegradient(pData, width, height, pGx, pGy);
  edtaa3(pData, pGx, pGy, width, height, pDistX, pDistY, pOutside);

  for (uint32_t i = 0; i < width * height; ++i)
    if (pOutside[i] < 0.0)
      pOutside[i] = 0.0;

  // Compute pInside = edtaa3(1-bitmap); % Transform foreground (1's)
  memset(pGx, 0, sizeof(double_t) * width * height);
  memset(pGy, 0, sizeof(double_t) * width * height);

  for (uint32_t i = 0; i < width * height; ++i)
    pData[i] = 1 - pData[i];

  computegradient(pData, width, height, pGx, pGy);
  edtaa3(pData, pGx, pGy, width, height, pDistX, pDistY, pInside);

  for (uint32_t i = 0; i < width * height; ++i)
    if (pInside[i] < 0)
      pInside[i] = 0.0;

  // distmap = pOutside - pInside; % Bipolar distance field
  for (uint32_t i = 0; i < width * height; ++i)
  {
    pOutside[i] -= pInside[i];

    if (pOutside[i] < vmin)
      vmin = pOutside[i];
  }

  vmin = mAbs(vmin);

  for (uint32_t i = 0; i < width * height; ++i)
  {
    double_t v = pOutside[i];

    if (v < -vmin)
      pOutside[i] = -vmin;
    else if (v > vmin)
      pOutside[i] = vmin;

    pData[i] = (pOutside[i] + vmin) / (2 * vmin);
  }

  mRETURN_SUCCESS();
}

mFUNCTION(make_distance_mapb, const uint8_t *pImg, OUT uint8_t *pOut, const uint32_t width, const uint32_t height)
{
  mFUNCTION_SETUP();

  double_t *pData = nullptr;
  mERROR_CHECK(mAllocZero(&pData, width * height));
  mDEFER_CALL(&pData, mFreePtr);

  // Find minimum and maximum values.
  double_t img_min = DBL_MAX;
  double_t img_max = DBL_MIN;

  for (uint32_t i = 0; i < width * height; ++i)
  {
    const double_t v = pImg[i];
    pData[i] = v;

    if (v > img_max)
      img_max = v;

    if (v < img_min)
      img_min = v;
  }

  // Map values from 0 - 255 to 0.0 - 1.0
  for (uint32_t i = 0; i < width * height; ++i)
    pData[i] = (pImg[i] - img_min) / img_max;

  mERROR_CHECK(make_distance_mapd(pData, width, height));

  // Map values from 0.0 - 1.0 to 0 - 255
  for (uint32_t i = 0; i < width * height; ++i)
    pOut[i] = (uint8_t)(255 * (1 - pData[i]));

  mRETURN_SUCCESS();
}

mFUNCTION(texture_font_generate_kerning, texture_font_t *pSelf, FT_Face *pFace)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSelf == nullptr || pFace == nullptr, mR_ArgumentNull);

  mPROFILE_SCOPED("Generate Font Kerning");

  // For each glyph couple combination, check if kerning is necessary.
  // Starts at index 1 since 0 is for the special background glyph.

  for (size_t i = 0; i < pSelf->glyphs.count; i++)
  {
    texture_glyph_t **ppGlyphsA = pSelf->glyphs[i];

    if (ppGlyphsA == nullptr)
      continue;

    for (int32_t subIndex0 = 0; subIndex0 < 0x100; subIndex0++)
    {
      texture_glyph_t *pGlyph = ppGlyphsA[subIndex0];

      if (pGlyph == nullptr)
        continue;

      const FT_UInt glyph_index = FT_Get_Char_Index(*pFace, pGlyph->codepoint);

      // This was the original code, but why the hell would anyone want to do that?
      // for (size_t k = 0; k < pGlyph->kerning.count; k++)
      //   mFreePtr(&pGlyph->kerning[k]);
      //
      // mERROR_CHECK(mList_Clear(pGlyph->kerning));

      for (size_t j = 0; j < pSelf->glyphs.count; j++)
      {
        texture_glyph_t **ppGlyphsB = pSelf->glyphs[j];

        if (ppGlyphsB == nullptr)
          continue;

        for (int32_t subIndex1 = 0; subIndex1 < 0x100; subIndex1++)
        {
          texture_glyph_t *pPreviousGlyph = ppGlyphsB[subIndex1];

          if (pPreviousGlyph == nullptr)
            continue;

          const FT_UInt prev_index = FT_Get_Char_Index(*pFace, pPreviousGlyph->codepoint);

          FT_Vector kerning;

          if (!texture_font_has_kerning_info(pGlyph, pPreviousGlyph->codepoint))
          {
            // FT_KERNING_UNFITTED returns FT_F26Dot6 values.
            FT_Get_Kerning(*pFace, prev_index, glyph_index, FT_KERNING_UNFITTED, &kerning);

            mERROR_CHECK(texture_font_index_kerning(pGlyph, pPreviousGlyph->codepoint, _F26Dot6ToFloat(kerning.x) / _HRESf));
          }

          if (!texture_font_has_kerning_info(pPreviousGlyph, pGlyph->codepoint))
          {
            // also insert kerning with the current added element
            FT_Get_Kerning(*pFace, glyph_index, prev_index, FT_KERNING_UNFITTED, &kerning);

            mERROR_CHECK(texture_font_index_kerning(pPreviousGlyph, pGlyph->codepoint, kerning.x / (float_t)(_HRESf * _HRESf)));
          }
        }
      }
    }
  }

  mRETURN_SUCCESS();
}

mFUNCTION(texture_font_generate_kerning_range, texture_font_t *pSelf, FT_Face *pFace, const uint32_t codepointA, const uint32_t codepointB)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSelf == nullptr || pFace == nullptr, mR_ArgumentNull);

  mPROFILE_SCOPED("Generate Kerning for requested Regions");

  const uint32_t rangeA = codepointA >> 8;
  const uint32_t rangeB = codepointB >> 8;

  mERROR_IF(pSelf->glyphs.count <= mMax(rangeA, rangeB), mR_ResourceNotFound);
  mERROR_IF(pSelf->glyphs[rangeA] == nullptr || pSelf->glyphs[rangeB] == nullptr, mR_ResourceNotFound);

  for (size_t i = 0; i < 0x100; i++)
  {
    texture_glyph_t *pGlyphA = pSelf->glyphs[rangeA][i];

    if (pGlyphA == nullptr)
      continue;

    const FT_UInt glyphIndexA = FT_Get_Char_Index(*pFace, pGlyphA->codepoint);
    
    for (size_t j = 0; j < 0x100; j++)
    {
      texture_glyph_t *pGlyphB = pSelf->glyphs[rangeB][j];

      if (pGlyphB == nullptr)
        continue;

      const FT_UInt glyphIndexB = FT_Get_Char_Index(*pFace, pGlyphB->codepoint);

      FT_Vector kerning;

      if (!texture_font_has_kerning_info(pGlyphA, pGlyphB->codepoint))
      {
        // FT_KERNING_UNFITTED returns FT_F26Dot6 values.
        FT_Get_Kerning(*pFace, glyphIndexB, glyphIndexA, FT_KERNING_UNFITTED, &kerning);

        mERROR_CHECK(texture_font_index_kerning(pGlyphA, pGlyphB->codepoint, _F26Dot6ToFloat(kerning.x) / _HRESf));
      }

      if (!texture_font_has_kerning_info(pGlyphB, pGlyphA->codepoint))
      {
        // also insert kerning with the current added element
        FT_Get_Kerning(*pFace, glyphIndexA, glyphIndexB, FT_KERNING_UNFITTED, &kerning);

        mERROR_CHECK(texture_font_index_kerning(pGlyphB, pGlyphA->codepoint, kerning.x / (float_t)(_HRESf * _HRESf)));
      }
    }
  }

  mRETURN_SUCCESS();
}

mFUNCTION(texture_font_load_glyph_gi, texture_font_t *pFont, uint32_t glyph_index, uint32_t ucodepoint, const bool failIfNotContained)
{
  mFUNCTION_SETUP();

  size_t i, x, y;

  FT_Error error;
  FT_Glyph ft_glyph = nullptr;
  FT_GlyphSlot slot;
  FT_Bitmap ft_bitmap = {};

  FT_Int32 flags = 0;
  int32_t ft_glyph_top = 0;
  int32_t ft_glyph_left = 0;

  mRectangle2D<int32_t> region;

  // Check if codepoint has been already loaded.
  mSILENT_ERROR_IF(nullptr != texture_font_find_glyph_gi(pFont, ucodepoint), mR_ResourceAlreadyExists);

  const bool fontWasNotLoaded = (pFont->face == nullptr);

  mDEFER_CALL_3(texture_font_close, pFont, MODE_AUTO_CLOSE, MODE_AUTO_CLOSE);
  mERROR_CHECK(texture_font_load_face(pFont, pFont->size, pFont->pLibrary));

  if (fontWasNotLoaded)
    glyph_index = FT_Get_Char_Index(pFont->face, ucodepoint);

  flags = 0;
  ft_glyph_top = 0;
  ft_glyph_left = 0;

  if (glyph_index == 0)
  {
    mSILENT_ERROR_IF(failIfNotContained, mR_ResourceNotFound);

    texture_glyph_t *pNullGlyph = texture_font_find_glyph(pFont, "\0");

    if (pNullGlyph != nullptr)
    {
      texture_glyph_t *pGlyph = nullptr;
      mERROR_CHECK(texture_glyph_clone(pNullGlyph, &pGlyph));
      mDEFER_CALL_ON_ERROR(&pGlyph, texture_glyph_delete);
      mERROR_CHECK(texture_font_index_glyph(pFont, pGlyph, ucodepoint));

      mRETURN_SUCCESS();
    }
  }

  // WARNING: We use texture-atlas depth to guess if user wants LCD subpixel rendering.
  if (pFont->rendermode != RENDER_NORMAL && pFont->rendermode != RENDER_SIGNED_DISTANCE_FIELD)
    flags |= FT_LOAD_NO_BITMAP;
  else
    flags |= FT_LOAD_RENDER;

  if (!pFont->useAutoHinting)
    flags |= FT_LOAD_NO_HINTING | FT_LOAD_NO_AUTOHINT;
  else
    flags |= FT_LOAD_FORCE_AUTOHINT;

  if (pFont->pAtlas->depth == 3)
  {
    mERROR_IF(FT_Library_SetLcdFilter(pFont->pLibrary->library, FT_LCD_FILTER_LIGHT), mR_InternalError);
    flags |= FT_LOAD_TARGET_LCD;

    if (pFont->useLcdFiltering)
      mERROR_IF(FT_Library_SetLcdFilterWeights(pFont->pLibrary->library, pFont->lcd_weights), mR_InternalError);
  }
  else if (_HRES == 1)
  {
    // FT_LOAD_TARGET_LIGHT:
    // A lighter hinting algorithm for gray-level modes. Many generated glyphs are fuzzier but better resemble their original shape.
    // This is achieved by snapping glyphs to the pixel grid only vertically (Y-axis), as is done by FreeType's new CFF engine or Microsoft's ClearType font renderer.
    // See: https://www.freetype.org/freetype2/docs/reference/ft2-base_interface.html#ft_load_target_xxx
    flags |= FT_LOAD_TARGET_LIGHT;
  }

  if (pFont->pAtlas->depth == 4)
    flags |= FT_LOAD_COLOR;

  error = FT_Activate_Size(pFont->ft_size);
  
  if (error)
  {
    _FreetypeError(error);
    mRETURN_RESULT(mR_InternalError);
  }

  error = FT_Load_Glyph(pFont->face, glyph_index, flags);

  if (error)
  {
    _FreetypeError(error);
    mRETURN_RESULT(mR_InternalError);
  }

  if (pFont->rendermode == RENDER_NORMAL || pFont->rendermode == RENDER_SIGNED_DISTANCE_FIELD)
  {
    slot = pFont->face->glyph;
    ft_bitmap = slot->bitmap;
    ft_glyph_top = slot->bitmap_top;
    ft_glyph_left = slot->bitmap_left;
  }
  else
  {
    FT_Stroker stroker;
    FT_BitmapGlyph ft_bitmap_glyph;

    error = FT_Stroker_New(pFont->pLibrary->library, &stroker);

    if (error)
    {
      _FreetypeError(error);
      FT_Stroker_Done(stroker);
      mRETURN_RESULT(mR_InternalError);
    }

    FT_Stroker_Set(stroker, (int32_t)(pFont->outline_thickness * _HRES), FT_STROKER_LINECAP_ROUND, FT_STROKER_LINEJOIN_ROUND, 0);

    error = FT_Get_Glyph(pFont->face->glyph, &ft_glyph);

    if (error)
    {
      _FreetypeError(error);
      FT_Stroker_Done(stroker);
      mRETURN_RESULT(mR_InternalError);
    }

    {
      mPROFILE_SCOPED("FreeType Stroke Glyph");

      if (pFont->rendermode == RENDER_OUTLINE_EDGE)
        error = FT_Glyph_Stroke(&ft_glyph, stroker, 1);
      else if (pFont->rendermode == RENDER_OUTLINE_POSITIVE)
        error = FT_Glyph_StrokeBorder(&ft_glyph, stroker, 0, 1);
      else if (pFont->rendermode == RENDER_OUTLINE_NEGATIVE)
        error = FT_Glyph_StrokeBorder(&ft_glyph, stroker, 1, 1);
    }

    if (error)
    {
      _FreetypeError(error);
      FT_Stroker_Done(stroker);
      mRETURN_RESULT(mR_InternalError);
    }

    switch (pFont->pAtlas->depth) {
    case 1:
      error = FT_Glyph_To_Bitmap(&ft_glyph, FT_RENDER_MODE_NORMAL, 0, 1);
      break;
    case 3:
      error = FT_Glyph_To_Bitmap(&ft_glyph, FT_RENDER_MODE_LCD, 0, 1);
      break;
    case 4:
      error = FT_Glyph_To_Bitmap(&ft_glyph, FT_RENDER_MODE_NORMAL, 0, 1);
      break;
    }

    if (error)
    {
      _FreetypeError(error);
      FT_Stroker_Done(stroker);
      mRETURN_RESULT(mR_InternalError);
    }

    ft_bitmap_glyph = (FT_BitmapGlyph)ft_glyph;
    ft_bitmap = ft_bitmap_glyph->bitmap;
    ft_glyph_top = ft_bitmap_glyph->top;
    ft_glyph_left = ft_bitmap_glyph->left;

    FT_Stroker_Done(stroker);

    mERROR_IF(error, mR_InternalError);
  }

  struct
  {
    int32_t left;
    int32_t top;
    int32_t right;
    int32_t bottom;
  } padding = { 0, 0, 1, 1 };

  if (pFont->rendermode == RENDER_SIGNED_DISTANCE_FIELD)
  {
    padding.top = 1;
    padding.left = 1;
  }

  if (pFont->padding != 0)
  {
    padding.top += pFont->padding;
    padding.left += pFont->padding;
    padding.right += pFont->padding;
    padding.bottom += pFont->padding;
  }

  size_t src_w = pFont->pAtlas->depth == 3 ? ft_bitmap.width / 3 : ft_bitmap.width;
  size_t src_h = ft_bitmap.rows;

  size_t tgt_w = src_w + padding.left + padding.right;
  size_t tgt_h = src_h + padding.top + padding.bottom;

  const mResult atlasResult = texture_atlas_get_region(pFont->pAtlas, tgt_w, tgt_h, &region);

  if (mFAILED(atlasResult))
  {
    if (atlasResult == mR_ResourceBusy)
      mSILENT_RETURN_RESULT(atlasResult);
    else
      mRETURN_RESULT(atlasResult);
  }

  mERROR_IF(region.x < 0, mR_InternalError); // Should've been returned by `texture_atlas_get_region` already as `mR_ResourceBusy`.

  x = region.x;
  y = region.y;

  // Copy pixel data over
  uint8_t *pBuffer = nullptr;
  mERROR_CHECK(mAllocZero(&pBuffer, tgt_w * tgt_h * pFont->pAtlas->depth));
  mDEFER_CALL(&pBuffer, mFreePtr);

  uint8_t *pDestination = pBuffer + (padding.top * tgt_w + padding.left) * pFont->pAtlas->depth;
  uint8_t *pSource = ft_bitmap.buffer;

  if (ft_bitmap.pixel_mode == FT_PIXEL_MODE_BGRA && pFont->pAtlas->depth == 4)
  {
    // BGRA in, RGBA out
    for (i = 0; i < src_h; i++)
    {
      for (size_t j = 0; j < ft_bitmap.width; j++)
      {
        uint32_t bgra, rgba;
        bgra = ((uint32_t *)pSource)[j];
#if __BYTE_ORDER == __BIG_ENDIAN
        rgba = _RollBits(_SwapBytes(bgra), 8);
#else
        rgba = _RollBits(_SwapBytes(bgra), 24);
#endif
        ((uint32_t *)pDestination)[j] = rgba;
      }

      pDestination += tgt_w * pFont->pAtlas->depth;
      pSource += ft_bitmap.pitch;
    }
  }
  else if (ft_bitmap.pixel_mode == FT_PIXEL_MODE_BGRA && pFont->pAtlas->depth == 1)
  {
    // BGRA in, grey out: Use weighted sum for luminosity, and multiply by alpha
    struct src_pixel_t { uint8_t b; uint8_t g; uint8_t r; uint8_t a; } *pSrc = reinterpret_cast<src_pixel_t *>(ft_bitmap.buffer);

    for (int32_t row = 0; row < src_h; row++, pDestination += tgt_w * pFont->pAtlas->depth)
      for (int32_t col = 0; col < src_w; col++, pSrc++)
        pDestination[col] = (uint8_t)((0.3 * pSrc->r + 0.59 * pSrc->g + 0.11 * pSrc->b) * (pSrc->a / 255.0));
  }
  else if (ft_bitmap.pixel_mode == FT_PIXEL_MODE_GRAY && pFont->pAtlas->depth == 4)
  {
    // Grey in, RGBA out: Use grey level for alpha channel, with white color
    struct dst_pixel_t { uint8_t r; uint8_t g; uint8_t b; uint8_t a; } *pDst = reinterpret_cast<dst_pixel_t *>(pDestination);

    for (int32_t row = 0; row < src_h; row++, pDst += tgt_w)
      for (int32_t col = 0; col < src_w; col++, pSource++)
        pDst[col] = { 255, 255, 255, *pSource };
  }
  else
  {
    // Straight copy, per row
    for (i = 0; i < src_h; i++)
    {
      //difference between width and pitch: https://www.freetype.org/freetype2/docs/reference/ft2-basic_types.html#FT_Bitmap
      memcpy(pDestination, pSource, ft_bitmap.width);
      pDestination += tgt_w * pFont->pAtlas->depth;
      pSource += ft_bitmap.pitch;
    }
  }

  if (pFont->rendermode == RENDER_SIGNED_DISTANCE_FIELD)
  {
    mPROFILE_SCOPED("Generate Signed Distance Field");

    uint8_t *pSignedDistanceField = nullptr;
    mERROR_CHECK(mAllocZero(&pSignedDistanceField, tgt_w * tgt_h));
    mDEFER_CALL(&pSignedDistanceField, mFreePtr);

    mERROR_CHECK(make_distance_mapb(pBuffer, pSignedDistanceField, (uint32_t)tgt_w, (uint32_t)tgt_h));
    
    if (pSignedDistanceField != nullptr)
      std::swap(pBuffer, pSignedDistanceField);
  }

  mERROR_CHECK(texture_atlas_set_region(pFont->pAtlas, x, y, tgt_w, tgt_h, pBuffer, tgt_w * pFont->pAtlas->depth));

  texture_glyph_t *pGlyph = nullptr;
  mERROR_CHECK(texture_glyph_new(&pGlyph, pFont->pAllocator));

  pGlyph->codepoint = glyph_index ? ucodepoint : 0;
  pGlyph->width = tgt_w;
  pGlyph->height = tgt_h;
  pGlyph->rendermode = pFont->rendermode;
  pGlyph->outline_thickness = pFont->outline_thickness;
  pGlyph->offset_x = ft_glyph_left;
  pGlyph->offset_y = ft_glyph_top;
  
  if (pFont->scaleTextureCoords)
  {
    pGlyph->s0 = x / (float_t)pFont->pAtlas->width;
    pGlyph->t0 = y / (float_t)pFont->pAtlas->height;
    pGlyph->s1 = (x + pGlyph->width) / (float_t)pFont->pAtlas->width;
    pGlyph->t1 = (y + pGlyph->height) / (float_t)pFont->pAtlas->height;
  }
  else
  {
    // fix up unscaled coordinates by subtracting 0.5
    // this avoids drawing pixels from neighboring characters
    // note that you also have to paint these glyphs with an offset of
    // half a pixel each to get crisp rendering
    pGlyph->s0 = x - 0.5f;
    pGlyph->t0 = y - 0.5f;
    pGlyph->s1 = x + tgt_w - 0.5f;
    pGlyph->t1 = y + tgt_h - 0.5f;
  }

  slot = pFont->face->glyph;

  if (FT_HAS_FIXED_SIZES(pFont->face))
  {
    // color fonts use actual pixels, not subpixels
    pGlyph->advance_x = (float_t)slot->advance.x;
    pGlyph->advance_y = (float_t)slot->advance.y;
  }
  else
  {
    pGlyph->advance_x = _F26Dot6ToFloat(slot->advance.x) * pFont->scale;
    pGlyph->advance_y = _F26Dot6ToFloat(slot->advance.y) * pFont->scale;
  }

  bool free_glyph = false;
  mERROR_CHECK(texture_font_index_glyph(pFont, pGlyph, ucodepoint, &free_glyph));

  if (glyph_index == 0)
  {
    if (!free_glyph)
      mERROR_CHECK(texture_glyph_clone(pGlyph, &pGlyph));

    mDEFER_ON_ERROR(if (!free_glyph) texture_glyph_delete(&pGlyph));
    mERROR_CHECK(texture_font_index_glyph(pFont, pGlyph, 0, &free_glyph));
  }

  if (free_glyph)
    mERROR_CHECK(mAllocator_FreePtr(pGlyph->pAllocator, &pGlyph));

  if (pFont->rendermode != RENDER_NORMAL && pFont->rendermode != RENDER_SIGNED_DISTANCE_FIELD)
    FT_Done_Glyph(ft_glyph);

  // Generate Kerning for immediate surroundings.
  mERROR_CHECK(texture_font_generate_kerning_range(pFont, &pFont->face, pGlyph->codepoint, pGlyph->codepoint));

  mRETURN_SUCCESS();
}

mFUNCTION(texture_font_load_glyph, texture_font_t *pSelf, const uint32_t codepoint, const bool failIfNotContained)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSelf == nullptr, mR_ArgumentNull);

  const mResult result = texture_font_load_glyph_gi(pSelf, FT_Get_Char_Index(pSelf->face, codepoint), codepoint, failIfNotContained);

  mSILENT_RETURN_RESULT(result);
}

texture_glyph_t * texture_font_find_glyph(texture_font_t *pSelf, const char *codepoint)
{
  if (!codepoint)
    return pSelf->pAtlas->pSpecial;

  return texture_font_find_glyph_gi(pSelf, _Utf8ToUtf32(codepoint));
}

texture_glyph_t * texture_font_find_glyph_gi(texture_font_t *pSelf, const uint32_t codepoint)
{
  const uint32_t i = codepoint >> 8;
  const uint32_t j = codepoint & 0xFF;

  if (pSelf->glyphs.count <= i)
    return nullptr;

  texture_glyph_t **ppGlyphIndex = pSelf->glyphs[i];
  
  if (!ppGlyphIndex)
    return nullptr;

  texture_glyph_t *pGlyph = ppGlyphIndex[j];

  while (pGlyph != nullptr && (pGlyph->rendermode != pSelf->rendermode || pGlyph->outline_thickness != pSelf->outline_thickness))
  {
    if (pGlyph->glyphmode != GLYPH_CONT)
      return nullptr;

    pGlyph++;
  }

  return pGlyph;
}

mFUNCTION(texture_font_get_glyph_cp, texture_font_t *pSelf, const uint32_t codepoint, texture_glyph_t **ppGlyph, const bool failIfNotContained)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSelf == nullptr || ppGlyph == nullptr, mR_ArgumentNull);
  mERROR_IF(pSelf->filename.bytes <= 1 || pSelf->pAtlas == nullptr, mR_InternalError);

  const uint32_t glyph_index = FT_Get_Char_Index(pSelf->face, codepoint);

  // Check if glyph_index has been already loaded.
  texture_glyph_t *pGlyph = texture_font_find_glyph_gi(pSelf, glyph_index);

  if (pGlyph == nullptr)
  {
    // Glyph has not been already loaded.
    mERROR_CHECK(texture_font_load_glyph(pSelf, glyph_index, failIfNotContained));

    pGlyph = texture_font_find_glyph_gi(pSelf, glyph_index);
  }

  mERROR_IF(pGlyph == nullptr, mR_ResourceNotFound);
  *ppGlyph = pGlyph;
 
  mRETURN_SUCCESS();
}

mFUNCTION(texture_font_get_glyph, texture_font_t *pSelf, const char *codepoint, texture_glyph_t **ppGlyph, const bool failIfNotContained)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSelf == nullptr || ppGlyph == nullptr, mR_ArgumentNull);
  mERROR_IF(pSelf->filename.bytes <= 1 || pSelf->pAtlas == nullptr, mR_InternalError);

  // Check if codepoint has been already loaded.
  texture_glyph_t *pGlyph = texture_font_find_glyph(pSelf, codepoint);

  if (pGlyph == nullptr)
  {
    // Glyph has not been already loaded.
    mERROR_CHECK(texture_font_load_glyph(pSelf, _Utf8ToUtf32(codepoint), failIfNotContained));
   
    pGlyph = texture_font_find_glyph(pSelf, codepoint);
  }

  mERROR_IF(pGlyph == nullptr, mR_ResourceNotFound);
  *ppGlyph = pGlyph;

  mRETURN_SUCCESS();
}

int32_t texture_atlas_fit(texture_atlas_t *pSelf, const size_t index, const size_t width, const size_t height)
{
  size_t i;

  mASSERT_DEBUG(pSelf != nullptr, "Invalid Parameter");

  mVec3i32 *node = &pSelf->nodes[index];
  int32_t x = node->x;
  int32_t y = node->y;
  int32_t width_left = (int32_t)width;
  i = index;

  if ((x + width) > (pSelf->width - 1))
    return -1;

  y = node->y;

  while (width_left > 0)
  {
    node = &pSelf->nodes[i];

    if (node->y > y)
      y = node->y;

    if ((y + height) > (pSelf->height - 1))
      return -1;

    width_left -= node->z;
    ++i;
  }

  return y;
}

mFUNCTION(texture_atlas_merge, texture_atlas_t *pSelf)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSelf == nullptr, mR_ArgumentNull);

  for (size_t i = 0; i < pSelf->nodes.count - 1; i++)
  {
    mVec3i32 *pNode = &pSelf->nodes[i];
    mVec3i32 *pNext = &pSelf->nodes[i + 1];
    
    if (pNode->y == pNext->y)
    {
      pNode->z += pNext->z;

      mVec3i32 _unused;
      mERROR_CHECK(mList_PopAt(pSelf->nodes, i + 1, &_unused));
      
      i--;
    }
  }

  mRETURN_SUCCESS();
}

mFUNCTION(texture_atlas_get_region, texture_atlas_t *pSelf, const size_t width, const size_t height, OUT mRectangle2D<int32_t> *pRegion)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSelf == nullptr, mR_ArgumentNull);
  mERROR_IF(pRegion == nullptr, mR_ArgumentNull);

  int32_t y, best_index;
  size_t best_height, best_width;
  mRectangle2D<int32_t> region(0, 0, (int32_t)width, (int32_t)height);

  best_height = UINT_MAX;
  best_index = -1;
  best_width = UINT_MAX;

  for (size_t i = 0; i < pSelf->nodes.count; ++i)
  {
    y = texture_atlas_fit(pSelf, i, width, height);

    if (y >= 0)
    {
      const mVec3i32 *pNode = &pSelf->nodes[i];

      if (((y + height) < best_height) ||
        (((y + height) == best_height) && (pNode->z > 0 && (size_t)pNode->z < best_width))) {
        best_height = y + height;
        best_index = (int32_t)i;
        best_width = pNode->z;
        region.x = pNode->x;
        region.y = y;
      }
    }
  }

  mSILENT_ERROR_IF(best_index == -1, mR_ResourceBusy);

  mVec3i32 node;

  node.x = region.x;
  node.y = region.y + (int32_t)height;
  node.z = (int32_t)width;

  mERROR_CHECK(mList_InsertAt(pSelf->nodes, best_index, std::move(node)));

  for (size_t i = best_index + 1; i < pSelf->nodes.count; ++i)
  {
    mVec3i32 *pNode = &pSelf->nodes[i];
    mVec3i32 *pPrevious = &pSelf->nodes[i - 1];

    if (pNode->x < (pPrevious->x + pPrevious->z))
    {
      const int32_t shrink = pPrevious->x + pPrevious->z - pNode->x;

      pNode->x += shrink;
      pNode->z -= shrink;

      if (pNode->z <= 0)
      {
        mVec3i32 _unused;
        mERROR_CHECK(mList_PopAt(pSelf->nodes, i, &_unused));
        i--;
      }
      else
      {
        break;
      }
    }
    else
    {
      break;
    }
  }

  mERROR_CHECK(texture_atlas_merge(pSelf));
  pSelf->used += width * height;
  pSelf->isModified = true;

  *pRegion = region;

  mRETURN_SUCCESS();
}

mFUNCTION(texture_atlas_set_region, texture_atlas_t *pSelf, const size_t x, const size_t y, const size_t width, const size_t height, const uint8_t *pData, const size_t stride)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSelf == nullptr, mR_ArgumentNull);
  mERROR_IF(pData == nullptr, mR_ArgumentNull);
  mERROR_IF(!(x > 0), mR_InvalidParameter);
  mERROR_IF(!(y > 0), mR_InvalidParameter);
  mERROR_IF(!(x < (pSelf->width - 1)), mR_ArgumentOutOfBounds);
  mERROR_IF(!((x + width) <= (pSelf->width - 1)), mR_ArgumentOutOfBounds);
  mERROR_IF(!(y < (pSelf->height - 1)), mR_ArgumentOutOfBounds);
  mERROR_IF(!((y + height) <= (pSelf->height - 1)), mR_ArgumentOutOfBounds);

  // prevent copying data from undefined position and prevent memcpy's undefined behavior when count is zero
  mERROR_IF(!(height == 0 || (pData != nullptr && width > 0)), mR_InvalidParameter);

  const size_t depth = pSelf->depth;
  const size_t charsize = sizeof(char);

  for (size_t i = 0; i < height; ++i)
    memcpy(pSelf->pData + ((y + i) * pSelf->width + x) * charsize * depth, pData + (i * stride) * charsize, width * charsize * depth);

  pSelf->isModified = 1;

  mRETURN_SUCCESS();
}

mFUNCTION(texture_atlas_special, texture_atlas_t *pSelf)
{
  mFUNCTION_SETUP();

  mRectangle2D<int32_t> region;
  mERROR_CHECK(texture_atlas_get_region(pSelf, 5, 5, &region));
  mERROR_IF(region.x < 0, mR_InternalError); // Should've been returned already in `texture_atlas_get_region` as `mR_ResourceBusy`.

  texture_glyph_t *pGlyph = nullptr;
  mERROR_CHECK(texture_glyph_new(&pGlyph, pSelf->pAllocator));
  mDEFER_CALL_ON_ERROR(&pGlyph, texture_glyph_delete);

  static uint8_t data[4 * 4 * 3] = 
  {
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF
  };

  mERROR_CHECK(texture_atlas_set_region(pSelf, region.x, region.y, 4, 4, data, 0));
  
  pGlyph->codepoint = (uint32_t)-1;
  
  pGlyph->s0 = (region.x + 2) / (float_t)pSelf->width;
  pGlyph->t0 = (region.y + 2) / (float_t)pSelf->height;
  pGlyph->s1 = (region.x + 3) / (float_t)pSelf->width;
  pGlyph->t1 = (region.y + 3) / (float_t)pSelf->height;

  pSelf->pSpecial = pGlyph;

  mRETURN_SUCCESS();
}

mFUNCTION(texture_atlas_new, OUT texture_atlas_t **ppAtlas, IN mAllocator *pAllocator, const size_t width, const size_t height, const size_t depth)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppAtlas == nullptr, mR_ArgumentNull);
  mERROR_IF(!((depth == 1) || (depth == 3) || (depth == 4)), mR_InvalidParameter);

  texture_atlas_t *pSelf = nullptr;
  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &pSelf, 1));
  mDEFER_ON_ERROR(mAllocator_FreePtr(pAllocator, &pSelf));

  pSelf->pAllocator = pAllocator;

  pSelf->used = 0;
  pSelf->width = width;
  pSelf->height = height;
  pSelf->depth = depth;
  pSelf->id = 0;
  pSelf->isModified = true;

  mERROR_CHECK(mList_Create(&pSelf->nodes, pAllocator));

  // We want a one pixel border around the whole atlas to avoid any artefact when sampling texture.
  mVec3i32 node(1, 1, (int32_t)width - 2);

  mERROR_CHECK(mList_PushBack(pSelf->nodes, node));

  mERROR_CHECK(mAllocator_AllocateZero(pSelf->pAllocator, &pSelf->pData, width * height * depth));
  mDEFER_ON_ERROR(mAllocator_FreePtr(pSelf->pAllocator, &pSelf->pData));

  mERROR_CHECK(texture_atlas_special(pSelf));

  *ppAtlas = pSelf;

  mRETURN_SUCCESS();
}

void texture_atlas_clear(texture_atlas_t *pSelf)
{
  mVec3i32 node = mVec3i32(1);

  if (pSelf == nullptr || pSelf->pData == nullptr)
    return;

  mList_Clear(pSelf->nodes);

  pSelf->used = 0;

  // We want a one pixel border around the whole atlas to avoid any artefact when sampling the texture.
  node.z = (int32_t)pSelf->width - 2;

  mList_PushBack(pSelf->nodes, node);

  memset(pSelf->pData, 0, pSelf->width * pSelf->height * pSelf->depth);
}

void texture_atlas_delete(texture_atlas_t *pSelf)
{
  if (pSelf == nullptr)
    return;

  mList_Destroy(&pSelf->nodes);

  texture_glyph_delete(&pSelf->pSpecial);
  
  mAllocator_FreePtr(pSelf->pAllocator, &pSelf->pData);
  mAllocator_Free(pSelf->pAllocator, pSelf);
}
