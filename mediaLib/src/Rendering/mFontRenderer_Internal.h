#ifndef mFontRenderer_Internal_h__
#define mFontRenderer_Internal_h__

#include "mList.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "AdTmmCyiBJ71Kvru4+XdyXEW/qAcNBrdtbytBLOU4cNdAzugIMKXYN3E7HHK/8by+EnEDZHl46hqVfxx"
#endif

struct texture_glyph_t;

// A texture atlas is used to pack several small regions into a single texture.
struct texture_atlas_t
{
  mList<mVec3i32> nodes;

  // Width (in pixels) of the underlying texture.
  size_t width;

  // Height (in pixels) of the underlying texture.
  size_t height;

  // Depth (in bytes) of the underlying texture.
  size_t depth;

  // Allocated surface size.
  size_t used;

  // OpenGL texture ID.
  uint32_t id;

  uint8_t *pData;
  bool isModified;
  texture_glyph_t *pSpecial;

  mAllocator *pAllocator;
};

// A list of possible ways to render a glyph.
enum rendermode_t
{
  RENDER_NORMAL,
  RENDER_OUTLINE_EDGE,
  RENDER_OUTLINE_POSITIVE,
  RENDER_OUTLINE_NEGATIVE,
  RENDER_SIGNED_DISTANCE_FIELD
};

// Glyph array end mark type.
enum glyphmode_t
{
  GLYPH_END = 0,
  GLYPH_CONT = 1
};

// If there is no Freetype included, just define that as incomplete pointer.
#if !defined(FT2BUILD_H_) && !defined(__FT2BUILD_H__) && !defined(FREETYPE_H_)
typedef struct FT_FaceRec_ *FT_Face;
typedef struct FT_LibraryRec_ *FT_Library;
typedef struct FT_SizeRec_ *FT_Size;
#endif

#if !defined(FTTYPES_H_)
typedef unsigned char FT_Byte;
#endif

// same for harfbuzz.
#ifndef HB_BUFFER_H
typedef struct hb_font_t hb_font_t;
#endif

struct texture_glyph_t
{
  // Unicode codepoint this glyph represents in UTF-32 LE encoding.
  uint32_t codepoint;

  // Glyph's width in pixels.
  size_t width;

  // Glyph's height in pixels.
  size_t height;

  // Glyph's left bearing expressed in integer pixels.
  int32_t offset_x;

  // Glyphs's top bearing expressed in integer pixels.
  // Remember that this is the distance from the baseline to the top-most glyph scanline, upwards y coordinates being positive.
  int32_t offset_y;

  // For horizontal text layouts, this is the horizontal distance (in fractional pixels) used to increment the pen position when the glyph is drawn as part of a string of text.
  float_t advance_x;

  // For vertical text layouts, this is the vertical distance (in fractional pixels) used to increment the pen position when the glyph is drawn as part of a string of text.
  float_t advance_y;

  // First normalized texture coordinate (x) of top-left corner.
  float_t s0;

  // Second normalized texture coordinate (y) of top-left corner. 
  float_t t0;

  // First normalized texture coordinate (x) of bottom-right corner.
  float_t s1;

  // Second normalized texture coordinate (y) of bottom-right corner.
  float_t t1;

  // A list of kerning pairs relative to this glyph.
  mList<float_t *> kerning;

  rendermode_t rendermode;
  float_t outline_thickness;
  glyphmode_t glyphmode;

  mAllocator *pAllocator;
};

enum font_location_t
{
  TEXTURE_FONT_FILE = 0,
  TEXTURE_FONT_MEMORY
};

enum font_mode_t
{
  MODE_AUTO_CLOSE = 0,
  MODE_GLYPHS_CLOSE,
  MODE_FREE_CLOSE,
  MODE_MANUAL_CLOSE,
  MODE_ALWAYS_OPEN
};

struct texture_font_library_t
{
  font_mode_t mode;
  FT_Library library;

  mAllocator *pAllocator;
};

struct texture_font_t
{
  // Vector of glyphs contained in this font.
  // This is actually a two-stage table, indexing into 256 glyphs each
  mList<texture_glyph_t **> glyphs;

  texture_atlas_t *pAtlas;
  font_location_t location;

  union
  {
    mInplaceString<0xFF> filename;

    struct
    {
      const FT_Byte *pBase;
      size_t size;
    } memory;
  };

  texture_font_library_t *pLibrary;

  // font size.
  float_t size;

  rendermode_t rendermode;
  float_t outline_thickness;
  bool useLcdFiltering, useKerning, useAutoHinting, scaleTextureCoords;
  uint8_t lcd_weights[5];

  // This field is simply used to compute a default line spacing (i.e., the baseline-to-baseline distance) when writing text with this font. Note that it usually is larger than the sum of the ascender and descender taken as absolute values. There is also no guarantee that no glyphs extend above or below subsequent baselines when using this distance.
  float_t height;

  // This field is the distance that must be placed between two lines of text. The baseline-to-baseline distance should be computed as: ascender - descender + linegap.
  float_t linegap;

  // The ascender is the vertical distance from the horizontal baseline to the highest 'character' coordinate in a font face. Unfortunately, font formats define the ascender differently. For some, it represents the ascent of all capital latin characters (without accents), for others it is the ascent of the highest accented character, and finally, other formats define it as being equal to bbox.yMax.
  float_t ascender;

  // The descender is the vertical distance from the horizontal baseline to the lowest 'character' coordinate in a font face. Unfortunately, font formats define the descender differently. For some, it represents the descent of all capital latin characters (without accents), for others it is the ascent of the lowest accented character, and finally, other formats define it as being equal to bbox.yMin. This field is negative for values below the baseline.
  float_t descender;

  // The position of the underline line for this face. It is the center of the underlining stem. Only relevant for scalable formats.
  float_t underline_position;

  // The thickness of the underline for this face. Only relevant for scalable formats. 
  float_t underline_thickness;

  // The padding to be add to the glyph's texture that are loaded by this font.
  // Usefull when adding effects with shaders.
  int32_t padding;

  font_mode_t mode;
  FT_Face face;
  FT_Size ft_size;
  hb_font_t *pHbFont;

  // factor to scale font coordinates.
  float_t scale;

  mAllocator *pAllocator;
};

mFUNCTION(texture_library_new, OUT texture_font_library_t **pLibrary, IN mAllocator *pAllocator);

mFUNCTION(texture_font_new_from_file, texture_atlas_t *pAtlas, OUT texture_font_t **ppFont, IN mAllocator *pAllocator, const float_t pt_size, const char *filename, const font_mode_t fontMode, texture_font_library_t *pLibrary);
void texture_font_delete(texture_font_t *pSelf);

mFUNCTION(texture_font_get_glyph, texture_font_t *pSelf, const char *codepoint, texture_glyph_t **ppGlyph, const bool failIfNotContained);
mFUNCTION(texture_font_get_glyph_cp, texture_font_t *pSelf, const uint32_t codepoint, texture_glyph_t **ppGlyph, const bool failIfNotContained);
mFUNCTION(texture_font_load_glyph, texture_font_t *pSelf, const uint32_t codepoint, const bool failIfNotContained);
texture_glyph_t * texture_font_find_glyph(texture_font_t *pSelf, const char *codepoint);

void texture_glyph_delete(texture_glyph_t **ppGlyph);
float texture_glyph_get_kerning(const texture_glyph_t *pSelf, const uint32_t codepoint);

mFUNCTION(texture_atlas_new, OUT texture_atlas_t **ppAtlas, IN mAllocator *pAllocator, const size_t width, const size_t height, const size_t depth);
void texture_atlas_clear(texture_atlas_t *pSelf);
void texture_atlas_delete(texture_atlas_t *pSelf);

texture_glyph_t *texture_font_find_glyph_gi(texture_font_t *pSelf, const uint32_t codepoint);

#endif // mFontRenderer_Internal_h__
