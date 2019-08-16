#ifndef mFontRenderer_h__
#define mFontRenderer_h__

#include "mRenderParams.h"

enum mFontDescription_BoundMode
{
  mFD_BM_Stop,
  mFD_BM_BreakLineOnBound,
  mFD_BM_StopWithTrippleDot,
};

struct mFontDescription
{
  mInplaceString<MAX_PATH> fontFileName;
  float_t fontSize = 24.0f;
  float_t lineHeightRatio = 1.0f;
  float_t glyphSpacingRatio = 1.0f;
  float_t scale = 1;

  bool hasBounds = false;
  mRectangle2D<float_t> bounds;
  mFontDescription_BoundMode boundMode = mFD_BM_Stop;

  bool ignoreBackupFonts = false;
};

mRectangle2D<float_t> mFontDescrption_GetDisplayBounds(const mRectangle2D<float_t> &displayBounds);

struct mFontRenderer;
struct mFontRenderable;

mFUNCTION(mFontRenderer_Create, OUT mPtr<mFontRenderer> *pFontRenderer, IN mAllocator *pAllocator, const size_t width = 2048, const size_t height = 2048);
mFUNCTION(mFontRenderer_Destroy, IN_OUT mPtr<mFontRenderer> *pFontRenderer);

mFUNCTION(mFontRenderer_Begin, mPtr<mFontRenderer> &fontRenderer, const mMatrix &matrix = mMatrix::Scale(2.0f / mRenderParams_CurrentRenderResolutionF.x, 2.0f / mRenderParams_CurrentRenderResolutionF.y, 1) * mMatrix::Translation(-1, -1, 0), const mVec3f right = mVec3f(1, 0, 0), const mVec3f up = mVec3f(0, 1, 0));
mFUNCTION(mFontRenderer_BeginRenderable, mPtr<mFontRenderer> &fontRenderer);
mFUNCTION(mFontRenderer_End, mPtr<mFontRenderer> &fontRenderer);
mFUNCTION(mFontRenderer_EndRenderable, mPtr<mFontRenderer> &fontRenderer, OUT mPtr<mFontRenderable> *pRenderable, IN mAllocator *pAllocator);

mFUNCTION(mFontRenderer_Draw, mPtr<mFontRenderer> &fontRenderer, const mFontDescription &fontDescription, const mString &string, const mVector colour = mVector(1, 1, 1, 1));

mFUNCTION(mFontRenderer_AddFont, mPtr<mFontRenderer> &fontRenderer, const mFontDescription &fontDescription);
mFUNCTION(mFontRenderer_SetOrigin, mPtr<mFontRenderer> &fontRenderer, const mVec3f position);
mFUNCTION(mFontRenderer_SetPosition, mPtr<mFontRenderer> &fontRenderer, const mVec2f position);
mFUNCTION(mFontRenderer_SetDisplayPosition, mPtr<mFontRenderer> &fontRenderer, const mVec2f position);
mFUNCTION(mFontRenderer_GetCurrentPosition, mPtr<mFontRenderer> &fontRenderer, OUT mVec2f *pPosition);
mFUNCTION(mFontRenderer_GetCurrentDisplayPosition, mPtr<mFontRenderer> &fontRenderer, OUT mVec2f *pPosition);

mFUNCTION(mFontRenderer_ResetRenderedRect, mPtr<mFontRenderer> &fontRenderer);
mFUNCTION(mFontRenderer_GetRenderedRect, mPtr<mFontRenderer> &fontRenderer, OUT mRectangle2D<float_t> *pRenderedArea);
mFUNCTION(mFontRenderer_GetRenderedDisplayRect, mPtr<mFontRenderer> &fontRenderer, OUT mRectangle2D<float_t> *pRenderedArea);

mFUNCTION(mFontRenderer_AddBackupFont, mPtr<mFontRenderer> &fontRenderer, const mString &backupFont);
mFUNCTION(mFontRenderer_ClearBackupFonts, mPtr<mFontRenderer> &fontRenderer);

mFUNCTION(mFontRenderable_Draw, mPtr<mFontRenderable> &fontRenderable, const mMatrix &matrix = mMatrix::Scale(2.0f / mRenderParams_CurrentRenderResolutionF.x, 2.0f / mRenderParams_CurrentRenderResolutionF.y, 1) * mMatrix::Translation(-1, -1, 0));
mFUNCTION(mFontRenderable_Destroy, IN_OUT mPtr<mFontRenderable> *pFontRenderable);

mFUNCTION(mFontRenderer_SaveAtlasTo, mPtr<mFontRenderer> &fontRenderer, const mString &directory);

#endif // mFontRenderer_h__
