#ifndef mFontRenderer_h__
#define mFontRenderer_h__

#include "mRenderParams.h"

struct mFontDescription
{
  mInplaceString<255> fontFileName;
  float_t fontSize = 24.0f;
  float_t lineHeightRatio = 1.0f;
  float_t glyphSpacingRatio = 1.0f;
};

struct mFontRenderer;
struct mFontRenderable;

mFUNCTION(mFontRenderer_Create, OUT mPtr<mFontRenderer> *pFontRenderer, IN mAllocator *pAllocator, const size_t width = 2048, const size_t height = 2048);
mFUNCTION(mFontRenderer_Destroy, IN_OUT mPtr<mFontRenderer> *pFontRenderer);

mFUNCTION(mFontRenderer_Begin, mPtr<mFontRenderer> &fontRenderer, const mMatrix &matrix = mMatrix::Scale(2.0f / mRenderParams_CurrentRenderResolutionF.x, 2.0f / mRenderParams_CurrentRenderResolutionF.y, 1) * mMatrix::Translation(-1, -1, 0));
mFUNCTION(mFontRenderer_BeginRenderable, mPtr<mFontRenderer> &fontRenderer);
mFUNCTION(mFontRenderer_End, mPtr<mFontRenderer> &fontRenderer);
mFUNCTION(mFontRenderer_EndRenderable, mPtr<mFontRenderer> &fontRenderer, OUT mPtr<mFontRenderable> *pRenderable, IN mAllocator *pAllocator);

mFUNCTION(mFontRenderer_Draw, mPtr<mFontRenderer> &fontRenderer, const mFontDescription &fontDescription, const mString &string, const mVector colour = mVector(1, 1, 1, 1));

mFUNCTION(mFontRenderer_AddFont, mPtr<mFontRenderer> &fontRenderer, const mFontDescription &fontDescription);
mFUNCTION(mFontRenderer_SetPosition, mPtr<mFontRenderer> &fontRenderer, const mVec2f position);
mFUNCTION(mFontRenderer_SetDisplayPosition, mPtr<mFontRenderer> &fontRenderer, const mVec2f position);
mFUNCTION(mFontRenderer_GetCurrentPosition, mPtr<mFontRenderer> &fontRenderer, OUT mVec2f *pPosition);
mFUNCTION(mFontRenderer_GetCurrentDisplayPosition, mPtr<mFontRenderer> &fontRenderer, OUT mVec2f *pPosition);

mFUNCTION(mFontRenderable_Draw, mPtr<mFontRenderable> &fontRenderable, const mMatrix &matrix = mMatrix::Scale(2.0f / mRenderParams_CurrentRenderResolutionF.x, 2.0f / mRenderParams_CurrentRenderResolutionF.y, 1) * mMatrix::Translation(-1, -1, 0));
mFUNCTION(mFontRenderable_Destroy, IN_OUT mPtr<mFontRenderable> *pFontRenderable);

#endif // mFontRenderer_h__
