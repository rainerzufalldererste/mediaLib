#ifndef mColourLookup_h__
#define mColourLookup_h__

#include "mediaLib.h"

struct mColourLookup
{
  mVec3s resolution;
  mVec3f *pData;
  size_t capacity;
  mAllocator *pAllocator;
};

mFUNCTION(mColourLookup_CreateFromFile, OUT mPtr<mColourLookup> *pColourLookup, IN mAllocator *pAllocator, const mString &filename);
mFUNCTION(mColourLookup_Destroy, IN_OUT mPtr<mColourLookup> *pColourLookup);

mFUNCTION(mColourLookup_LoadFromFile, mPtr<mColourLookup> &colourLookup, const mString &filename);

#endif // mColourLookup_h__
