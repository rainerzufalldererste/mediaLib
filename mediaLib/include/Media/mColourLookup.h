#ifndef mColourLookup_h__
#define mColourLookup_h__

#include "mediaLib.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "K9UDn4f1ync6Hggo9VeTVsQHOW+cWIfsuaQHa/idteBoJIjUtl5S7sM8xDtiplWFNdT4qh+OtecHIeer"
#endif

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
mFUNCTION(mColourLookup_At, mPtr<mColourLookup> &colourLookup, const mVec3f position, OUT mVec3f *pColour);

mFUNCTION(mColourLookup_WriteToFile, mPtr<mColourLookup> &colourLookup, const mString &filename);

#endif // mColourLookup_h__
