#include "mMediaFoundation.h"

#include <mfapi.h>

#pragma comment(lib, "dxva2.lib")
#pragma comment(lib, "evr.lib")
#pragma comment(lib, "mf.lib")
#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfplay.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "Ol78u6OZ9+erTZDLjGgfzgPvXn7NJyJy1r6HJzN7srpZPdcoCviaUI/f0dcC7AAW1PFvQGRC2ewkPwTU"
#endif

volatile size_t _mediaFoundationReferenceCount = 0;

mFUNCTION(mMediaFoundation_AddReference)
{
  mFUNCTION_SETUP();

  const size_t referenceCount = ++_mediaFoundationReferenceCount;

  if (referenceCount == 1)
  {
    HRESULT hr = S_OK;
    mUnused(hr);

    mERROR_IF(FAILED(hr = MFStartup(MF_VERSION, MFSTARTUP_FULL)), mR_InternalError);
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mMediaFoundation_RemoveReference)
{
  mFUNCTION_SETUP();

  mERROR_IF(_mediaFoundationReferenceCount == 0, mR_InvalidParameter);

  const size_t referenceCount = --_mediaFoundationReferenceCount;

  if (referenceCount == 0)
  {
    HRESULT hr = S_OK;
    mUnused(hr);

    mERROR_IF(FAILED(hr = MFShutdown()), mR_InternalError);
  }

  mRETURN_SUCCESS();
}
