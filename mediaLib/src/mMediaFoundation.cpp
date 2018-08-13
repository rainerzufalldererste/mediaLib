// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "mMediaFoundation.h"
#include <mfapi.h>

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
