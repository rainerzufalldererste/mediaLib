// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "mAllocator.h"

mFUNCTION(mDefaultAllocator_Alloc, OUT uint8_t **ppData, const size_t size, const size_t count, IN void *)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mAlloc(ppData, size * count));

  mRETURN_SUCCESS();
}

mFUNCTION(mDefaultAllocator_Free, OUT uint8_t *pData, IN void *)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mFree(pData));

  mRETURN_SUCCESS();
}

mFUNCTION(mDefaultAllocator_Move, IN_OUT uint8_t *pDestimation, IN uint8_t *pSource, const size_t size, const size_t count, IN void *)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mMemmove(pDestimation, pSource, size * count));

  mRETURN_SUCCESS();
}

mFUNCTION(mDefaultAllocator_Copy, IN_OUT uint8_t *pDestimation, IN uint8_t *pSource, const size_t size, const size_t count, IN void *)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mMemcpy(pDestimation, pSource, size * count));

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mAllocator_Create, OUT mAllocator *pAllocator, IN mAllocator_AllocFunction *pAllocFunction, IN mAllocator_FreeFunction *pFree, IN OPTIONAL mAllocator_MoveFunction *pMove /* = nullptr */, IN OPTIONAL mAllocator_CopyFunction *pCopy /* = nullptr */, IN OPTIONAL mAllocator_DestroyAllocator *pDestroyAllocator /* = nullptr */, IN OPTIONAL void *pUserData /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAllocator == nullptr || pAllocFunction == nullptr || pFree == nullptr, mR_ArgumentNull);

  pAllocator->pAllocate = pAllocFunction;
  pAllocator->pFree = pFree;
  pAllocator->pMove = pMove;
  pAllocator->pCopy = pCopy;
  pAllocator->pDestroyAllocator = pDestroyAllocator;
  pAllocator->pUserData = pUserData;

  pAllocator->initialized = true;

  mRETURN_SUCCESS();
}

mFUNCTION(mAllocator_Destoy, IN_OUT mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAllocator == nullptr, mR_ArgumentNull);

  pAllocator->initialized = false;
  pAllocator->pDestroyAllocator(pAllocator->pUserData);

  mRETURN_SUCCESS();
}
