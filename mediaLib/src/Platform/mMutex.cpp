// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "mMutex.h"
#include <mutex>

struct mMutex
{
  std::mutex lock;
  mAllocator *pAllocator;
};

mFUNCTION(mMutex_Destroy_Internal, IN mMutex *pMutex);

mFUNCTION(mMutex_Create, OUT mMutex **ppMutex, IN OPTIONAL mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppMutex == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, ppMutex, 1));
  mDEFER_ON_ERROR(mAllocator_FreePtr(pAllocator, ppMutex));

  (*ppMutex)->pAllocator = pAllocator;
  new (&(*ppMutex)->lock) std::mutex();

  mRETURN_SUCCESS();
}

mFUNCTION(mMutex_Destroy, IN_OUT mMutex **ppMutex)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppMutex == nullptr, mR_ArgumentNull);

  mDEFER_DESTRUCTION(ppMutex, mSetToNullptr);
  mERROR_CHECK(mMutex_Destroy_Internal(*ppMutex));

  mAllocator *pAllocator = (*ppMutex)->pAllocator;
  mERROR_CHECK(mAllocator_FreePtr(pAllocator, ppMutex));

  mRETURN_SUCCESS();
}

mFUNCTION(mMutex_Lock, IN mMutex *pMutex)
{
  mFUNCTION_SETUP();

  mERROR_IF(pMutex == nullptr, mR_ArgumentNull);
  pMutex->lock.lock();

  mRETURN_SUCCESS();
}

mFUNCTION(mMutex_Unlock, IN mMutex *pMutex)
{
  mFUNCTION_SETUP();

  mERROR_IF(pMutex == nullptr, mR_ArgumentNull);
  pMutex->lock.unlock();

  mRETURN_SUCCESS();
}

mFUNCTION(mMutex_Destroy_Internal, IN mMutex *pMutex)
{
  mFUNCTION_SETUP();

  mERROR_IF(pMutex == nullptr, mR_ArgumentNull);
  
  pMutex->lock.~mutex();

  mRETURN_SUCCESS();
}
