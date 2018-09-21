// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "mTestLib.h"
#include "mResourceManager.h"

mAllocator *pTestAllocator = nullptr;

mFUNCTION(mCreateResource, OUT mDummyDestructible *pDummy, mString &)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mDummyDestructible_Create(pDummy, pTestAllocator));

  mRETURN_SUCCESS();
}

mFUNCTION(mDestroyResource, IN_OUT mDummyDestructible *pDummy)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mDestruct(pDummy));

  mRETURN_SUCCESS();
}

mTEST(mResourceManager, CleanupTest)
{
  mTEST_ALLOCATOR_SETUP();

  pTestAllocator = pAllocator;
  mDEFER(pTestAllocator = nullptr);

  mTEST_ASSERT_SUCCESS((mResourceManager_CreateResourceManager_Explicit<mString, mDummyDestructible>)(pAllocator));
  
  mPtr<mDummyDestructible> dummyDestructible0;
  mTEST_ASSERT_SUCCESS(mResourceManager_GetResource(&dummyDestructible0, (mString)"don't care"));

  mPtr<mDummyDestructible> dummyDestructible1;
  mTEST_ASSERT_SUCCESS(mResourceManager_GetResource(&dummyDestructible1, (mString)"don't care"));

  mTEST_ASSERT_EQUAL(dummyDestructible0->index, dummyDestructible1->index);

  mPtr<mDummyDestructible> dummyDestructible2;
  mTEST_ASSERT_SUCCESS(mResourceManager_GetResource(&dummyDestructible2, (mString)"still don't care"));

  mTEST_ASSERT_SUCCESS(mSharedPointer_Destroy(&mResourceManager<mDummyDestructible, mString>::Instance()));

  mTEST_ALLOCATOR_ZERO_CHECK();
}
