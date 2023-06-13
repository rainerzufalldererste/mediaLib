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

//////////////////////////////////////////////////////////////////////////

mTEST(mResourceManager, CleanupTest)
{
  mTEST_ALLOCATOR_SETUP();

  pTestAllocator = pAllocator;
  mDEFER(pTestAllocator = nullptr);

  mTEST_ASSERT_SUCCESS((mResourceManager_CreateResourceManager_Explicit<mString, mDummyDestructible>)(pAllocator));
  
  {
    mString key;
    mTEST_ASSERT_SUCCESS(mString_Create(&key, "don't care", pAllocator));

    mPtr<mDummyDestructible> dummyDestructible0;
    mTEST_ASSERT_SUCCESS(mResourceManager_GetResource(&dummyDestructible0, key));

    mString keyCopy = mString(key);
    mPtr<mDummyDestructible> dummyDestructible1;
    mTEST_ASSERT_SUCCESS(mResourceManager_GetResource(&dummyDestructible1, keyCopy));

    mTEST_ASSERT_EQUAL(dummyDestructible0->index, dummyDestructible1->index);

    mString key2;
    mTEST_ASSERT_SUCCESS(mString_Create(&key2, "still don't care", pAllocator));

    mPtr<mDummyDestructible> dummyDestructible2;
    mTEST_ASSERT_SUCCESS(mResourceManager_GetResource(&dummyDestructible2, key2));
  }

  mTEST_ASSERT_SUCCESS(mSharedPointer_Destroy(&mResourceManager<mDummyDestructible, mString>::Instance()));

  mTEST_ALLOCATOR_ZERO_CHECK();
}
