#include "mTestLib.h"
#include "mResourceManager.h"
#include "mQueue.h"

mAllocator *pTestAllocator = nullptr;
int64_t creationCount[64] = { 0 };

struct mResourceManagerTestNumberContainer
{
  size_t index;
};

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

mFUNCTION(mCreateResource, OUT mResourceManagerTestNumberContainer *pData, size_t index)
{
  mFUNCTION_SETUP();

  ++creationCount[index];
  pData->index = index;

  mRETURN_SUCCESS();
}

mFUNCTION(mDestroyResource, IN_OUT mResourceManagerTestNumberContainer *pData)
{
  mFUNCTION_SETUP();

  --creationCount[pData->index];

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

mTEST(mResourceManager, SingleDestructionTest)
{
  mTEST_ALLOCATOR_SETUP();

  pTestAllocator = pAllocator;
  mDEFER(pTestAllocator = nullptr);

  mTEST_ASSERT_SUCCESS((mResourceManager_CreateResourceManager_Explicit<size_t, mResourceManagerTestNumberContainer>)(pAllocator));

  mPtr<mQueue<mPtr<mResourceManagerTestNumberContainer>>> queue;
  mDEFER_CALL(&queue, mQueue_Destroy);
  mTEST_ASSERT_SUCCESS(mQueue_Create(&queue, pAllocator));

  {
    for (size_t i = 0; i < mARRAYSIZE(creationCount); i++)
    {
      for (size_t j = 0; j <= i; j++)
      {
        mPtr<mResourceManagerTestNumberContainer> data;
        mDEFER_CALL(&data, mSharedPointer_Destroy);
        mTEST_ASSERT_SUCCESS(mResourceManager_GetResource(&data, i));

        mTEST_ASSERT_SUCCESS(mQueue_PushBack(queue, data));
      }
    }

    for (size_t i = 0; i < mARRAYSIZE(creationCount); i++)
      mTEST_ASSERT_EQUAL(1, creationCount[i]);

    for (size_t i = 0; i < mARRAYSIZE(creationCount) / 2; i++)
    {
      mPtr<mResourceManagerTestNumberContainer> data;
      mDEFER_CALL(&data, mSharedPointer_Destroy);
      mTEST_ASSERT_SUCCESS(mQueue_PopFront(queue, &data));
    }

    mTEST_ASSERT_SUCCESS(mQueue_Destroy(&queue));

    for (size_t i = 0; i < mARRAYSIZE(creationCount); i++)
      mTEST_ASSERT_EQUAL(0, creationCount[i]);
  }

  mTEST_ASSERT_SUCCESS(mSharedPointer_Destroy(&mResourceManager<mResourceManagerTestNumberContainer, size_t>::Instance()));

  mTEST_ALLOCATOR_ZERO_CHECK();
}
