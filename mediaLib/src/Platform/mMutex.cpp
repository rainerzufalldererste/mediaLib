#include "mMutex.h"
#include <mutex>

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "5hAQ/qRmTft9l2339rKOdVCCrnrJRHXyroLFjabrQU7GRFZcPwCmqAy+dQI5bCr7oJD0PX6cGFKTiblQ"
#endif

struct mMutex
{
  std::mutex lock;
  mAllocator *pAllocator;
};

static mFUNCTION(mMutex_Destroy_Internal, IN mMutex *pMutex);

//////////////////////////////////////////////////////////////////////////

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
  mERROR_IF(*ppMutex == nullptr, mR_Success);

  mDEFER_CALL(ppMutex, mSetToNullptr);
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

//////////////////////////////////////////////////////////////////////////

static mFUNCTION(mMutex_Destroy_Internal, IN mMutex *pMutex)
{
  mFUNCTION_SETUP();

  mERROR_IF(pMutex == nullptr, mR_ArgumentNull);
  
  pMutex->lock.~mutex();

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

struct mRecursiveMutex
{
  std::recursive_mutex lock;
  mAllocator *pAllocator;
};

static mFUNCTION(mRecursiveMutex_Destroy_Internal, IN mRecursiveMutex *pMutex);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mRecursiveMutex_Create, OUT mRecursiveMutex **ppMutex, IN OPTIONAL mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppMutex == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, ppMutex, 1));
  mDEFER_ON_ERROR(mAllocator_FreePtr(pAllocator, ppMutex));

  (*ppMutex)->pAllocator = pAllocator;
  new (&(*ppMutex)->lock) std::recursive_mutex();

  mRETURN_SUCCESS();
}

mFUNCTION(mRecursiveMutex_Destroy, IN_OUT mRecursiveMutex **ppMutex)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppMutex == nullptr, mR_ArgumentNull);
  mERROR_IF(*ppMutex == nullptr, mR_Success);

  mDEFER_CALL(ppMutex, mSetToNullptr);
  mERROR_CHECK(mRecursiveMutex_Destroy_Internal(*ppMutex));

  mAllocator *pAllocator = (*ppMutex)->pAllocator;
  mERROR_CHECK(mAllocator_FreePtr(pAllocator, ppMutex));

  mRETURN_SUCCESS();
}

mFUNCTION(mRecursiveMutex_Lock, IN mRecursiveMutex *pMutex)
{
  mFUNCTION_SETUP();

  mERROR_IF(pMutex == nullptr, mR_ArgumentNull);
  pMutex->lock.lock();

  mRETURN_SUCCESS();
}

mFUNCTION(mRecursiveMutex_Unlock, IN mRecursiveMutex *pMutex)
{
  mFUNCTION_SETUP();

  mERROR_IF(pMutex == nullptr, mR_ArgumentNull);
  pMutex->lock.unlock();

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

static mFUNCTION(mRecursiveMutex_Destroy_Internal, IN mRecursiveMutex *pMutex)
{
  mFUNCTION_SETUP();

  mERROR_IF(pMutex == nullptr, mR_ArgumentNull);

  pMutex->lock.~recursive_mutex();

  mRETURN_SUCCESS();
}
