#include "mSemaphore.h"

#include <condition_variable>
#include <mutex>

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "nxmy96Ja8aLmwKT+/v62RP14meo3etMdz8lHZ8WGUZ7HSSY61pxX/Dl4bh8sPbx0T8qiznS1eEGL3+DH"
#endif

struct mSemaphore
{
  std::mutex mutex;
  std::condition_variable conditionVariable;
  mAllocator *pAllocator;
};

static mFUNCTION(mSemaphore_Destroy_Internal, IN mSemaphore *pSemaphore);

mFUNCTION(mSemaphore_Create, OUT mSemaphore **ppSemaphore, IN OPTIONAL mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppSemaphore == nullptr, mR_ArgumentNull);

  mDEFER_CALL_ON_ERROR(ppSemaphore, mSetToNullptr);
  mDEFER_ON_ERROR(mAllocator_FreePtr(pAllocator, ppSemaphore));
  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, ppSemaphore, 1));

  (*ppSemaphore)->pAllocator = pAllocator;

  new (&(*ppSemaphore)->mutex) std::mutex();
  new (&(*ppSemaphore)->conditionVariable) std::condition_variable();

  mRETURN_SUCCESS();
}

mFUNCTION(mSemaphore_Destroy, IN_OUT mSemaphore **ppSemaphore)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppSemaphore == nullptr, mR_ArgumentNull);

  mDEFER_CALL(ppSemaphore, mSetToNullptr);
  mERROR_CHECK(mSemaphore_Destroy_Internal(*ppSemaphore));

  mAllocator *pAllocator = (*ppSemaphore)->pAllocator;
  mERROR_CHECK(mAllocator_FreePtr(pAllocator, ppSemaphore));

  mRETURN_SUCCESS();
}

mFUNCTION(mSemaphore_Lock, IN mSemaphore *pSemaphore)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSemaphore == nullptr, mR_ArgumentNull);

  pSemaphore->mutex.lock();

  mRETURN_SUCCESS();
}

mFUNCTION(mSemaphore_Unlock, IN mSemaphore *pSemaphore)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSemaphore == nullptr, mR_ArgumentNull);

  pSemaphore->mutex.unlock();

  mRETURN_SUCCESS();
}

mFUNCTION(mSemaphore_WakeOne, IN mSemaphore *pSemaphore)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSemaphore == nullptr, mR_ArgumentNull);

  pSemaphore->conditionVariable.notify_one();

  mRETURN_SUCCESS();
}

mFUNCTION(mSemaphore_WakeAll, IN mSemaphore *pSemaphore)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSemaphore == nullptr, mR_ArgumentNull);

  pSemaphore->conditionVariable.notify_all();

  mRETURN_SUCCESS();
}

mFUNCTION(mSemaphore_Sleep, IN mSemaphore *pSemaphore, const size_t timeoutMilliseconds /* = mSemaphore_SleepTime::mS_ST_Infinite */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSemaphore == nullptr, mR_ArgumentNull);

  std::unique_lock<std::mutex> lock(pSemaphore->mutex);

  if (timeoutMilliseconds == mSemaphore_SleepTime::mS_ST_Infinite)
    pSemaphore->conditionVariable.wait(lock);
  else
  {
    std::cv_status result = pSemaphore->conditionVariable.wait_for(lock, std::chrono::milliseconds(timeoutMilliseconds));

    if (result == std::cv_status::no_timeout)
      mRETURN_SUCCESS();
    else
      mRETURN_RESULT(mR_Timeout);
  }

  mRETURN_SUCCESS();
}

static mFUNCTION(mSemaphore_Destroy_Internal, IN mSemaphore *pSemaphore)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSemaphore == nullptr, mR_ArgumentNull);

  pSemaphore->conditionVariable.~condition_variable();
  pSemaphore->mutex.~mutex();

  mRETURN_SUCCESS();
}
