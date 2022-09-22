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

//////////////////////////////////////////////////////////////////////////

struct mSharedSemaphore
{
  mAllocator *pAllocator;
  HANDLE handle;
  mSharedSemaphore_CreationFlags creationFlags;
};

static mFUNCTION(mSharedSemaphore_Destroy_Internal, IN mSharedSemaphore *pMutex);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mSharedSemaphore_Create, OUT mSharedSemaphore **ppSemaphore, IN OPTIONAL mAllocator *pAllocator, const mString &name, const mSharedSemaphore_CreationFlags flags /* = mSS_CF_Default */, const size_t initialCount /* = 0 */, const size_t maximumCount /* = 1 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppSemaphore == nullptr, mR_ArgumentNull);
  mERROR_IF(!!(flags & mSS_CF_Unnamed) && !!(flags & mSS_CF_Global), mR_InvalidParameter);
  mERROR_IF((name.hasFailed || name.bytes <= 1) && !(flags & mSS_CF_Unnamed), mR_InvalidParameter);
  mERROR_IF(initialCount > LONG_MAX || maximumCount > LONG_MAX, mR_InvalidParameter);

  mString properName;
  wchar_t wName[MAX_PATH + 1];

  if (!(flags & mSS_CF_Unnamed))
  {
    if ((flags & mSS_CF_Local) == mSS_CF_Local)
      mERROR_CHECK(mString_Format(&properName, &mDefaultTempAllocator, "Local\\", name));
    else
      mERROR_CHECK(mString_Format(&properName, &mDefaultTempAllocator, "Global\\", name));

    mERROR_CHECK(mString_ToWideString(properName, wName, mARRAYSIZE(wName)));
  }

  const HANDLE handle = CreateSemaphoreExW(nullptr, (LONG)initialCount, (LONG)maximumCount, !!(flags & mSS_CF_Unnamed) ? nullptr : wName, 0, SYNCHRONIZE | SEMAPHORE_MODIFY_STATE);

  if (handle == nullptr)
  {
    switch (GetLastError())
    {
    case ERROR_ACCESS_DENIED:
      mRETURN_RESULT(mR_InsufficientPrivileges);

    default:
      mRETURN_RESULT(mR_InternalError);
    }
  }

  mDEFER_CALL_ON_ERROR(handle, CloseHandle);

  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, ppSemaphore, 1));
  mDEFER_ON_ERROR(mAllocator_FreePtr(pAllocator, ppSemaphore));

  (*ppSemaphore)->pAllocator = pAllocator;
  (*ppSemaphore)->creationFlags = flags;
  (*ppSemaphore)->handle = handle;

  mRETURN_SUCCESS();
}

mFUNCTION(mSharedSemaphore_OpenExisting, OUT mSharedSemaphore **ppSemaphore, IN OPTIONAL mAllocator *pAllocator, const mString &name, const bool isGlobal /* = false */)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppSemaphore == nullptr, mR_ArgumentNull);
  mERROR_IF(name.hasFailed || name.bytes <= 1, mR_InvalidParameter);

  mString properName;

  if (!isGlobal)
    mERROR_CHECK(mString_Format(&properName, &mDefaultTempAllocator, "Local\\", name));
  else
    mERROR_CHECK(mString_Format(&properName, &mDefaultTempAllocator, "Global\\", name));

  wchar_t wName[MAX_PATH + 1];
  mERROR_CHECK(mString_ToWideString(properName, wName, mARRAYSIZE(wName)));

  const HANDLE handle = OpenSemaphoreW(SYNCHRONIZE | SEMAPHORE_MODIFY_STATE, FALSE, wName);

  if (handle == nullptr)
  {
    switch (GetLastError())
    {
    case ERROR_ACCESS_DENIED:
      mRETURN_RESULT(mR_InsufficientPrivileges);

    default:
      mRETURN_RESULT(mR_InternalError);
    }
  }

  mDEFER_CALL_ON_ERROR(handle, CloseHandle);

  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, ppSemaphore, 1));
  mDEFER_ON_ERROR(mAllocator_FreePtr(pAllocator, ppSemaphore));

  (*ppSemaphore)->pAllocator = pAllocator;
  (*ppSemaphore)->creationFlags = isGlobal ? mSS_CF_Global : mSS_CF_Local;
  (*ppSemaphore)->handle = handle;

  mRETURN_SUCCESS();
}

mFUNCTION(mSharedSemaphore_Destroy, IN_OUT mSharedSemaphore **ppSemaphore)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppSemaphore == nullptr, mR_ArgumentNull);
  mERROR_IF(*ppSemaphore == nullptr, mR_Success);

  mDEFER_CALL(ppSemaphore, mSetToNullptr);
  mERROR_CHECK(mSharedSemaphore_Destroy_Internal(*ppSemaphore));

  mAllocator *pAllocator = (*ppSemaphore)->pAllocator;
  mERROR_CHECK(mAllocator_FreePtr(pAllocator, ppSemaphore));

  mRETURN_SUCCESS();
}

mFUNCTION(mSharedSemaphore_Lock, IN mSharedSemaphore *pSemaphore, const size_t timeoutMs /* = (size_t)-1 */, OUT OPTIONAL bool *pLastAbandoned /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSemaphore == nullptr, mR_ArgumentNull);
  mERROR_IF(timeoutMs != (size_t)-1 && timeoutMs > MAXDWORD, mR_ArgumentOutOfBounds);

  if (pLastAbandoned != nullptr)
    *pLastAbandoned = false;

  const DWORD timeout = timeoutMs == (size_t)-1 ? MAXDWORD : (DWORD)timeoutMs;
  const DWORD result = WaitForSingleObject(pSemaphore->handle, timeout);

  switch (result)
  {
  case WAIT_ABANDONED:
  {
    if (pLastAbandoned != nullptr)
      *pLastAbandoned = true;

    break;
  }

  case WAIT_OBJECT_0:
    break;

  case WAIT_TIMEOUT:
    mRETURN_RESULT(mR_Timeout);

  case WAIT_FAILED:
  default:
    mRETURN_RESULT(mR_InternalError);
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mSharedSemaphore_Unlock, IN mSharedSemaphore *pSemaphore, const size_t amountToRelease /* = 1 */, OPTIONAL OUT size_t *pPreviousCount /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSemaphore == nullptr, mR_ArgumentNull);
  mERROR_IF(amountToRelease > LONG_MAX || amountToRelease == 0, mR_ArgumentOutOfBounds);

  LONG previousCount = 0;
  mERROR_IF(0 == ReleaseSemaphore(pSemaphore->handle, (LONG)amountToRelease, &previousCount), mR_InternalError);

  if (pPreviousCount != nullptr)
    *pPreviousCount = previousCount;

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

static mFUNCTION(mSharedSemaphore_Destroy_Internal, IN mSharedSemaphore *pSemaphore)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSemaphore == nullptr, mR_ArgumentNull);

  if (pSemaphore->handle != nullptr)
    CloseHandle(pSemaphore->handle);

  mRETURN_SUCCESS();
}
