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

//////////////////////////////////////////////////////////////////////////

struct mSharedMutex
{
  mAllocator *pAllocator;
  HANDLE handle;
  mSharedMutex_CreationFlags creationFlags;
};

static mFUNCTION(mSharedMutex_Destroy_Internal, IN mSharedMutex *pMutex);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mSharedMutex_Create, OUT mSharedMutex **ppMutex, IN OPTIONAL mAllocator *pAllocator, const mString &name, const mSharedMutex_CreationFlags flags /* = mSM_CF_Default */)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppMutex == nullptr, mR_ArgumentNull);
  mERROR_IF(!!(flags & mSM_CF_Unnamed) && !!(flags & mSM_CF_Global), mR_InvalidParameter);
  mERROR_IF((name.hasFailed || name.bytes <= 1) && !(flags & mSM_CF_Unnamed), mR_InvalidParameter);

  mString properName;
  wchar_t wName[MAX_PATH + 1];

  if (!(flags & mSM_CF_Unnamed))
  {
    if ((flags & mSM_CF_Local) == mSM_CF_Local)
      mERROR_CHECK(mString_Format(&properName, &mDefaultTempAllocator, "Local\\", name));
    else
      mERROR_CHECK(mString_Format(&properName, &mDefaultTempAllocator, "Global\\", name));

    mERROR_CHECK(mString_ToWideString(properName, wName, mARRAYSIZE(wName)));
  }

  const HANDLE handle = CreateMutexExW(nullptr, !!(flags & mSM_CF_Unnamed) ? nullptr : wName, CREATE_MUTEX_INITIAL_OWNER * !!(flags & mSM_CF_InitiallyLocked), SYNCHRONIZE);

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

  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, ppMutex, 1));
  mDEFER_ON_ERROR(mAllocator_FreePtr(pAllocator, ppMutex));

  (*ppMutex)->pAllocator = pAllocator;
  (*ppMutex)->creationFlags = flags;
  (*ppMutex)->handle = handle;

  mRETURN_SUCCESS();
}

mFUNCTION(mSharedMutex_OpenExisting, OUT mSharedMutex **ppMutex, IN OPTIONAL mAllocator *pAllocator, const mString &name, const bool isGlobal /* = false */)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppMutex == nullptr, mR_ArgumentNull);
  mERROR_IF(name.hasFailed || name.bytes <= 1, mR_InvalidParameter);

  mString properName;

  if (!isGlobal)
    mERROR_CHECK(mString_Format(&properName, &mDefaultTempAllocator, "Local\\", name));
  else
    mERROR_CHECK(mString_Format(&properName, &mDefaultTempAllocator, "Global\\", name));

  wchar_t wName[MAX_PATH + 1];
  mERROR_CHECK(mString_ToWideString(properName, wName, mARRAYSIZE(wName)));

  const HANDLE handle = OpenMutexW(SYNCHRONIZE, FALSE, wName);

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

  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, ppMutex, 1));
  mDEFER_ON_ERROR(mAllocator_FreePtr(pAllocator, ppMutex));

  (*ppMutex)->pAllocator = pAllocator;
  (*ppMutex)->creationFlags = isGlobal ? mSM_CF_Global : mSM_CF_Local;
  (*ppMutex)->handle = handle;

  mRETURN_SUCCESS();
}

mFUNCTION(mSharedMutex_Destroy, IN_OUT mSharedMutex **ppMutex)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppMutex == nullptr, mR_ArgumentNull);
  mERROR_IF(*ppMutex == nullptr, mR_Success);

  mDEFER_CALL(ppMutex, mSetToNullptr);
  mERROR_CHECK(mSharedMutex_Destroy_Internal(*ppMutex));

  mAllocator *pAllocator = (*ppMutex)->pAllocator;
  mERROR_CHECK(mAllocator_FreePtr(pAllocator, ppMutex));

  mRETURN_SUCCESS();
}

mFUNCTION(mSharedMutex_Lock, IN mSharedMutex *pMutex, const size_t timeoutMs /* = (size_t)-1 */, OUT OPTIONAL bool *pLastAbandoned /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pMutex == nullptr, mR_ArgumentNull);
  mERROR_IF(timeoutMs != (size_t)-1 && timeoutMs > MAXDWORD, mR_ArgumentOutOfBounds);

  if (pLastAbandoned != nullptr)
    *pLastAbandoned = false;

  const DWORD timeout = timeoutMs == (size_t)-1 ? MAXDWORD : (DWORD)timeoutMs;
  const DWORD result = WaitForSingleObject(pMutex->handle, timeout);

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

mFUNCTION(mSharedMutex_Unlock, IN mSharedMutex *pMutex)
{
  mFUNCTION_SETUP();

  mERROR_IF(pMutex == nullptr, mR_ArgumentNull);

  mERROR_IF(0 == ReleaseMutex(pMutex->handle), mR_InternalError);

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

static mFUNCTION(mSharedMutex_Destroy_Internal, IN mSharedMutex *pMutex)
{
  mFUNCTION_SETUP();

  mERROR_IF(pMutex == nullptr, mR_ArgumentNull);

  if (pMutex->handle != nullptr)
    CloseHandle(pMutex->handle);

  mRETURN_SUCCESS();
}
