#ifndef mSemaphore_h__
#define mSemaphore_h__

#include "mediaLib.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "1A5xPftfV/GgFOCD2f2OiiABg5KcV/hib43wtFhXrXIOhs43SA++ni7VcGDDQPno7CTVBh9ByBwphrKf"
#endif

enum mSemaphore_SleepTime : size_t
{
  mS_ST_Infinite = 0xFFFFFFFF,
};

struct mSemaphore;

mFUNCTION(mSemaphore_Create, OUT mSemaphore **ppSemaphore, IN OPTIONAL mAllocator *pAllocator);
mFUNCTION(mSemaphore_Destroy, IN_OUT mSemaphore **ppSemaphore);

mFUNCTION(mSemaphore_Lock, IN mSemaphore *pSemaphore);
mFUNCTION(mSemaphore_Unlock, IN mSemaphore *pSemaphore);
mFUNCTION(mSemaphore_WakeOne, IN mSemaphore *pSemaphore);
mFUNCTION(mSemaphore_WakeAll, IN mSemaphore *pSemaphore);
mFUNCTION(mSemaphore_Sleep, IN mSemaphore *pSemaphore, const size_t timeoutMilliseconds = mSemaphore_SleepTime::mS_ST_Infinite);

struct mSharedSemaphore;

enum mSharedSemaphore_CreationFlags_ : size_t
{
  mSS_CF_Local = 0,
  mSS_CF_Global = 1,
  mSS_CF_Unnamed = 1 << 1, // only valid if not `mSM_CF_Global`.

  mSS_CF_Default = mSS_CF_Local,
};

typedef size_t mSharedSemaphore_CreationFlags;

mFUNCTION(mSharedSemaphore_Create, OUT mSharedSemaphore **ppSemaphore, IN OPTIONAL mAllocator *pAllocator, const mString &name, const mSharedSemaphore_CreationFlags flags = mSS_CF_Default, const size_t initialCount = 0, const size_t maximumCount = 1);
mFUNCTION(mSharedSemaphore_OpenExisting, OUT mSharedSemaphore **ppSemaphore, IN OPTIONAL mAllocator *pAllocator, const mString &name, const bool isGlobal = false);
mFUNCTION(mSharedSemaphore_Destroy, IN_OUT mSharedSemaphore **ppSemaphore);

mFUNCTION(mSharedSemaphore_Lock, IN mSharedSemaphore *pSemaphore, const size_t timeoutMs = (size_t)-1, OUT OPTIONAL bool *pLastAbandoned = nullptr);
mFUNCTION(mSharedSemaphore_Unlock, IN mSharedSemaphore *pSemaphore, const size_t amountToRelease = 1, OPTIONAL OUT size_t *pPreviousCount = nullptr);

#endif // mSemaphore_h__
