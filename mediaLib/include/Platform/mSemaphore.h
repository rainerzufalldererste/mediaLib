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

#endif // mSemaphore_h__
