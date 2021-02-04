#ifndef mMutex_h__
#define mMutex_h__

#include "mediaLib.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "yNOtm4httszbIMhYl1tOfZ8MbPFmilFPDHdR16Z077hLP+XQkgVPN+6OalxZWPySKzIyj3h351dJsf87"
#endif

struct mMutex;

mFUNCTION(mMutex_Create, OUT mMutex **ppMutex, IN OPTIONAL mAllocator *pAllocator);
mFUNCTION(mMutex_Destroy, IN_OUT mMutex **ppMutex);

mFUNCTION(mMutex_Lock, IN mMutex *pMutex);
mFUNCTION(mMutex_Unlock, IN mMutex *pMutex);

struct mRecursiveMutex;

mFUNCTION(mRecursiveMutex_Create, OUT mRecursiveMutex **ppMutex, IN OPTIONAL mAllocator *pAllocator);
mFUNCTION(mRecursiveMutex_Destroy, IN_OUT mRecursiveMutex **ppMutex);

mFUNCTION(mRecursiveMutex_Lock, IN mRecursiveMutex *pMutex);
mFUNCTION(mRecursiveMutex_Unlock, IN mRecursiveMutex *pMutex);

#endif // mMutex_h__
