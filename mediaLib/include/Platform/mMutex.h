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

struct mSharedMutex;

enum mSharedMutex_CreationFlags_ : size_t
{
  mSM_CF_Local = 0,
  mSM_CF_Global = 1,
  mSM_CF_InitiallyLocked = 1 << 1,
  mSM_CF_Unnamed = 1 << 2, // only valid if not `mSM_CF_Global`.

  mSM_CF_Default = mSM_CF_Local,
};

typedef size_t mSharedMutex_CreationFlags;

mFUNCTION(mSharedMutex_Create, OUT mSharedMutex **ppMutex, IN OPTIONAL mAllocator *pAllocator, const mString &name, const mSharedMutex_CreationFlags flags = mSM_CF_Default);
mFUNCTION(mSharedMutex_OpenExisting, OUT mSharedMutex **ppMutex, IN OPTIONAL mAllocator *pAllocator, const mString &name, const bool isGlobal = false);
mFUNCTION(mSharedMutex_Destroy, IN_OUT mSharedMutex **ppMutex);

mFUNCTION(mSharedMutex_Lock, IN mSharedMutex *pMutex, const size_t timeoutMs = (size_t)-1, OUT OPTIONAL bool *pLastAbandoned = nullptr);
mFUNCTION(mSharedMutex_Unlock, IN mSharedMutex *pMutex);

#endif // mMutex_h__
