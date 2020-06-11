#ifndef mFileTransacted_h__
#define mFileTransacted_h__

#include "mediaLib.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "f54yUWD3jKhMN4ydMoYsNxZjjlHV5SEo4bN9Zouwt1ydgzLcbCbFHs/cpUOsihLrXWwbucwSHhYf3SCp"
#endif

//////////////////////////////////////////////////////////////////////////

struct mFileTransaction;

mFUNCTION(mFileTransaction_Create, OUT mPtr<mFileTransaction> *pTransaction, IN mAllocator *pAllocator);
mFUNCTION(mFileTransaction_Destroy, IN_OUT mPtr<mFileTransaction> *pTransaction);

mFUNCTION(mFileTransaction_Perform, mPtr<mFileTransaction> &transaction);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mFileTransaction_CreateDirectory, mPtr<mFileTransaction> &transaction, const mString &directoryPath);

#endif // mFileTransacted_h__
