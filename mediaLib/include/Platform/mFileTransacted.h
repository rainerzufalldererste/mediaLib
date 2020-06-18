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
mFUNCTION(mFileTransaction_WriteFile, mPtr<mFileTransaction> &transaction, const mString &filename, const uint8_t *pData, const size_t bytes);
mFUNCTION(mFileTransaction_DeleteFile, mPtr<mFileTransaction> &transaction, const mString &filename);
mFUNCTION(mFileTransaction_CopyFile, mPtr<mFileTransaction> &transaction, const mString &target, const mString &source, const bool replaceIfExistent);
mFUNCTION(mFileTransaction_MoveFile, mPtr<mFileTransaction> &transaction, const mString &target, const mString &source, const bool replaceIfExistent);

mFUNCTION(mFileTransaction_WriteRegistryKey, mPtr<mFileTransaction> &transaction, const mString &keyUrl, const mString &value, OUT OPTIONAL bool *pNewlyCreated = nullptr);

#endif // mFileTransacted_h__
