#ifndef mPipe_h__
#define mPipe_h__

#include "mediaLib.h"

struct mPipe;

mFUNCTION(mPipe_Create, OUT mPtr<mPipe> *pPipe, IN mAllocator *pAllocator);
mFUNCTION(mPipe_Destroy, OUT mPtr<mPipe> *pPipe);
mFUNCTION(mPipe_Read, mPtr<mPipe> &pipe, OUT uint8_t *pData, const size_t capacity, OUT size_t *pLength);
mFUNCTION(mPipe_Write, mPtr<mPipe> &pipe, IN uint8_t *pData, const size_t length, OUT OPTIONAL size_t *pBytesWritten = nullptr);

#ifdef mPLATFORM_WINDOWS
mFUNCTION(mPipe_GetReadHandle, mPtr<mPipe> &pipe, OUT HANDLE *pReadHandle);
mFUNCTION(mPipe_GetWriteHandle, mPtr<mPipe> &pipe, OUT HANDLE *pWriteHandle);
#endif

#endif // mPipe_h__
