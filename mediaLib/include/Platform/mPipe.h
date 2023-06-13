#ifndef mPipe_h__
#define mPipe_h__

#include "mediaLib.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "QVXfL9u4qirQRs94cAla64SQDSmmoE3GZs1r/8TiwctArrW4E+CENBJ8ArLp3GfCJg37rMlYmfPQV8Do"
#endif

struct mPipe;

mFUNCTION(mPipe_Create, OUT mPtr<mPipe> *pPipe, IN mAllocator *pAllocator);
mFUNCTION(mPipe_Create, OUT mPtr<mPipe> *pPipe, IN mAllocator *pAllocator, const size_t bufferSize, const size_t timeoutMs = 30 * 1000);
mFUNCTION(mPipe_Destroy, OUT mPtr<mPipe> *pPipe);
mFUNCTION(mPipe_Read, mPtr<mPipe> &pipe, OUT uint8_t *pData, const size_t capacity, OUT size_t *pLength);
mFUNCTION(mPipe_Write, mPtr<mPipe> &pipe, IN uint8_t *pData, const size_t length, OUT OPTIONAL size_t *pBytesWritten = nullptr);

#ifdef mPLATFORM_WINDOWS
mFUNCTION(mPipe_GetReadHandle, mPtr<mPipe> &pipe, OUT HANDLE *pReadHandle);
mFUNCTION(mPipe_GetWriteHandle, mPtr<mPipe> &pipe, OUT HANDLE *pWriteHandle);
#endif

#endif // mPipe_h__
