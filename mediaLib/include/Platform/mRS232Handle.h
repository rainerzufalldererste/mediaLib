#ifndef mRS232Handle_h__
#define mRS232Handle_h__

#include "mediaLib.h"
#include "mQueue.h"

struct mRS232Handle;

mFUNCTION(mRS232Handle_Create, OUT mRS232Handle **ppHandle, IN mAllocator *pAllocator, const char *name, const size_t baudrate);
mFUNCTION(mRS232Handle_Destroy, IN_OUT mRS232Handle **ppHandle);

mFUNCTION(mRS232Handle_SetTimeouts, IN mRS232Handle *pHandle, const uint32_t readIntervalTimeout = 0xFFFFFFFF, const uint32_t readTotalTimeoutMultiplier = 0, const uint32_t readTotalTimeoutConstant = 0, const uint32_t writeTotalTimeoutMultiplier = 0, const uint32_t writeTotalTimeoutConstant = 0);
mFUNCTION(mRS232Handle_FlushBuffers, IN mRS232Handle *pHandle, const bool flushRead = true, const bool flushWrite = true);

#if defined(mPLATFORM_WINDOWS)
mFUNCTION(mRS232Handle_GetPortsFromName, const char *name, OUT mPtr<mQueue<uint32_t>> *pPorts, IN mAllocator *pAllocator);
#endif

mFUNCTION(mRS232Handle_Read, mRS232Handle *pHandle, OUT uint8_t *pBuffer, const size_t length, OUT OPTIONAL size_t *pBytesReadCount);
mFUNCTION(mRS232Handle_Write, mRS232Handle *pHandle, IN const uint8_t *pBuffer, const size_t length, OUT OPTIONAL size_t *pBytesWriteCount);

#endif // mRS232Handle_h__
