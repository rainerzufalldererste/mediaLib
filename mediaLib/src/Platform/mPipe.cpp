#include "mPipe.h"

struct mPipe
{
  HANDLE read, write;
};

mFUNCTION(mPipe_Destroy_Internal, IN_OUT mPipe *pPipe);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mPipe_Create, OUT mPtr<mPipe> *pPipe, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pPipe == nullptr, mR_ArgumentNull);

  SECURITY_ATTRIBUTES securityAttrib;
  securityAttrib.nLength = sizeof(SECURITY_ATTRIBUTES);
  securityAttrib.bInheritHandle = TRUE;
  securityAttrib.lpSecurityDescriptor = nullptr;

  HANDLE read = nullptr;
  HANDLE write = nullptr;

  mDEFER_ON_ERROR(
    CloseHandle(read);
    CloseHandle(write);
  );

  if (FALSE == CreatePipe(&read, &write, &securityAttrib, 0))
  {
    const DWORD error = GetLastError();
    mUnused(error);

    mRETURN_RESULT(mR_InternalError);
  }

  mERROR_IF(FALSE == SetHandleInformation(read, HANDLE_FLAG_INHERIT, 0), mR_InternalError);

  mERROR_CHECK(mSharedPointer_Allocate<mPipe>(pPipe, pAllocator, [](mPipe *pData) { mPipe_Destroy_Internal(pData); }, 1));

  (*pPipe)->read = read;
  (*pPipe)->write = write;

  mRETURN_SUCCESS();
}

mFUNCTION(mPipe_Destroy, OUT mPtr<mPipe> *pPipe)
{
  mFUNCTION_SETUP();

  mERROR_IF(pPipe == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Destroy(pPipe));

  mRETURN_SUCCESS();
}

mFUNCTION(mPipe_Read, mPtr<mPipe> &pipe, OUT uint8_t *pData, const size_t capacity, OUT size_t *pLength)
{
  mFUNCTION_SETUP();

  mERROR_IF(pipe == nullptr || pData == nullptr || pLength == nullptr, mR_ArgumentNull);

  DWORD bytesAvailable = 0;

  if (FALSE == PeekNamedPipe(pipe->read, nullptr, 0, nullptr, &bytesAvailable, nullptr))
  {
    *pLength = 0;

    const DWORD error = GetLastError();
    mUnused(error);

    mRETURN_RESULT(mR_InternalError);
  }

  if (bytesAvailable == 0)
  {
    *pLength = 0;
    mRETURN_SUCCESS();
  }

  DWORD bytesRead = 0;

  if (FALSE == ReadFile(pipe->read, pData, (DWORD)mMin(capacity, (size_t)MAXDWORD), &bytesRead, NULL))
  {
    *pLength = bytesRead;

    const DWORD error = GetLastError();
    mUnused(error);

    mRETURN_RESULT(mR_InternalError);
  }

  *pLength = bytesRead;

  mRETURN_SUCCESS();
}

mFUNCTION(mPipe_Write, mPtr<mPipe> &pipe, IN uint8_t *pData, const size_t length, OUT OPTIONAL size_t *pBytesWritten /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pipe == nullptr || pData == nullptr, mR_ArgumentNull);
  mERROR_IF(length > MAXDWORD, mR_ArgumentOutOfBounds);

  DWORD bytesWritten = 0;

  if (FALSE == WriteFile(pipe->write, pData, (DWORD)length, &bytesWritten, nullptr))
  {
    if (pBytesWritten != nullptr)
      *pBytesWritten = bytesWritten;

    const DWORD error = GetLastError();
    mUnused(error);

    mRETURN_RESULT(mR_InternalError);
  }

  if (pBytesWritten != nullptr)
    *pBytesWritten = bytesWritten;

  mRETURN_SUCCESS();
}

#ifdef mPLATFORM_WINDOWS

mFUNCTION(mPipe_GetReadHandle, mPtr<mPipe> &pipe, OUT HANDLE *pReadHandle)
{
  mFUNCTION_SETUP();

  mERROR_IF(pipe == nullptr || pReadHandle == nullptr, mR_ArgumentNull);

  *pReadHandle = pipe->read;

  mRETURN_SUCCESS();
}

mFUNCTION(mPipe_GetWriteHandle, mPtr<mPipe> &pipe, OUT HANDLE *pWriteHandle)
{
  mFUNCTION_SETUP();

  mERROR_IF(pipe == nullptr || pWriteHandle == nullptr, mR_ArgumentNull);

  *pWriteHandle = pipe->write;

  mRETURN_SUCCESS();
}

#endif

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mPipe_Destroy_Internal, IN_OUT mPipe *pPipe)
{
  mFUNCTION_SETUP();

  mERROR_IF(pPipe == nullptr, mR_ArgumentNull);

  CloseHandle(pPipe->read);
  CloseHandle(pPipe->write);

  mRETURN_SUCCESS();
}
