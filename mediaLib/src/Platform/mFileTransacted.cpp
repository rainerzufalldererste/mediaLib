#include "mFileTransacted.h"

#include "mFile.h"

#include <ktmw32.h>

#pragma comment (lib, "KtmW32.lib")

#if (NTDDI_VERSION >= NTDDI_WIN8)
#include "pathcch.h"

#pragma comment(lib, "Pathcch.lib")
#endif

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "EoRawu1Wp6eVM6KH7CNc9TvnZd4za5XTxyCP+G50dVZurPiESRG8hgCOjZ5xkOvnjHxD9o/junjzIdkW"
#endif

//////////////////////////////////////////////////////////////////////////

struct mFileTransaction
{
  HANDLE transactionHandle;
  bool hasBeenTransacted;
};

mFUNCTION(mFileTransaction_Destroy_Internal, IN_OUT mFileTransaction *pTransaction);

mFUNCTION(_CreateDirectoryRecursive, mPtr<mFileTransaction> &transaction, const wchar_t *directoryPath);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mFileTransaction_Create, OUT mPtr<mFileTransaction> *pTransaction, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTransaction == nullptr, mR_ArgumentNull);

#ifdef mPLATFORM_WINDOWS
  HANDLE transactionHandle = CreateTransaction(nullptr, 0, 0, 0, 0, 0, L"");

  if (transactionHandle == INVALID_HANDLE_VALUE)
  {
    const DWORD error = GetLastError();
    mUnused(error);

    mRETURN_RESULT(mR_InternalError);
  }

  mDEFER_ON_ERROR(CloseHandle(transactionHandle));

  mERROR_CHECK((mSharedPointer_Allocate<mFileTransaction>(pTransaction, pAllocator, [](mFileTransaction *pData) { mFileTransaction_Destroy_Internal(pData); }, 1)));

  (*pTransaction)->transactionHandle = transactionHandle;
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mFileTransaction_Destroy, IN_OUT mPtr<mFileTransaction> *pTransaction)
{
  return mSharedPointer_Destroy(pTransaction);
}

mFUNCTION(mFileTransaction_Perform, mPtr<mFileTransaction> &transaction)
{
  mFUNCTION_SETUP();

  mERROR_IF(transaction == nullptr, mR_ArgumentNull);
  mERROR_IF(transaction->hasBeenTransacted, mR_ResourceStateInvalid);

  if (0 == CommitTransaction(transaction->transactionHandle))
  {
    const DWORD error = GetLastError();
    mUnused(error);

    mRETURN_RESULT(mR_InternalError);
  }
  else
  {
    transaction->hasBeenTransacted = true;
  }

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mFileTransaction_CreateDirectory, mPtr<mFileTransaction> &transaction, const mString &directoryPath)
{
  mFUNCTION_SETUP();

  mERROR_IF(transaction == nullptr, mR_ArgumentNull);
  mERROR_IF(transaction->hasBeenTransacted, mR_ResourceStateInvalid);
  mERROR_IF(directoryPath.hasFailed || directoryPath.bytes <= 1, mR_InvalidParameter);

  mString path;
  mERROR_CHECK(mFile_FailOnInvalidDirectoryPath(directoryPath, &path));

  wchar_t wDirectoryName[MAX_PATH];
  mERROR_CHECK(mString_ToWideString(path, wDirectoryName, mARRAYSIZE(wDirectoryName)));

  mERROR_CHECK(_CreateDirectoryRecursive(transaction, wDirectoryName));

  mRETURN_SUCCESS();
}

mFUNCTION(mFileTransaction_WriteFile, mPtr<mFileTransaction> &transaction, const mString &filename, const uint8_t *pData, const size_t bytes)
{
  mFUNCTION_SETUP();

  mERROR_IF(transaction == nullptr || pData == nullptr, mR_ArgumentNull);
  mERROR_IF(transaction->hasBeenTransacted, mR_ResourceStateInvalid);
  mERROR_IF(filename.hasFailed || filename.bytes <= 1, mR_InvalidParameter);

  wchar_t wFilename[MAX_PATH];
  mERROR_CHECK(mString_ToWideString(filename, wFilename, mARRAYSIZE(wFilename)));

  HANDLE fileHandle = CreateFileTransactedW(wFilename, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr, transaction->transactionHandle, nullptr, nullptr);
  DWORD error = GetLastError(); // Might be `ERROR_ALREADY_EXISTS` even if the `fileHandle` is valid.

  if (fileHandle == INVALID_HANDLE_VALUE && error == ERROR_PATH_NOT_FOUND)
  {
    wchar_t parentDirectory[MAX_PATH];
    mERROR_CHECK(mStringCopy(parentDirectory, mARRAYSIZE(parentDirectory), wFilename, MAX_PATH));

#if (NTDDI_VERSION >= NTDDI_WIN8)
    HRESULT hr = S_OK;

    // Requires `pathcch.h` && `Pathcch.lib`.
    mERROR_IF(FAILED(hr = PathCchRemoveFileSpec(parentDirectory, mARRAYSIZE(parentDirectory))), mR_InternalError);
    mERROR_IF(hr == S_FALSE, mR_InvalidParameter);
#else
    // deprecated since windows 8.
    mERROR_IF(PathRemoveFileSpecW(parentDirectory), mR_InvalidParameter);
#endif

    mERROR_CHECK(_CreateDirectoryRecursive(transaction, parentDirectory));
    
    fileHandle = CreateFileTransactedW(wFilename, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr, transaction->transactionHandle, nullptr, nullptr);
    error = GetLastError(); // Might be `ERROR_ALREADY_EXISTS` even if the `fileHandle` is valid.
  }

  mERROR_IF(fileHandle == INVALID_HANDLE_VALUE, mR_InternalError);
  mDEFER(CloseHandle(fileHandle));

  size_t bytesRemaining = bytes;
  size_t offset = 0;

  while (bytesRemaining > 0)
  {
    const DWORD bytesToWrite = (DWORD)mMin((size_t)MAXDWORD, bytesRemaining);
    DWORD bytesWritten = 0;

    if (0 == WriteFile(fileHandle, pData + offset, bytesToWrite, &bytesWritten, nullptr))
    {
      error = GetLastError();

      mRETURN_RESULT(mR_IOFailure);
    }

    mERROR_IF(bytesWritten == 0, mR_IOFailure);

    offset += bytesWritten;
    bytesRemaining -= bytesWritten;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mFileTransaction_DeleteFile, mPtr<mFileTransaction> &transaction, const mString &filename)
{
  mFUNCTION_SETUP();

  mERROR_IF(transaction == nullptr, mR_ArgumentNull);
  mERROR_IF(transaction->hasBeenTransacted, mR_ResourceStateInvalid);
  mERROR_IF(filename.hasFailed || filename.bytes <= 1, mR_InvalidParameter);

  wchar_t wFilename[MAX_PATH];
  mERROR_CHECK(mString_ToWideString(filename, wFilename, mARRAYSIZE(wFilename)));

  if (0 == DeleteFileTransactedW(wFilename, transaction->transactionHandle))
  {
    const DWORD error = GetLastError();

    mERROR_IF(error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND, mR_ResourceNotFound);
    mRETURN_RESULT(mR_InternalError);
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mFileTransaction_CopyFile, mPtr<mFileTransaction> &transaction, const mString &target, const mString &source)
{
  mFUNCTION_SETUP();

  mERROR_IF(transaction == nullptr, mR_ArgumentNull);
  mERROR_IF(transaction->hasBeenTransacted, mR_ResourceStateInvalid);
  mERROR_IF(target.hasFailed || source.hasFailed || target.bytes <= 1 || source.bytes <= 1, mR_InvalidParameter);

  wchar_t wSource[MAX_PATH];
  mERROR_CHECK(mString_ToWideString(source, wSource, mARRAYSIZE(wSource)));

  wchar_t wTarget[MAX_PATH];
  mERROR_CHECK(mString_ToWideString(target, wTarget, mARRAYSIZE(wSource)));

  if (0 == CopyFileTransactedW(wSource, wTarget, nullptr, nullptr, nullptr, 0, transaction->transactionHandle))
  {
    DWORD error = GetLastError();

    if (error == ERROR_PATH_NOT_FOUND)
    {
      HANDLE fileHandle = CreateFileTransactedW(wTarget, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr, transaction->transactionHandle, nullptr, nullptr);
      error = GetLastError(); // Might be `ERROR_ALREADY_EXISTS` even if the `fileHandle` is valid.

      if (fileHandle == INVALID_HANDLE_VALUE && error == ERROR_PATH_NOT_FOUND)
      {
        wchar_t parentDirectory[MAX_PATH];
        mERROR_CHECK(mStringCopy(parentDirectory, mARRAYSIZE(parentDirectory), wTarget, MAX_PATH));

#if (NTDDI_VERSION >= NTDDI_WIN8)
        HRESULT hr = S_OK;

        // Requires `pathcch.h` && `Pathcch.lib`.
        mERROR_IF(FAILED(hr = PathCchRemoveFileSpec(parentDirectory, mARRAYSIZE(parentDirectory))), mR_InternalError);
        mERROR_IF(hr == S_FALSE, mR_InvalidParameter);
#else
        // deprecated since windows 8.
        mERROR_IF(PathRemoveFileSpecW(parentDirectory), mR_InvalidParameter);
#endif

        mERROR_CHECK(_CreateDirectoryRecursive(transaction, parentDirectory));

        fileHandle = CreateFileTransactedW(wTarget, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr, transaction->transactionHandle, nullptr, nullptr);
        error = GetLastError(); // Might be `ERROR_ALREADY_EXISTS` even if the `fileHandle` is valid.
      }

      mERROR_IF(fileHandle == INVALID_HANDLE_VALUE, mR_InternalError);
      CloseHandle(fileHandle);

      if (0 == CopyFileTransactedW(wSource, wTarget, nullptr, nullptr, nullptr, 0, transaction->transactionHandle))
      {
        error = GetLastError();

        mRETURN_RESULT(mR_ResourceNotFound);
      }
      else
      {
        mRETURN_SUCCESS();
      }
    }

    mRETURN_RESULT(mR_IOFailure);
  }

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mFileTransaction_Destroy_Internal, IN_OUT mFileTransaction *pTransaction)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTransaction == nullptr, mR_ArgumentNull);

  CloseHandle(pTransaction->transactionHandle);

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(_CreateDirectoryRecursive, mPtr<mFileTransaction> &transaction, const wchar_t *directoryPath)
{
  mFUNCTION_SETUP();

  if (0 == CreateDirectoryTransactedW(nullptr, directoryPath, NULL, transaction->transactionHandle))
  {
    DWORD error = GetLastError();

    if (error == ERROR_ALREADY_EXISTS)
    {
      // Do nothing. We will return success later.
    }
    else if (error == ERROR_PATH_NOT_FOUND)
    {
      wchar_t parentDirectory[MAX_PATH];
      mERROR_CHECK(mStringCopy(parentDirectory, mARRAYSIZE(parentDirectory), directoryPath, MAX_PATH));

#if (NTDDI_VERSION >= NTDDI_WIN8)
      HRESULT hr = S_OK;

      // Requires `pathcch.h` && `Pathcch.lib`.
      mERROR_IF(FAILED(hr = PathCchRemoveFileSpec(parentDirectory, mARRAYSIZE(parentDirectory))), mR_InternalError);
      mERROR_IF(hr == S_FALSE, mR_InvalidParameter);
#else
      // deprecated since windows 8.
      mERROR_IF(PathRemoveFileSpecW(parentDirectory), mR_InvalidParameter);
#endif

      mERROR_CHECK(_CreateDirectoryRecursive(transaction, parentDirectory));

      if (0 == CreateDirectoryTransactedW(nullptr, directoryPath, NULL, transaction->transactionHandle))
      {
        error = GetLastError();

        mERROR_IF(error != ERROR_ALREADY_EXISTS, mR_InternalError);
      }
    }
    else
    {
      mRETURN_RESULT(mR_InternalError);
    }
  }

  mRETURN_SUCCESS();
}
