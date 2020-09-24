#include "mFile.h"

#include <sys\stat.h>

#include <shobjidl.h>
#include <shlobj.h>
#include <knownfolders.h>
#include <Shlwapi.h>

#include <aclapi.h>
#pragma comment(lib, "Advapi32.lib")

#if (NTDDI_VERSION >= NTDDI_WIN8)
#include "pathcch.h"

#pragma comment(lib, "Pathcch.lib")
#endif

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "SFo9prBt690qC0bDbPT05QaLU/H7Z7MJbc+QVmnEGYSYb72Yau3AznddRqFbpMzfsvluBUZPhqYFy894"
#endif

HRESULT mFile_CreateAndInitializeFileOperation_Internal(REFIID riid, void **ppFileOperation);
mFUNCTION(mFile_GetKnownPath_Internal, OUT mString *pString, const GUID &guid);
mFUNCTION(mFileInfo_FromFindDataWStruct_Internal, IN_OUT mFileInfo *pFileInfo, IN const WIN32_FIND_DATAW *pFileData, IN mAllocator *pAllocator);
mFUNCTION(mFileInfo_FromByHandleFileInformationStruct_Internal, IN_OUT mFileInfo *pFileInfo, IN const BY_HANDLE_FILE_INFORMATION *pFileData);
mFUNCTION(mFile_CreateDirectoryRecursive_Internal, const wchar_t *directoryPath);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mFile_Exists, const mString &filename, OUT bool *pExists)
{
  mFUNCTION_SETUP();

  mERROR_IF(pExists == nullptr, mR_ArgumentNull);
  mERROR_IF(filename.c_str() == nullptr || filename.hasFailed, mR_InvalidParameter);

  wchar_t wfilename[MAX_PATH];
  mERROR_CHECK(mString_ToWideString(filename, wfilename, mARRAYSIZE(wfilename)));

  mERROR_CHECK(mFile_Exists(wfilename, pExists));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_Exists, IN const wchar_t *filename, OUT bool *pExists)
{
  mFUNCTION_SETUP();

  mERROR_IF(pExists == nullptr || filename == nullptr, mR_ArgumentNull);

  *pExists = (PathFileExistsW(filename) != FALSE);

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_DirectoryExists, const mString &path, OUT bool *pExists)
{
  mFUNCTION_SETUP();

  mERROR_IF(path.c_str() == nullptr || path.hasFailed, mR_InvalidParameter);

  wchar_t wpath[MAX_PATH];
  mERROR_CHECK(mString_ToWideString(path, wpath, mARRAYSIZE(wpath)));

  mERROR_CHECK(mFile_DirectoryExists(wpath, pExists));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_DirectoryExists, IN const wchar_t *path, OUT bool *pExists)
{
  mFUNCTION_SETUP();

  mERROR_IF(pExists == nullptr || path == nullptr, mR_ArgumentNull);

  const DWORD attributes = GetFileAttributesW(path);

  if (attributes == INVALID_FILE_ATTRIBUTES)
  {
    const DWORD error = GetLastError();
    mUnused(error);

    *pExists = false;

    mRETURN_SUCCESS();
  }

  *pExists = !!(attributes & FILE_ATTRIBUTE_DIRECTORY);

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_ReadAllBytes, const mString &filename, IN OPTIONAL mAllocator *pAllocator, OUT mArray<uint8_t> *pBytes)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mFile_ReadAllItems(filename, pAllocator, pBytes));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_ReadAllBytes, IN const wchar_t *filename, IN OPTIONAL mAllocator *pAllocator, OUT mArray<uint8_t> *pBytes)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mFile_ReadAllItems(filename, pAllocator, pBytes));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_ReadAllText, const mString &filename, IN OPTIONAL mAllocator *pAllocator, OUT mString *pText, const mFile_Encoding /* encoding = mF_E_UTF8 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pText == nullptr, mR_ArgumentNull);

  wchar_t wfilename[MAX_PATH];
  mERROR_CHECK(mString_ToWideString(filename, wfilename, mARRAYSIZE(wfilename)));

  size_t count = 0;
  char *text = nullptr;

  mDEFER(mAllocator_FreePtr(nullptr, &text));
  mERROR_CHECK(mFile_ReadRaw(wfilename, &text, pAllocator, &count));

  mERROR_IF(count == 0, mR_ResourceNotFound);

  text[count] = '\0';
  *pText = mString(text);
  mERROR_CHECK(mAllocator_FreePtr(pAllocator, &text));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_WriteAllBytes, IN const wchar_t *filename, mArray<uint8_t> &bytes)
{
  mFUNCTION_SETUP();

  wchar_t wfilename[MAX_PATH];
  mERROR_CHECK(mString_ToWideString(filename, wfilename, mARRAYSIZE(wfilename)));

  mERROR_CHECK(mFile_WriteRawBytes(wfilename, bytes.pData, bytes.count));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_WriteAllBytes, const mString &filename, mArray<uint8_t> &bytes)
{
  mFUNCTION_SETUP();

  wchar_t wfilename[MAX_PATH];
  mERROR_CHECK(mString_ToWideString(filename, wfilename, mARRAYSIZE(wfilename)));

  mERROR_CHECK(mFile_WriteRawBytes(wfilename, reinterpret_cast<const uint8_t *>(bytes.pData), bytes.count));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_WriteAllText, const mString &filename, const mString &text, const mFile_Encoding /* encoding = mF_E_UTF8 */)
{
  mFUNCTION_SETUP();

  wchar_t wfilename[MAX_PATH];
  mERROR_CHECK(mString_ToWideString(filename, wfilename, mARRAYSIZE(wfilename)));

  mERROR_CHECK(mFile_WriteRawBytes(wfilename, reinterpret_cast<const uint8_t *>(text.c_str()), text.bytes - 1));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_WriteRawBytes, const wchar_t *filename, IN const uint8_t *pData, const size_t bytes)
{
  mFUNCTION_SETUP();

  mERROR_IF(pData == nullptr || filename == nullptr, mR_ArgumentNull);

  HANDLE fileHandle = CreateFileW(filename, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  DWORD error = GetLastError(); // Might be `ERROR_ALREADY_EXISTS` even if the `fileHandle` is valid.

  if (fileHandle == INVALID_HANDLE_VALUE && error == ERROR_PATH_NOT_FOUND)
  {
    wchar_t parentDirectory[MAX_PATH];
    mERROR_CHECK(mStringCopy(parentDirectory, mARRAYSIZE(parentDirectory), filename, MAX_PATH));

#if (NTDDI_VERSION >= NTDDI_WIN8)
    HRESULT hr = S_OK;

    // Requires `pathcch.h` && `Pathcch.lib`.
    mERROR_IF(FAILED(hr = PathCchRemoveFileSpec(parentDirectory, mARRAYSIZE(parentDirectory))), mR_InternalError);
    mERROR_IF(hr == S_FALSE, mR_InvalidParameter);
#else
    // deprecated since windows 8.
    mERROR_IF(PathRemoveFileSpecW(parentDirectory), mR_InvalidParameter);
#endif

    mERROR_CHECK(mFile_CreateDirectoryRecursive_Internal(parentDirectory));

    fileHandle = CreateFileW(filename, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
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

mFUNCTION(mFile_FailOnInvalidDirectoryPath, const mString &folderPath, OUT OPTIONAL mString *pAbsolutePath)
{
  mFUNCTION_SETUP();

  mERROR_IF(folderPath.hasFailed, mR_InvalidParameter);

  mString path;
  mERROR_CHECK(mFile_GetAbsoluteDirectoryPath(&path, folderPath));

  mDEFER(
    if (pAbsolutePath != nullptr)
      *pAbsolutePath = std::move(path);
  );

  // Prevent user from creating a directory that windows explorer doesn't support.
  {
    bool hasSpaceOrPeriodInCurrentPath = false;
    bool hasNonSpaceOrPeriodInCurrentPath = false;
    bool lastWasSpaceOrDot = false;

    const mchar_t space = mToChar<2>(" ");
    const mchar_t period = mToChar<2>(".");
    const mchar_t backslash = mToChar<2>("\\");

    for (auto _char : path)
    {
      if (_char.codePoint == space || _char.codePoint == period)
      {
        hasSpaceOrPeriodInCurrentPath = true;
        lastWasSpaceOrDot = true;
      }
      else if (_char.codePoint == backslash)
      {
        mERROR_IF(!hasNonSpaceOrPeriodInCurrentPath || lastWasSpaceOrDot, mR_InvalidParameter);
        hasSpaceOrPeriodInCurrentPath = false;
        hasNonSpaceOrPeriodInCurrentPath = false;
        lastWasSpaceOrDot = false;
      }
      else
      {
        hasNonSpaceOrPeriodInCurrentPath = true;
        lastWasSpaceOrDot = false;
      }
    }

    mERROR_IF(lastWasSpaceOrDot, mR_InvalidParameter);
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_CreateDirectory, const mString &folderPath)
{
  mFUNCTION_SETUP();

  mString path;
  mERROR_CHECK(mFile_FailOnInvalidDirectoryPath(folderPath, &path));

  wchar_t wDirectoryName[MAX_PATH];
  mERROR_CHECK(mString_ToWideString(path, wDirectoryName, mARRAYSIZE(wDirectoryName)));

  const int errorCode = SHCreateDirectoryExW(NULL, wDirectoryName, NULL);

  switch (errorCode)
  {
  case ERROR_SUCCESS:
  case ERROR_ALREADY_EXISTS:
  case ERROR_FILE_EXISTS:
    break;

  case ERROR_PATH_NOT_FOUND:
    mRETURN_RESULT(mR_ResourceNotFound);
    break;

  case ERROR_BAD_PATHNAME:
  case ERROR_FILENAME_EXCED_RANGE:
    mRETURN_RESULT(mR_InvalidParameter);
    break;

  case ERROR_CANCELLED:
  default:
    mRETURN_RESULT(mR_InternalError);
    break;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_DeleteFolder, const mString &folderPath)
{
  mFUNCTION_SETUP();

  mERROR_IF(folderPath.hasFailed, mR_InvalidParameter);

  mString path;
  mERROR_CHECK(mString_ToDirectoryPath(&path, folderPath));

  mString pathWithoutLastSlash;
  mERROR_CHECK(mString_Substring(path, &pathWithoutLastSlash, 0, path.Count() - 2));

  wchar_t wdirectoryName[MAX_PATH];
  mERROR_CHECK(mString_ToWideString(path, wdirectoryName, mARRAYSIZE(wdirectoryName)));

  HRESULT hr = S_OK;

  wchar_t absolutePath[MAX_PATH];
  const DWORD length = GetFullPathNameW(wdirectoryName, mARRAYSIZE(absolutePath), absolutePath, nullptr);
  mUnused(length);

  IShellItem *pItem = nullptr;
  mERROR_IF(FAILED(hr = SHCreateItemFromParsingName(absolutePath, nullptr, IID_PPV_ARGS(&pItem))), mR_InternalError);
  mDEFER(pItem->Release());

  IFileOperation *pFileOperation = nullptr;
  mERROR_IF(FAILED(hr = mFile_CreateAndInitializeFileOperation_Internal(IID_PPV_ARGS(&pFileOperation))), mR_InternalError);
  mDEFER(pFileOperation->Release());

  mERROR_IF(FAILED(hr = pFileOperation->DeleteItem(pItem, nullptr)), mR_InternalError);
  mERROR_IF(FAILED(hr = pFileOperation->PerformOperations()), mR_InternalError);

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_Copy, const mString &destinationFileName, const mString &sourceFileName, const bool overrideFileIfExistent /* = false */)
{
  mFUNCTION_SETUP();

  wchar_t source[MAX_PATH];
  mERROR_CHECK(mString_ToWideString(sourceFileName, source, mARRAYSIZE(source)));

  wchar_t target[MAX_PATH];
  mERROR_CHECK(mString_ToWideString(destinationFileName, target, mARRAYSIZE(source)));

  const BOOL succeeded = CopyFileW(source, target, !overrideFileIfExistent);
  mERROR_IF(succeeded, mR_Success);

  DWORD error = GetLastError();

  switch (error)
  {
  case ERROR_ACCESS_DENIED:
  {
    mRETURN_RESULT(mR_ResourceAlreadyExists);
  }

  case ERROR_FILE_NOT_FOUND:
  {
    mRETURN_RESULT(mR_ResourceNotFound);
  }

  case ERROR_PATH_NOT_FOUND:
  {
    HANDLE fileHandle = CreateFileW(target, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    error = GetLastError(); // Might be `ERROR_ALREADY_EXISTS` even if the `fileHandle` is valid.

    mERROR_IF(error == ERROR_ALREADY_EXISTS && !overrideFileIfExistent, mR_Failure);

    if (fileHandle == INVALID_HANDLE_VALUE && error == ERROR_PATH_NOT_FOUND)
    {
      wchar_t parentDirectory[MAX_PATH];
      mERROR_CHECK(mStringCopy(parentDirectory, mARRAYSIZE(parentDirectory), target, MAX_PATH));

#if (NTDDI_VERSION >= NTDDI_WIN8)
      HRESULT hr = S_OK;

      // Requires `pathcch.h` && `Pathcch.lib`.
      mERROR_IF(FAILED(hr = PathCchRemoveFileSpec(parentDirectory, mARRAYSIZE(parentDirectory))), mR_InternalError);
      mERROR_IF(hr == S_FALSE, mR_InvalidParameter);
#else
      // deprecated since windows 8.
      mERROR_IF(PathRemoveFileSpecW(parentDirectory), mR_InvalidParameter);
#endif

      mERROR_CHECK(mFile_CreateDirectoryRecursive_Internal(parentDirectory));

      fileHandle = CreateFileW(target, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
      error = GetLastError(); // Might be `ERROR_ALREADY_EXISTS` even if the `fileHandle` is valid.
    }

    mERROR_IF(fileHandle == INVALID_HANDLE_VALUE, mR_InternalError);
    CloseHandle(fileHandle);

    if (0 == CopyFileW(source, target, FALSE))
    {
      error = GetLastError();

      mRETURN_RESULT(mR_ResourceNotFound);
    }
    else
    {
      mRETURN_SUCCESS();
    }
  }

  default:
  {
    mRETURN_RESULT(mR_IOFailure);
  }
  }
}

struct mFile_ProgressCallbackParams
{
  const std::function<mResult(const double_t progress)> &func;
  BOOL *pCanceled;
  mResult innerResult;

  mFile_ProgressCallbackParams(BOOL *pCanceled, const std::function<mResult(const double_t progress)> &func) :
    pCanceled(pCanceled),
    func(func)
  { }
};

struct mFile_ProgressCallback_Internal
{
  static DWORD WINAPI Callback(
    _In_     LARGE_INTEGER TotalFileSize,
    _In_     LARGE_INTEGER TotalBytesTransferred,
    _In_     LARGE_INTEGER /* StreamSize */,
    _In_     LARGE_INTEGER /* StreamBytesTransferred */,
    _In_     DWORD /* dwStreamNumber */,
    _In_     DWORD /* dwCallbackReason */,
    _In_     HANDLE /* hSourceFile */,
    _In_     HANDLE /* hDestinationFile */,
    _In_opt_ LPVOID lpData)
  {
    mFile_ProgressCallbackParams *pParams = reinterpret_cast<mFile_ProgressCallbackParams *>(lpData);

    if (mFAILED(pParams->innerResult = pParams->func(TotalBytesTransferred.QuadPart / (double_t)TotalFileSize.QuadPart)))
    {
      *pParams->pCanceled = TRUE;
      return PROGRESS_CANCEL;
    }

    return ERROR_SUCCESS;
  }
};

mFUNCTION(mFile_Copy, const mString &destinationFileName, const mString &sourceFileName, const std::function<mResult(const double_t progress)> &progressCallback, const bool overrideFileIfExistent)
{
  mFUNCTION_SETUP();

  mERROR_IF(progressCallback == nullptr, mR_ArgumentNull);

  wchar_t source[MAX_PATH];
  mERROR_CHECK(mString_ToWideString(sourceFileName, source, mARRAYSIZE(source)));

  wchar_t target[MAX_PATH];
  mERROR_CHECK(mString_ToWideString(destinationFileName, target, mARRAYSIZE(source)));

  BOOL canceled = FALSE;
  mFile_ProgressCallbackParams parameters(&canceled, progressCallback);

  if (0 == CopyFileExW(source, target, &mFile_ProgressCallback_Internal::Callback, reinterpret_cast<void *>(&parameters), &canceled, (overrideFileIfExistent ? 0 : COPY_FILE_FAIL_IF_EXISTS) | COPY_FILE_NO_BUFFERING))
  {
    if (canceled == TRUE)
      mRETURN_RESULT(mFAILED(parameters.innerResult) ? parameters.innerResult : mR_Break);

    DWORD error = GetLastError();

    switch (error)
    {
    case ERROR_REQUEST_ABORTED:
    {
      mRETURN_RESULT(mFAILED(parameters.innerResult) ? parameters.innerResult : mR_Break);
    }

    case ERROR_FILE_EXISTS:
    {
      mRETURN_RESULT(mR_Failure);
    }

    case ERROR_FILE_NOT_FOUND:
    {
      mRETURN_RESULT(mR_ResourceNotFound);
    }

    case ERROR_PATH_NOT_FOUND:
    {
      HANDLE fileHandle = CreateFileW(target, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
      error = GetLastError(); // Might be `ERROR_ALREADY_EXISTS` even if the `fileHandle` is valid.

      mERROR_IF(error == ERROR_ALREADY_EXISTS && !overrideFileIfExistent, mR_Failure);

      if (fileHandle == INVALID_HANDLE_VALUE && error == ERROR_PATH_NOT_FOUND)
      {
        wchar_t parentDirectory[MAX_PATH];
        mERROR_CHECK(mStringCopy(parentDirectory, mARRAYSIZE(parentDirectory), target, MAX_PATH));

#if (NTDDI_VERSION >= NTDDI_WIN8)
        HRESULT hr = S_OK;

        // Requires `pathcch.h` && `Pathcch.lib`.
        mERROR_IF(FAILED(hr = PathCchRemoveFileSpec(parentDirectory, mARRAYSIZE(parentDirectory))), mR_InternalError);
        mERROR_IF(hr == S_FALSE, mR_InvalidParameter);
#else
        // deprecated since windows 8.
        mERROR_IF(PathRemoveFileSpecW(parentDirectory), mR_InvalidParameter);
#endif

        mERROR_CHECK(mFile_CreateDirectoryRecursive_Internal(parentDirectory));

        fileHandle = CreateFileW(target, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        error = GetLastError(); // Might be `ERROR_ALREADY_EXISTS` even if the `fileHandle` is valid.
      }

      mERROR_IF(fileHandle == INVALID_HANDLE_VALUE, mR_InternalError);
      CloseHandle(fileHandle);

      canceled = FALSE;

      if (0 == CopyFileExW(source, target, &mFile_ProgressCallback_Internal::Callback, reinterpret_cast<void *>(&parameters), &canceled, COPY_FILE_NO_BUFFERING))
      {
        error = GetLastError();

        if (canceled == TRUE || error == ERROR_REQUEST_ABORTED)
          mRETURN_RESULT(mFAILED(parameters.innerResult) ? parameters.innerResult : mR_Break);

        mRETURN_RESULT(mR_ResourceNotFound);
      }
      else
      {
        mRETURN_SUCCESS();
      }
    }

    default:
    {
      mRETURN_RESULT(mR_IOFailure);
    }
    }
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_Move, const mString &destinationFileName, const mString &sourceFileName, const bool overrideFileIfExistent /* = true */)
{
  mFUNCTION_SETUP();

  wchar_t source[MAX_PATH];
  mERROR_CHECK(mString_ToWideString(sourceFileName, source, mARRAYSIZE(source)));

  wchar_t target[MAX_PATH];
  mERROR_CHECK(mString_ToWideString(destinationFileName, target, mARRAYSIZE(source)));

  const BOOL succeeded = MoveFileExW(source, target, (overrideFileIfExistent ? MOVEFILE_REPLACE_EXISTING : 0) | MOVEFILE_COPY_ALLOWED);
  mERROR_IF(succeeded, mR_Success);

  DWORD error = GetLastError();

  switch (error)
  {
  case ERROR_FILE_NOT_FOUND:
  {
    mRETURN_RESULT(mR_ResourceNotFound);
  }

  case ERROR_PATH_NOT_FOUND:
  {
    HANDLE fileHandle = CreateFileW(target, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    error = GetLastError(); // Might be `ERROR_ALREADY_EXISTS` even if the `fileHandle` is valid.

    mERROR_IF(error == ERROR_ALREADY_EXISTS && !overrideFileIfExistent, mR_Failure);

    if (fileHandle == INVALID_HANDLE_VALUE && error == ERROR_PATH_NOT_FOUND)
    {
      wchar_t parentDirectory[MAX_PATH];
      mERROR_CHECK(mStringCopy(parentDirectory, mARRAYSIZE(parentDirectory), target, MAX_PATH));

#if (NTDDI_VERSION >= NTDDI_WIN8)
      HRESULT hr = S_OK;

      // Requires `pathcch.h` && `Pathcch.lib`.
      mERROR_IF(FAILED(hr = PathCchRemoveFileSpec(parentDirectory, mARRAYSIZE(parentDirectory))), mR_InternalError);
      mERROR_IF(hr == S_FALSE, mR_InvalidParameter);
#else
      // deprecated since windows 8.
      mERROR_IF(PathRemoveFileSpecW(parentDirectory), mR_InvalidParameter);
#endif

      mERROR_CHECK(mFile_CreateDirectoryRecursive_Internal(parentDirectory));

      fileHandle = CreateFileW(target, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
      error = GetLastError(); // Might be `ERROR_ALREADY_EXISTS` even if the `fileHandle` is valid.
    }

    mERROR_IF(fileHandle == INVALID_HANDLE_VALUE, mR_InternalError);
    CloseHandle(fileHandle);

    if (0 == MoveFileExW(source, target, MOVEFILE_REPLACE_EXISTING | MOVEFILE_COPY_ALLOWED))
    {
      error = GetLastError();

      mRETURN_RESULT(mR_ResourceNotFound);
    }
    else
    {
      mRETURN_SUCCESS();
    }
  }

  default:
  {
    mRETURN_RESULT(mR_IOFailure);
  }
  }
}

mFUNCTION(mFile_Move, const mString &destinationFileName, const mString &sourceFileName, const std::function<mResult(const double_t progress)> &progressCallback, const bool overrideFileIfExistent)
{
  mFUNCTION_SETUP();

  mERROR_IF(progressCallback == nullptr, mR_ArgumentNull);

  wchar_t source[MAX_PATH];
  mERROR_CHECK(mString_ToWideString(sourceFileName, source, mARRAYSIZE(source)));

  wchar_t target[MAX_PATH];
  mERROR_CHECK(mString_ToWideString(destinationFileName, target, mARRAYSIZE(source)));

  BOOL canceled = FALSE;
  mFile_ProgressCallbackParams parameters(&canceled, progressCallback);

  if (0 == MoveFileWithProgressW(source, target, &mFile_ProgressCallback_Internal::Callback, reinterpret_cast<void *>(&parameters), (overrideFileIfExistent ? MOVEFILE_REPLACE_EXISTING : 0) | MOVEFILE_COPY_ALLOWED))
  {
    DWORD error = GetLastError();

    switch (error)
    {
    case ERROR_REQUEST_ABORTED:
    {
      mRETURN_RESULT(mFAILED(parameters.innerResult) ? parameters.innerResult : mR_Break);
    }

    case ERROR_FILE_EXISTS:
    {
      mRETURN_RESULT(mR_Failure);
    }

    case ERROR_FILE_NOT_FOUND:
    {
      mRETURN_RESULT(mR_ResourceNotFound);
    }

    case ERROR_PATH_NOT_FOUND:
    {
      HANDLE fileHandle = CreateFileW(target, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
      error = GetLastError(); // Might be `ERROR_ALREADY_EXISTS` even if the `fileHandle` is valid.

      mERROR_IF(error == ERROR_ALREADY_EXISTS && !overrideFileIfExistent, mR_Failure);

      if (fileHandle == INVALID_HANDLE_VALUE && error == ERROR_PATH_NOT_FOUND)
      {
        wchar_t parentDirectory[MAX_PATH];
        mERROR_CHECK(mStringCopy(parentDirectory, mARRAYSIZE(parentDirectory), target, MAX_PATH));

#if (NTDDI_VERSION >= NTDDI_WIN8)
        HRESULT hr = S_OK;

        // Requires `pathcch.h` && `Pathcch.lib`.
        mERROR_IF(FAILED(hr = PathCchRemoveFileSpec(parentDirectory, mARRAYSIZE(parentDirectory))), mR_InternalError);
        mERROR_IF(hr == S_FALSE, mR_InvalidParameter);
#else
        // deprecated since windows 8.
        mERROR_IF(PathRemoveFileSpecW(parentDirectory), mR_InvalidParameter);
#endif

        mERROR_CHECK(mFile_CreateDirectoryRecursive_Internal(parentDirectory));

        fileHandle = CreateFileW(target, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        error = GetLastError(); // Might be `ERROR_ALREADY_EXISTS` even if the `fileHandle` is valid.
      }

      mERROR_IF(fileHandle == INVALID_HANDLE_VALUE, mR_InternalError);
      CloseHandle(fileHandle);

      if (0 == MoveFileWithProgressW(source, target, &mFile_ProgressCallback_Internal::Callback, reinterpret_cast<void *>(&parameters), MOVEFILE_REPLACE_EXISTING | MOVEFILE_COPY_ALLOWED))
      {
        error = GetLastError();

        if (error == ERROR_REQUEST_ABORTED)
          mRETURN_RESULT(mFAILED(parameters.innerResult) ? parameters.innerResult : mR_Break);

        mRETURN_RESULT(mR_ResourceNotFound);
      }
      else
      {
        mRETURN_SUCCESS();
      }
    }

    default:
    {  mRETURN_RESULT(mR_IOFailure);
    }
    }
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_Delete, const mString &filename)
{
  mFUNCTION_SETUP();

  wchar_t wfilename[MAX_PATH];
  mERROR_CHECK(mString_ToWideString(filename, wfilename, mARRAYSIZE(wfilename)));

  const BOOL succeeded = DeleteFileW(wfilename);
  mERROR_IF(succeeded, mR_Success);

  const DWORD errorCode = GetLastError();

  switch (errorCode)
  {
  case ERROR_FILE_NOT_FOUND:
    mRETURN_RESULT(mR_ResourceNotFound);

  default:
    mRETURN_RESULT(mR_IOFailure);
  }
}

mFUNCTION(mFile_GetTempDirectory, OUT mString *pString)
{
  mFUNCTION_SETUP();

  mERROR_IF(pString == nullptr, mR_ArgumentNull);
  
  wchar_t buffer[MAX_PATH + 1];
  const DWORD length = GetTempPathW(mARRAYSIZE(buffer), buffer);

  mERROR_IF(length == 0, mR_InternalError); // This would theoretically set GetLastError() to some error code, however msdn doesn't list any possible error codes.

  mERROR_CHECK(mString_Create(pString, buffer, length, pString->pAllocator)); // According to msdn this is terminated by a backslash.

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_GetAppDataDirectory, OUT mString *pString)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mFile_GetKnownPath_Internal(pString, FOLDERID_RoamingAppData));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_GetDesktopDirectory, OUT mString *pString)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mFile_GetKnownPath_Internal(pString, FOLDERID_Desktop));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_GetDocumentsDirectory, OUT mString *pString)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mFile_GetKnownPath_Internal(pString, FOLDERID_Documents));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_GetFontsDirectory, OUT mString *pString)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mFile_GetKnownPath_Internal(pString, FOLDERID_Fonts));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_GetCurrentUserDirectory, OUT mString *pString)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mFile_GetKnownPath_Internal(pString, FOLDERID_Profile));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_GetProgramFilesDirectory, OUT mString *pString)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mFile_GetKnownPath_Internal(pString, FOLDERID_ProgramFiles));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_GetStartupDirectory, OUT mString *pString)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mFile_GetKnownPath_Internal(pString, FOLDERID_Startup));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_GetStartMenu, OUT mString *pString)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mFile_GetKnownPath_Internal(pString, FOLDERID_StartMenu));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_GetStartMenuPrograms, OUT mString *pString)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mFile_GetKnownPath_Internal(pString, FOLDERID_Programs));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_GetSystemFolder, OUT mString *pString)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mFile_GetKnownPath_Internal(pString, FOLDERID_System));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_GetSystemFolderX86, OUT mString *pString)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mFile_GetKnownPath_Internal(pString, FOLDERID_SystemX86));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_GetOSFolder, OUT mString *pString)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mFile_GetKnownPath_Internal(pString, FOLDERID_Windows));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_GetStartupDirectory_AllUsers, OUT mString *pString)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mFile_GetKnownPath_Internal(pString, FOLDERID_CommonStartup));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_GetStartMenu_AllUsers, OUT mString *pString)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mFile_GetKnownPath_Internal(pString, FOLDERID_CommonStartMenu));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_GetStartMenuPrograms_AllUsers, OUT mString *pString)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mFile_GetKnownPath_Internal(pString, FOLDERID_CommonPrograms));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_GetWorkingDirectory, OUT mString *pWorkingDirectory)
{
  mFUNCTION_SETUP();

  mERROR_IF(pWorkingDirectory == nullptr, mR_ArgumentNull);

  wchar_t directory[MAX_PATH];

  DWORD result = GetCurrentDirectoryW(mARRAYSIZE(directory), directory);

  if (result == 0)
  {
    const DWORD error = GetLastError();
    mUnused(error);

    mRETURN_RESULT(mR_IOFailure);
  }

  wchar_t fullPathName[MAX_PATH];

  result = GetFullPathNameW(directory, mARRAYSIZE(fullPathName), fullPathName, nullptr);

  if (result == 0)
  {
    const DWORD error = GetLastError();
    mUnused(error);

    mRETURN_RESULT(mR_IOFailure);
  }

  mERROR_CHECK(mString_Create(pWorkingDirectory, fullPathName, result));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_SetWorkingDirectory, const mString &workingDirectory)
{
  mFUNCTION_SETUP();

  mERROR_IF(workingDirectory.hasFailed, mR_InvalidParameter);

  wchar_t wPath[MAX_PATH];
  mERROR_CHECK(mString_ToWideString(workingDirectory, wPath, mARRAYSIZE(wPath)));

  if (!SetCurrentDirectoryW(wPath))
  {
    const DWORD error = GetLastError();
    mUnused(error);

    mRETURN_RESULT(mR_IOFailure);
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_GetCurrentApplicationFilePath, OUT mString *pAppDirectory)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAppDirectory == nullptr, mR_ArgumentNull);

  wchar_t filePath[MAX_PATH];

  const DWORD result = GetModuleFileNameW(nullptr, filePath, mARRAYSIZE(filePath));

  if (result != 0)
  {
    const DWORD error = GetLastError();

    switch (error)
    {
    case ERROR_SUCCESS:
      mERROR_CHECK(mString_Create(pAppDirectory, filePath, pAppDirectory->pAllocator));
      mRETURN_SUCCESS();
      break;

    case ERROR_INSUFFICIENT_BUFFER:
    default:
      mRETURN_RESULT(mR_InternalError);
      break;
    }
  }

  mRETURN_RESULT(mR_InternalError);
}

mFUNCTION(mFile_ExtractDirectoryFromPath, OUT mString *pDirectory, const mString &filePath)
{
  mFUNCTION_SETUP();

  mERROR_IF(pDirectory == nullptr, mR_ArgumentNull);

  wchar_t wPath[MAX_PATH];
  mERROR_CHECK(mString_ToWideString(filePath, wPath, mARRAYSIZE(wPath)));

#if (NTDDI_VERSION >= NTDDI_WIN8)
  HRESULT hr = S_OK;

  // Requires `pathcch.h` && `Pathcch.lib`.
  mERROR_IF(FAILED(hr = PathCchRemoveFileSpec(wPath, mARRAYSIZE(wPath))), mR_InternalError); // on success returns `S_OK`, or `S_FALSE` if nothing was removed.
  //mERROR_IF(hr == S_FALSE, mR_InvalidParameter); // we don't actually care, do we?
  mUnused(hr);
#else
  // deprecated since windows 8.
  const BOOL result = PathRemoveFileSpecW(wPath); // result is true if something was removed.

  //mERROR_IF(result, mR_InvalidParameter); // we don't actually care.
  mUnused(result);
#endif

  mERROR_CHECK(mString_Create(pDirectory, wPath, mARRAYSIZE(wPath), pDirectory->pAllocator));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_ExtractFileExtensionFromPath, OUT mString *pExtension, const mString &filePath)
{
  mFUNCTION_SETUP();

  mERROR_IF(pExtension == nullptr, mR_ArgumentNull);

  wchar_t wPath[MAX_PATH];
  mERROR_CHECK(mString_ToWideString(filePath, wPath, mARRAYSIZE(wPath)));

  wchar_t *extension = PathFindExtensionW(wPath);
  mERROR_IF(extension == nullptr || *extension == L'\0', mR_ResourceNotFound);

  mERROR_CHECK(mString_Create(pExtension, extension, pExtension->pAllocator));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_ExtractFileNameFromPath, OUT mString *pFileName, const mString &filePath, const bool withExtension)
{
  mFUNCTION_SETUP();

  mERROR_IF(pFileName == nullptr, mR_ArgumentNull);

  wchar_t wPath[MAX_PATH];
  mERROR_CHECK(mString_ToWideString(filePath, wPath, mARRAYSIZE(wPath)));

  wchar_t *fileName = PathFindFileNameW(wPath);
  mERROR_IF(fileName == nullptr || *fileName == L'\0', mR_ResourceNotFound);

  if (!withExtension)
  {
    wchar_t *extension = PathFindExtensionW(wPath);

    if (!(extension == nullptr || *extension == L'\0'))
      *extension = '\0';
  }

  mERROR_CHECK(mString_Create(pFileName, fileName, pFileName->pAllocator));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_GetDirectoryContents, const mString &directoryPath, const mString &searchTerm, OUT mPtr<mQueue<mFileInfo>> *pFiles, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pFiles == nullptr, mR_ArgumentNull);
  mERROR_IF(directoryPath.hasFailed, mR_InvalidParameter);

  mString actualFolderPath;
  mERROR_CHECK(mString_ToDirectoryPath(&actualFolderPath, directoryPath));

  mERROR_CHECK(mString_Append(actualFolderPath, searchTerm));

  wchar_t folderPath[MAX_PATH];
  mERROR_CHECK(mString_ToWideString(actualFolderPath, folderPath, mARRAYSIZE(folderPath)));

  wchar_t absolutePath[MAX_PATH];
  const DWORD length = GetFullPathNameW(folderPath, mARRAYSIZE(absolutePath), absolutePath, nullptr);
  mERROR_IF(length == 0, mR_InternalError);

  if (*pFiles == nullptr)
    mERROR_CHECK(mQueue_Create(pFiles, pAllocator));
  else
    mERROR_CHECK(mQueue_Clear(*pFiles));
  
  WIN32_FIND_DATAW fileData;
  const HANDLE handle = FindFirstFileExW(absolutePath, FindExInfoBasic, &fileData, FindExSearchNameMatch, NULL, FIND_FIRST_EX_LARGE_FETCH);

  if (handle == INVALID_HANDLE_VALUE)
  {
    const DWORD error = GetLastError();

    switch (error)
    {
    case ERROR_FILE_NOT_FOUND:
    case ERROR_PATH_NOT_FOUND:
    case ERROR_BAD_NETPATH:
      mRETURN_RESULT(mR_ResourceNotFound);

    case ERROR_DIRECTORY:
    default:
      mRETURN_RESULT(mR_InternalError);
    }
  }

  mDEFER(FindClose(handle));

  do
  {
    mFileInfo fileInfo;
    mERROR_CHECK(mFileInfo_FromFindDataWStruct_Internal(&fileInfo, &fileData, pAllocator));
    mERROR_CHECK(mQueue_PushBack(*pFiles, fileInfo));
  } while (FindNextFileW(handle, &fileData));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_GetDirectoryContents, const mString &directoryPath, const char *searchTerm, OUT mPtr<mQueue<mFileInfo>> *pFiles, IN mAllocator *pAllocator)
{
  return mFile_GetDirectoryContents(directoryPath, mString(searchTerm), pFiles, pAllocator);
}

mFUNCTION(mFile_GetDirectoryContents, const mString &directoryPath, const wchar_t *searchTerm, OUT mPtr<mQueue<mFileInfo>> *pFiles, IN mAllocator *pAllocator)
{
  return mFile_GetDirectoryContents(directoryPath, mString(searchTerm), pFiles, pAllocator);
}

mFUNCTION(mFile_GetDirectoryContents, const mString &directoryPath, OUT mPtr<mQueue<mFileInfo>> *pFiles, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pFiles == nullptr, mR_ArgumentNull);
  mERROR_IF(directoryPath.hasFailed, mR_InvalidParameter);

  mString searchTerm;
  mERROR_CHECK(mString_Create(&searchTerm, "*", 2, &mDefaultTempAllocator));

  mERROR_CHECK(mFile_GetDirectoryContents(directoryPath, searchTerm, pFiles, pAllocator));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_GetDirectoryContents, const mString &directoryPath, const bool recursive, OUT mPtr<mQueue<mFileInfo>> *pFiles, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  if (!recursive)
    mRETURN_RESULT(mFile_GetDirectoryContents(directoryPath, pFiles, pAllocator));

  if (*pFiles == nullptr)
    mERROR_CHECK(mQueue_Create(pFiles, pAllocator));
  else
    mERROR_CHECK(mQueue_Clear(*pFiles));

  mString currentFolder, currentFile;
  mERROR_CHECK(mString_Create(&currentFolder, directoryPath));

  mPtr<mQueue<mString>> enqueuedDirectories;
  mDEFER_CALL(&enqueuedDirectories, mQueue_Destroy);
  mERROR_CHECK(mQueue_Create(&enqueuedDirectories, pAllocator));

  mERROR_CHECK(mQueue_PushBack(enqueuedDirectories, currentFolder));

  mPtr<mQueue<mFileInfo>> directoryContents;
  mDEFER_CALL(&directoryContents, mQueue_Destroy);

  while (true)
  {
    size_t count;
    mERROR_CHECK(mQueue_GetCount(enqueuedDirectories, &count));

    if (count == 0)
      break;

    mERROR_CHECK(mQueue_PopFront(enqueuedDirectories, &currentFolder));

    if (mFAILED(mFile_GetDirectoryContents(currentFolder, &directoryContents, pAllocator)))
      continue;

    for (auto &_file : directoryContents->Iterate())
    {
      // Skip '.' and '..'.
      if (_file.name.Count() >= 2 && _file.name[0] == mToChar<2>("."))
      {
        if (_file.name.Count() == 2)
          continue;
        else if (_file.name[1] == mToChar<2>("."))
          continue;
      }

      mERROR_CHECK(mString_Create(&currentFile, currentFolder));
      mERROR_CHECK(mString_Append(currentFile, "\\"));
      mERROR_CHECK(mString_Append(currentFile, _file.name));

      if (_file.size == 0)
      {
        mERROR_CHECK(mQueue_PushBack(enqueuedDirectories, currentFile));
      }
      else
      {
        mERROR_CHECK(mString_Create(&_file.name, currentFile));
        mERROR_CHECK(mQueue_PushBack(*pFiles, _file));
      }
    }
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_GetInfo, const mString &filename, OUT mFileInfo *pFileInfo)
{
  mFUNCTION_SETUP();

  mERROR_IF(pFileInfo == nullptr, mR_ArgumentNull);
  mERROR_IF(filename.hasFailed, mR_InvalidParameter);

  wchar_t wfilename[MAX_PATH];
  mERROR_CHECK(mString_ToWideString(filename, wfilename, mARRAYSIZE(wfilename)));

  HANDLE fileHandle = CreateFileW(wfilename, STANDARD_RIGHTS_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);

  if (fileHandle == INVALID_HANDLE_VALUE)
  {
    const DWORD error = GetLastError();

    switch (error)
    {
    case ERROR_FILE_NOT_FOUND:
      mRETURN_RESULT(mR_ResourceNotFound);

    case ERROR_ACCESS_DENIED:
      mRETURN_RESULT(mR_InsufficientPrivileges);

    default:
      mRETURN_RESULT(mR_InternalError);
    }
  }

  mDEFER(CloseHandle(fileHandle));

  BY_HANDLE_FILE_INFORMATION info;
  mERROR_IF(0 == GetFileInformationByHandle(fileHandle, &info), mR_InternalError); // Sets `GetLastError()`.

  mERROR_CHECK(mString_Create(&pFileInfo->name, filename));
  mERROR_CHECK(mFileInfo_FromByHandleFileInformationStruct_Internal(pFileInfo, &info));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_GetDrives, OUT mPtr<mQueue<mDriveInfo>> *pDrives, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pDrives == nullptr, mR_ArgumentNull);

  if (*pDrives == nullptr)
    mERROR_CHECK(mQueue_Create(pDrives, pAllocator));
  else
    mERROR_CHECK(mQueue_Clear(*pDrives));

  wchar_t driveLabels[MAX_PATH];
  const size_t length = GetLogicalDriveStringsW(mARRAYSIZE(driveLabels), driveLabels);
  
  if (length == 0 || length >= mARRAYSIZE(driveLabels))
  {
    const DWORD error = GetLastError();
    mUnused(error);

    mRETURN_RESULT(mR_InternalError);
  }

  wchar_t *nextDriveLabel = driveLabels;

  while (*nextDriveLabel != L'\0')
  {
    mDriveInfo driveInfo;
    mERROR_CHECK(mZeroMemory(&driveInfo));

    mDEFER_CALL(&driveInfo.drivePath, mString_Destroy);
    mERROR_CHECK(mString_Create(&driveInfo.drivePath, nextDriveLabel, pAllocator));

    wchar_t driveName[MAX_PATH + 1];
    wchar_t volumeTypeName[MAX_PATH + 1]; // FAT, NTFS, ...
    const BOOL result = GetVolumeInformationW(nextDriveLabel, driveName, mARRAYSIZE(driveName), nullptr, nullptr, nullptr, volumeTypeName, mARRAYSIZE(volumeTypeName));

    if (result)
      mERROR_CHECK(mString_Create(&driveInfo.driveName, driveName, pAllocator));

    const UINT driveType = GetDriveTypeW(nextDriveLabel);

    switch (driveType)
    {
    case DRIVE_UNKNOWN:
    case DRIVE_NO_ROOT_DIR:
    default:
      driveInfo.driveType = mDT_Unknown;
      break;

    case DRIVE_REMOVABLE:
      driveInfo.driveType = mDT_Removable;
      break;

    case DRIVE_FIXED:
      driveInfo.driveType = mDT_NonRemovable;
      break;

    case DRIVE_REMOTE:
      driveInfo.driveType = mDT_Remote;
      break;

    case DRIVE_CDROM:
      driveInfo.driveType = mDT_CDRom;
      break;

    case DRIVE_RAMDISK:
      driveInfo.driveType = mDT_RamDisk;
      break;
    }

    mERROR_CHECK(mQueue_PushBack(*pDrives, std::move(driveInfo)));

    nextDriveLabel = &nextDriveLabel[lstrlenW(nextDriveLabel) + 1];
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_GetDrives, OUT mPtr<mQueue<mString>> *pDrives, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pDrives == nullptr, mR_ArgumentNull);

  if (*pDrives == nullptr)
    mERROR_CHECK(mQueue_Create(pDrives, pAllocator));
  else
    mERROR_CHECK(mQueue_Clear(*pDrives));

  wchar_t driveLabels[MAX_PATH];
  const size_t length = GetLogicalDriveStringsW(mARRAYSIZE(driveLabels), driveLabels);

  if (length == 0 || length >= mARRAYSIZE(driveLabels))
  {
    const DWORD error = GetLastError();
    mUnused(error);

    mRETURN_RESULT(mR_InternalError);
  }

  wchar_t *nextDriveLabel = driveLabels;

  while (*nextDriveLabel != L'\0')
  {
    mString driveName;
    mDEFER_CALL(&driveName, mString_Destroy);
    mERROR_CHECK(mString_Create(&driveName, nextDriveLabel, pAllocator));

    mERROR_CHECK(mQueue_PushBack(*pDrives, std::move(driveName)));

    nextDriveLabel = &nextDriveLabel[lstrlenW(nextDriveLabel) + 1];
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_GetFreeStorageSpace, const mString &path, OUT size_t *pSize)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSize == nullptr, mR_ArgumentNull);
  mERROR_IF(path.hasFailed || path.bytes <= 1, mR_InvalidParameter);

  wchar_t wPath[MAX_PATH + 1];

  mERROR_CHECK(mString_ToWideString(path, wPath, mARRAYSIZE(wPath)));

  ULARGE_INTEGER freeBytesAvailableToCaller;

  if (FALSE == GetDiskFreeSpaceExW(wPath, &freeBytesAvailableToCaller, nullptr, nullptr))
  {
    const DWORD error = GetLastError();
    mUnused(error);

    mRETURN_RESULT(mR_InternalError);
  }

  *pSize = freeBytesAvailableToCaller.QuadPart;

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_GetAbsoluteDirectoryPath, OUT mString *pAbsolutePath, const mString &directoryPath)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAbsolutePath == nullptr, mR_ArgumentNull);
  mERROR_IF(directoryPath.hasFailed, mR_InvalidParameter);

  wchar_t folderPath[MAX_PATH];
  mERROR_CHECK(mString_ToWideString(directoryPath, folderPath, mARRAYSIZE(folderPath)));

  wchar_t absolutePath[MAX_PATH];
  const DWORD length = GetFullPathNameW(folderPath, mARRAYSIZE(absolutePath), absolutePath, nullptr);

  if (length == 0)
  {
    const DWORD error = GetLastError();
    mUnused(error);

    mRETURN_RESULT(mR_InternalError);
  }

  mERROR_CHECK(mString_Create(pAbsolutePath, absolutePath, length + 1, directoryPath.pAllocator));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_GetAbsoluteFilePath, OUT mString *pAbsolutePath, const mString &filePath)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAbsolutePath == nullptr, mR_ArgumentNull);
  mERROR_IF(filePath.hasFailed, mR_InvalidParameter);

  wchar_t wfilepath[MAX_PATH];
  mERROR_CHECK(mString_ToWideString(filePath, wfilepath, mARRAYSIZE(wfilepath)));

  wchar_t absolutePath[MAX_PATH];
  const DWORD length = GetFullPathNameW(wfilepath, mARRAYSIZE(absolutePath), absolutePath, nullptr);

  if (length == 0)
  {
    const DWORD error = GetLastError();
    mUnused(error);

    mRETURN_RESULT(mR_InternalError);
  }

  mERROR_CHECK(mString_Create(pAbsolutePath, absolutePath, length + 1, filePath.pAllocator));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_LaunchFile, const mString &filename)
{
  mFUNCTION_SETUP();

  mERROR_IF(filename.hasFailed || filename.bytes <= 1, mR_InvalidParameter);

  wchar_t wfilename[MAX_PATH];
  mERROR_CHECK(mString_ToWideString(filename, wfilename, mARRAYSIZE(wfilename)));

  const size_t result = reinterpret_cast<size_t>(ShellExecuteW(nullptr, nullptr, wfilename, nullptr, nullptr, SW_SHOW));

  if (result <= 32)
  {
    switch (result)
    {
    case ERROR_FILE_NOT_FOUND:
    case ERROR_PATH_NOT_FOUND:
    case SE_ERR_ASSOCINCOMPLETE:
      mRETURN_RESULT(mR_ResourceNotFound);

    case SE_ERR_NOASSOC:
      mRETURN_RESULT(mR_NotSupported);

    case SE_ERR_ACCESSDENIED:
    case SE_ERR_SHARE:
      mRETURN_RESULT(mR_InsufficientPrivileges);

    default:
      mRETURN_RESULT(mR_InternalError);
    }
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_LaunchApplication, const mString &applicationFilename, const mString &arguments, const mString &workingDirectory)
{
  mFUNCTION_SETUP();

  mERROR_IF(applicationFilename.hasFailed || arguments.hasFailed || workingDirectory.hasFailed, mR_InvalidParameter);

  mAllocator *pAllocator = &mDefaultTempAllocator;

  wchar_t *wPath = nullptr;
  size_t wPathCount = 0;

  mERROR_CHECK(mString_GetRequiredWideStringCount(applicationFilename, &wPathCount));
  mDEFER(mAllocator_FreePtr(pAllocator, &wPath));
  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &wPath, wPathCount));
  mERROR_CHECK(mString_ToWideString(applicationFilename, wPath, wPathCount));

  wchar_t *wArguments = nullptr;
  size_t wArgumentsCount = 0;

  mERROR_CHECK(mString_GetRequiredWideStringCount(arguments, &wArgumentsCount));
  mDEFER(mAllocator_FreePtr(pAllocator, &wArguments));
  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &wArguments, wArgumentsCount));
  mERROR_CHECK(mString_ToWideString(arguments, wArguments, wArgumentsCount));

  wchar_t *wWorkingDirectory = nullptr;
  size_t wWorkingDirectoryCount = 0;

  mERROR_CHECK(mString_GetRequiredWideStringCount(workingDirectory, &wWorkingDirectoryCount));
  mDEFER(mAllocator_FreePtr(pAllocator, &wWorkingDirectory));
  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &wWorkingDirectory, wWorkingDirectoryCount));
  mERROR_CHECK(mString_ToWideString(workingDirectory, wWorkingDirectory, wWorkingDirectoryCount));

  const size_t result = reinterpret_cast<size_t>(ShellExecuteW(nullptr, nullptr, wPath, wArguments, wWorkingDirectory, SW_SHOW));

  if (result <= 32)
  {
    switch (result)
    {
    case ERROR_FILE_NOT_FOUND:
    case ERROR_PATH_NOT_FOUND:
    case SE_ERR_ASSOCINCOMPLETE:
      mRETURN_RESULT(mR_ResourceNotFound);

    case SE_ERR_NOASSOC:
      mRETURN_RESULT(mR_NotSupported);

    case SE_ERR_ACCESSDENIED:
    case SE_ERR_SHARE:
      mRETURN_RESULT(mR_InsufficientPrivileges);

    default:
      mRETURN_RESULT(mR_InternalError);
    }
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_CreateShortcut, const mString &path, const mString &targetDestination, const mString &arguments, const mString &workingDirectory, const mString &description, const mString &iconLocation)
{
  mFUNCTION_SETUP();

  mERROR_IF(path.hasFailed || targetDestination.hasFailed || arguments.hasFailed || workingDirectory.hasFailed || description.hasFailed || iconLocation.hasFailed, mR_InvalidParameter);

  mAllocator *pAllocator = &mDefaultTempAllocator;

  wchar_t *wPath = nullptr;
  size_t wPathCount = 0;

  mERROR_CHECK(mString_GetRequiredWideStringCount(path, &wPathCount));
  mDEFER(mAllocator_FreePtr(pAllocator, &wPath));
  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &wPath, wPathCount));
  mERROR_CHECK(mString_ToWideString(path, wPath, wPathCount));
  
  wchar_t *wTargetDestination = nullptr;
  size_t wTargetDestinationCount = 0;

  mERROR_CHECK(mString_GetRequiredWideStringCount(targetDestination, &wTargetDestinationCount));
  mDEFER(mAllocator_FreePtr(pAllocator, &wTargetDestination));
  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &wTargetDestination, wTargetDestinationCount));
  mERROR_CHECK(mString_ToWideString(targetDestination, wTargetDestination, wTargetDestinationCount));

  wchar_t *wArguments = nullptr;
  size_t wArgumentsCount = 0;

  mERROR_CHECK(mString_GetRequiredWideStringCount(arguments, &wArgumentsCount));
  mDEFER(mAllocator_FreePtr(pAllocator, &wArguments));
  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &wArguments, wArgumentsCount));
  mERROR_CHECK(mString_ToWideString(arguments, wArguments, wArgumentsCount));

  wchar_t *wWorkingDirectory = nullptr;
  size_t wWorkingDirectoryCount = 0;

  mERROR_CHECK(mString_GetRequiredWideStringCount(workingDirectory, &wWorkingDirectoryCount));
  mDEFER(mAllocator_FreePtr(pAllocator, &wWorkingDirectory));
  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &wWorkingDirectory, wWorkingDirectoryCount));
  mERROR_CHECK(mString_ToWideString(workingDirectory, wWorkingDirectory, wWorkingDirectoryCount));

  wchar_t *wDescription = nullptr;
  size_t wDescriptionCount = 0;

  mERROR_CHECK(mString_GetRequiredWideStringCount(description, &wDescriptionCount));
  mDEFER(mAllocator_FreePtr(pAllocator, &wDescription));
  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &wDescription, wDescriptionCount));
  mERROR_CHECK(mString_ToWideString(description, wDescription, wDescriptionCount));

  wchar_t *wIconLocation = nullptr;
  size_t wIconLocationCount = 0;

  mERROR_CHECK(mString_GetRequiredWideStringCount(iconLocation, &wIconLocationCount));
  mDEFER(mAllocator_FreePtr(pAllocator, &wIconLocation));
  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &wIconLocation, wIconLocationCount));
  mERROR_CHECK(mString_ToWideString(iconLocation, wIconLocation, wIconLocationCount));

  IShellLinkW *pShellLink = nullptr;

  mDEFER(
    if (pShellLink != nullptr)
      pShellLink->Release()
  );

  HRESULT hr = CoCreateInstance(CLSID_ShellLink, NULL, CLSCTX_ALL, IID_IShellLink, (void **)&pShellLink);
  mERROR_IF(FAILED(hr), mR_InternalError);

  mERROR_IF(FAILED(hr = pShellLink->SetPath(wTargetDestination)), mR_InternalError);

  if (wArgumentsCount > 1)
    mERROR_IF(FAILED(hr = pShellLink->SetArguments(wArguments)), mR_InternalError);

  if (wWorkingDirectoryCount > 1)
    mERROR_IF(FAILED(hr = pShellLink->SetWorkingDirectory(wWorkingDirectory)), mR_InternalError);

  if (wDescriptionCount > 1)
    mERROR_IF(FAILED(hr = pShellLink->SetDescription(wDescription)), mR_InternalError);

  if (wIconLocationCount > 1)
    mERROR_IF(FAILED(hr = pShellLink->SetIconLocation(wIconLocation, 0)), mR_InternalError) ;

  IPersistFile *pPersistFile = nullptr;
  
  mDEFER(
    if (pPersistFile != nullptr)
      pPersistFile->Release()
  );

  hr = pShellLink->QueryInterface(IID_IPersistFile, (void **)&pPersistFile);
  mERROR_IF(FAILED(hr), mR_InternalError);

  hr = pPersistFile->Save(wPath, TRUE);

  if (hr == HRESULT_FROM_WIN32(ERROR_PATH_NOT_FOUND))
  {
    wchar_t parentDirectory[MAX_PATH];
    mERROR_CHECK(mStringCopy(parentDirectory, mARRAYSIZE(parentDirectory), wPath, MAX_PATH));

#if (NTDDI_VERSION >= NTDDI_WIN8)
    // Requires `pathcch.h` && `Pathcch.lib`.
    mERROR_IF(FAILED(hr = PathCchRemoveFileSpec(parentDirectory, mARRAYSIZE(parentDirectory))), mR_InternalError);
    mERROR_IF(hr == S_FALSE, mR_InvalidParameter);
#else
    // deprecated since windows 8.
    mERROR_IF(PathRemoveFileSpecW(parentDirectory), mR_InvalidParameter);
#endif

    mERROR_CHECK(mFile_CreateDirectoryRecursive_Internal(parentDirectory));

    hr = pPersistFile->Save(wPath, TRUE);
  }

  mERROR_IF(FAILED(hr), mR_InternalError);

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_GrantAccessToAllUsers, const mString &path)
{
  mFUNCTION_SETUP();

  mERROR_IF(path.hasFailed || path.bytes <= 1, mR_InvalidParameter);

  wchar_t wPath[MAX_PATH];
  mERROR_CHECK(mString_ToWideString(path, wPath, mARRAYSIZE(wPath)));

  HANDLE fileHandle = CreateFileW(wPath, READ_CONTROL | WRITE_DAC, 0, nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
  mDEFER(CloseHandle(fileHandle));

  if (fileHandle == INVALID_HANDLE_VALUE)
  {
    const DWORD error = GetLastError();

    switch (error)
    {
    case ERROR_FILE_NOT_FOUND:
    case ERROR_PATH_NOT_FOUND:
      mRETURN_RESULT(mR_ResourceNotFound);

    case ERROR_ACCESS_DENIED:
      mRETURN_RESULT(mR_InsufficientPrivileges);

    default:
      mRETURN_RESULT(mR_InternalError);
    }
  }

  ACL *pOldDACL = nullptr;
  SECURITY_DESCRIPTOR *pSD = nullptr;
  
  mDEFER(
    LocalFree(pSD);
  );

  mERROR_IF(ERROR_SUCCESS != GetSecurityInfo(fileHandle, SE_FILE_OBJECT, DACL_SECURITY_INFORMATION, nullptr, nullptr, &pOldDACL, nullptr, reinterpret_cast<void **>(&pSD)), mR_InternalError);

  PSID pSid = nullptr;
  mDEFER(FreeSid(pSid));

  SID_IDENTIFIER_AUTHORITY authNt = SECURITY_NT_AUTHORITY;

  if (0 == AllocateAndInitializeSid(&authNt, 2, SECURITY_BUILTIN_DOMAIN_RID, DOMAIN_ALIAS_RID_USERS, 0, 0, 0, 0, 0, 0, &pSid))
  {
    const DWORD error = GetLastError();
    mUnused(error);

    mRETURN_RESULT(mR_InternalError);
  }

  EXPLICIT_ACCESS ea = { 0 };
  ea.grfAccessMode = GRANT_ACCESS;
  ea.grfAccessPermissions = GENERIC_ALL;
  ea.grfInheritance = CONTAINER_INHERIT_ACE | OBJECT_INHERIT_ACE;
  ea.Trustee.TrusteeType = TRUSTEE_IS_GROUP;
  ea.Trustee.TrusteeForm = TRUSTEE_IS_SID;
  ea.Trustee.ptstrName = (LPTSTR)pSid;

  ACL *pNewDACL = nullptr;
  mDEFER(LocalFree(pNewDACL));
  mERROR_IF(ERROR_SUCCESS != SetEntriesInAclW(1, &ea, pOldDACL, &pNewDACL), mR_InternalError);

  if (pNewDACL)
    mERROR_IF(ERROR_SUCCESS != SetSecurityInfo(fileHandle, SE_FILE_OBJECT, DACL_SECURITY_INFORMATION, nullptr, nullptr, pNewDACL, nullptr), mR_InternalError);

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_GetDriveFromFilePath, OUT mString *pDrivePath, const mString &filePath)
{
  mFUNCTION_SETUP();

  mERROR_IF(pDrivePath == nullptr, mR_ArgumentNull);
  mERROR_IF(filePath.hasFailed, mR_InvalidParameter);

  wchar_t wfilepath[MAX_PATH];
  mERROR_CHECK(mString_ToWideString(filePath, wfilepath, mARRAYSIZE(wfilepath)));

  wchar_t volumePath[MAX_PATH];

  if (0 == GetVolumePathNameW(wfilepath, volumePath, mARRAYSIZE(volumePath)))
  {
    const DWORD error = GetLastError();
    mUnused(error);

    mRETURN_RESULT(mR_InternalError);
  }

  mERROR_CHECK(mString_Create(pDrivePath, volumePath, mARRAYSIZE(volumePath), filePath.pAllocator));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_GetDriveInfo, const mString &drivePath, OUT mDriveInfo *pDriveInfo, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(drivePath.hasFailed, mR_InvalidParameter);
  mERROR_IF(pDriveInfo == nullptr, mR_ArgumentNull);

  wchar_t wDrivePath[MAX_PATH];
  mERROR_CHECK(mString_ToWideString(drivePath, wDrivePath, mARRAYSIZE(wDrivePath)));

  wchar_t driveName[MAX_PATH + 1];
  wchar_t volumeTypeName[MAX_PATH + 1]; // FAT, NTFS, ...
  const BOOL result = GetVolumeInformationW(wDrivePath, driveName, mARRAYSIZE(driveName), nullptr, nullptr, nullptr, volumeTypeName, mARRAYSIZE(volumeTypeName));

  if (result)
    mERROR_CHECK(mString_Create(&pDriveInfo->driveName, driveName, pAllocator));

  const UINT driveType = GetDriveTypeW(wDrivePath);

  switch (driveType)
  {
  case DRIVE_UNKNOWN:
  case DRIVE_NO_ROOT_DIR:
  default:
    pDriveInfo->driveType = mDT_Unknown;
    break;

  case DRIVE_REMOVABLE:
    pDriveInfo->driveType = mDT_Removable;
    break;

  case DRIVE_FIXED:
    pDriveInfo->driveType = mDT_NonRemovable;
    break;

  case DRIVE_REMOTE:
    pDriveInfo->driveType = mDT_Remote;
    break;

  case DRIVE_CDROM:
    pDriveInfo->driveType = mDT_CDRom;
    break;

  case DRIVE_RAMDISK:
    pDriveInfo->driveType = mDT_RamDisk;
    break;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_GetFolderSize, const mString &directoryPath, OUT size_t *pFolderSize)
{
  mFUNCTION_SETUP();

  mERROR_IF(directoryPath.hasFailed, mR_InvalidParameter);
  mERROR_IF(pFolderSize == nullptr, mR_ArgumentNull);

  mPtr<mQueue<mFileInfo>> fileInfo;
  mERROR_CHECK(mFile_GetDirectoryContents(directoryPath, true, &fileInfo, &mDefaultTempAllocator));

  *pFolderSize = 0;

  for (const auto &_file : fileInfo->Iterate())
    *pFolderSize += _file.size;

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mFile_CreateDirectoryRecursive_Internal, const wchar_t *directoryPath)
{
  mFUNCTION_SETUP();

  if (0 == CreateDirectoryW(directoryPath, nullptr))
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

      mERROR_CHECK(mFile_CreateDirectoryRecursive_Internal(parentDirectory));

      if (0 == CreateDirectoryW(directoryPath, nullptr))
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

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mFileWriter_Destroy_Internal, IN_OUT mFileWriter *pWriter);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mFileWriter_Create, OUT mPtr<mFileWriter> *pWriter, IN mAllocator *pAllocator, const mString &filename)
{
  mFUNCTION_SETUP();

  mERROR_IF(pWriter == nullptr, mR_ArgumentNull);
  mERROR_IF(filename.hasFailed || filename.bytes <= 1, mR_InvalidParameter);

#ifdef mPLATFORM_WINDOWS
  wchar_t wFilename[MAX_PATH + 1];
  mERROR_CHECK(mString_ToWideString(filename, wFilename, mARRAYSIZE(wFilename)));

  HANDLE file = CreateFileW(wFilename, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  DWORD error = GetLastError(); // Might be `ERROR_ALREADY_EXISTS` even if the `file` is valid.

  if (file == INVALID_HANDLE_VALUE && error == ERROR_PATH_NOT_FOUND)
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

    mERROR_CHECK(mFile_CreateDirectoryRecursive_Internal(parentDirectory));

    file = CreateFileW(wFilename, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    error = GetLastError(); // Might be `ERROR_ALREADY_EXISTS` even if the `fileHandle` is valid.
  }

  mERROR_IF(file == INVALID_HANDLE_VALUE, mR_InternalError);

  mDEFER(CloseHandle(file));
#else
  FILE *pFile = fopen(filename.c_str(), "wb");
  
  mERROR_IF(pFile == nullptr, mR_IOFailure);

  mDEFER(fclose(pFile));
#endif

  mERROR_CHECK(mSharedPointer_Allocate<mFileWriter>(pWriter, pAllocator, [](mFileWriter *pData) { mFileWriter_Destroy_Internal(pData); }, 1));

#ifdef mPLATFORM_WINDOWS
  (*pWriter)->file = file;

  file = INVALID_HANDLE_VALUE;
#else
  (*pWriter)->pFile = pFile;

  pFile = nullptr;
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mFileWriter_Create, OUT mUniqueContainer<mFileWriter> *pWriter, const mString &filename)
{
  mFUNCTION_SETUP();

  mERROR_IF(pWriter == nullptr, mR_ArgumentNull);
  mERROR_IF(filename.hasFailed || filename.bytes <= 1, mR_InvalidParameter);

#ifdef mPLATFORM_WINDOWS
  wchar_t filenameW[MAX_PATH + 1];
  mERROR_CHECK(mString_ToWideString(filename, filenameW, mARRAYSIZE(filenameW)));

  HANDLE file = CreateFileW(filenameW, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);

  if (file == INVALID_HANDLE_VALUE)
  {
    const DWORD error = GetLastError();
    mUnused(error);

    mRETURN_RESULT(mR_IOFailure);
  }
#else
  FILE *pFile = fopen(filename.c_str(), "wb");

  mERROR_IF(pFile == nullptr, mR_IOFailure);
#endif

  mUniqueContainer<mFileWriter>::CreateWithCleanupFunction(pWriter, [](mFileWriter *pData) { mFileWriter_Destroy_Internal(pData); });

#ifdef mPLATFORM_WINDOWS
  (*pWriter)->file = file;

  file = INVALID_HANDLE_VALUE;
#else
  (*pWriter)->pFile = pFile;

  pFile = nullptr;
#endif

  mRETURN_SUCCESS();
}

mFUNCTION(mFileWriter_Destroy, IN_OUT mPtr<mFileWriter> *pWriter)
{
  mFUNCTION_SETUP();

  mERROR_IF(pWriter == nullptr, mR_ArgumentNull);
  
  mERROR_CHECK(mSharedPointer_Destroy(pWriter));

  mRETURN_SUCCESS();
}

mFUNCTION(mFileWriter_WriteRaw, mPtr<mFileWriter> &writer, const uint8_t *pData, const size_t size)
{
  mFUNCTION_SETUP();

  mERROR_IF(writer == nullptr || pData == nullptr, mR_ArgumentNull);
  mERROR_IF(size == 0, mR_Success);

#ifdef mPLATFORM_WINDOWS
  size_t sizeRemaining = size;
  size_t offset = 0;

  do
  {
    const size_t bytesToWrite = mMin(sizeRemaining, (size_t)MAXDWORD);
    DWORD bytesWritten = 0;

    if (FALSE == WriteFile(writer->file, reinterpret_cast<const void *>(pData + offset), (DWORD)bytesToWrite, &bytesWritten, nullptr))
    {
      writer->bytesWritten += bytesWritten;

      const DWORD error = GetLastError();
      mUnused(error);

      mRETURN_RESULT(mR_IOFailure);
    }

    mERROR_IF(bytesWritten == 0, mR_IOFailure);

    writer->bytesWritten += bytesWritten;
    offset += bytesWritten;
    sizeRemaining -= bytesWritten;

  } while (sizeRemaining > 0);
#else
  const size_t bytesWritten = fwrite(pData, sizeof(uint8_t), size, pWriter->pFile);

  writer->bytesWritten += bytesWritten;

  mERROR_IF(size != bytesWritten, mR_IOFailure);
#endif

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mFileWriter_Destroy_Internal, IN_OUT mFileWriter *pWriter)
{
  mFUNCTION_SETUP();

  mERROR_IF(pWriter == nullptr, mR_ArgumentNull);

#ifdef mPLATFORM_WINDOWS
  if (pWriter->file != nullptr)
    CloseHandle(pWriter->file);

  pWriter->file = nullptr;
#else
  if (pWriter->pFile != nullptr)
    fclose(pWriter->pFile);

  pWriter->pFile = nullptr;
#endif
  
  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

HRESULT mFile_CreateAndInitializeFileOperation_Internal(REFIID riid, void **ppFileOperation)
{
  *ppFileOperation = nullptr;

  // Create the IFileOperation object.
  IFileOperation *pfo;

  HRESULT hr = CoCreateInstance(__uuidof(FileOperation), NULL, CLSCTX_ALL, IID_PPV_ARGS(&pfo));

  if (SUCCEEDED(hr))
  {
    // Set the operation flags. Turn off all UI from being shown to the user during the operation. This includes error, confirmation and progress dialogs.
    hr = pfo->SetOperationFlags(FOF_NO_UI);

    if (SUCCEEDED(hr))
    {
      hr = pfo->QueryInterface(riid, ppFileOperation);
    }

    pfo->Release();
  }

  return hr;
}

mFUNCTION(mFile_GetKnownPath_Internal, OUT mString *pString, const GUID &guid)
{
  mFUNCTION_SETUP();

  mERROR_IF(pString == nullptr, mR_ArgumentNull);

  wchar_t *pBuffer = nullptr;

  mDEFER(
    if (pBuffer != nullptr)
      CoTaskMemFree(pBuffer)
      );

  HRESULT result = S_OK;
  mERROR_IF(FAILED(result = SHGetKnownFolderPath(guid, 0, nullptr, &pBuffer)), mR_InternalError);

  mString tempString;
  mERROR_CHECK(mString_Create(&tempString, pBuffer, &mDefaultTempAllocator));
  mERROR_CHECK(mString_ToDirectoryPath(pString, tempString));

  mRETURN_SUCCESS();
}

mFUNCTION(mFileInfo_FromFindDataWStruct_Internal, IN_OUT mFileInfo *pFileInfo, IN const WIN32_FIND_DATAW *pFileData, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();
  
  mERROR_CHECK(mString_Create(&pFileInfo->name, (wchar_t *)pFileData->cFileName, mARRAYSIZE(pFileData->cFileName), pAllocator));
  pFileInfo->size = ((size_t)pFileData->nFileSizeHigh * (size_t)(MAXDWORD + 1ULL)) + (size_t)pFileData->nFileSizeLow;
  pFileInfo->creationTimeStamp = ((size_t)pFileData->ftCreationTime.dwHighDateTime * (size_t)(MAXDWORD + 1ULL)) + (size_t)pFileData->ftCreationTime.dwLowDateTime;
  pFileInfo->lastAccessTimeStamp = ((size_t)pFileData->ftLastAccessTime.dwHighDateTime * (size_t)(MAXDWORD + 1ULL)) + (size_t)pFileData->ftLastAccessTime.dwLowDateTime;
  pFileInfo->lastWriteTimeStamp = ((size_t)pFileData->ftLastWriteTime.dwHighDateTime * (size_t)(MAXDWORD + 1ULL)) + (size_t)pFileData->ftLastWriteTime.dwLowDateTime;
  pFileInfo->isDirectory = !!(pFileData->dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY);
  pFileInfo->isHidden = !!(pFileData->dwFileAttributes & FILE_ATTRIBUTE_HIDDEN);
  pFileInfo->isOffline = !!(pFileData->dwFileAttributes & FILE_ATTRIBUTE_OFFLINE);
  pFileInfo->isReadonly = !!(pFileData->dwFileAttributes & FILE_ATTRIBUTE_READONLY);
  pFileInfo->isSystemResource = !!(pFileData->dwFileAttributes & (FILE_ATTRIBUTE_SYSTEM | FILE_ATTRIBUTE_DEVICE));

  mRETURN_SUCCESS();
}

mFUNCTION(mFileInfo_FromByHandleFileInformationStruct_Internal, IN_OUT mFileInfo *pFileInfo, IN const BY_HANDLE_FILE_INFORMATION *pFileData)
{
  mFUNCTION_SETUP();

  pFileInfo->size = ((size_t)pFileData->nFileSizeHigh * (size_t)(MAXDWORD + 1ULL)) + (size_t)pFileData->nFileSizeLow;
  pFileInfo->creationTimeStamp = ((size_t)pFileData->ftCreationTime.dwHighDateTime * (size_t)(MAXDWORD + 1ULL)) + (size_t)pFileData->ftCreationTime.dwLowDateTime;
  pFileInfo->lastAccessTimeStamp = ((size_t)pFileData->ftLastAccessTime.dwHighDateTime * (size_t)(MAXDWORD + 1ULL)) + (size_t)pFileData->ftLastAccessTime.dwLowDateTime;
  pFileInfo->lastWriteTimeStamp = ((size_t)pFileData->ftLastWriteTime.dwHighDateTime * (size_t)(MAXDWORD + 1ULL)) + (size_t)pFileData->ftLastWriteTime.dwLowDateTime;
  pFileInfo->isDirectory = !!(pFileData->dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY);
  pFileInfo->isHidden = !!(pFileData->dwFileAttributes & FILE_ATTRIBUTE_HIDDEN);
  pFileInfo->isOffline = !!(pFileData->dwFileAttributes & FILE_ATTRIBUTE_OFFLINE);
  pFileInfo->isReadonly = !!(pFileData->dwFileAttributes & FILE_ATTRIBUTE_READONLY);
  pFileInfo->isSystemResource = !!(pFileData->dwFileAttributes & (FILE_ATTRIBUTE_SYSTEM | FILE_ATTRIBUTE_DEVICE));

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mRegistry_WriteKey, const mString &keyUrl, const mString &value, OUT OPTIONAL bool *pNewlyCreated /* = nullptr */)
{
  mFUNCTION_SETUP();

  mString subAddress;
  subAddress.pAllocator = &mDefaultTempAllocator;

  const char classesRoot[] = "HKEY_CLASSES_ROOT\\";
  const char currentConfig[] = "HKEY_CURRENT_CONFIG\\";
  const char currentUser[] = "HKEY_CURRENT_USER\\";
  const char localMachine[] = "HKEY_LOCAL_MACHINE\\";
  const char users[] = "HKEY_USERS\\";

  HKEY parentKey = nullptr;

  if (keyUrl.StartsWith(classesRoot))
  {
    parentKey = HKEY_CLASSES_ROOT;

    mERROR_CHECK(mString_Substring(keyUrl, &subAddress, mARRAYSIZE(classesRoot) - 1));
  }
  else if (keyUrl.StartsWith(currentConfig))
  {
    parentKey = HKEY_CURRENT_CONFIG;

    mERROR_CHECK(mString_Substring(keyUrl, &subAddress, mARRAYSIZE(currentConfig) - 1));
  }
  else if (keyUrl.StartsWith(currentUser))
  {
    parentKey = HKEY_CURRENT_USER;

    mERROR_CHECK(mString_Substring(keyUrl, &subAddress, mARRAYSIZE(currentUser) - 1));
  }
  else if (keyUrl.StartsWith(localMachine))
  {
    parentKey = HKEY_LOCAL_MACHINE;

    mERROR_CHECK(mString_Substring(keyUrl, &subAddress, mARRAYSIZE(localMachine) - 1));
  }
  else if (keyUrl.StartsWith(users))
  {
    parentKey = HKEY_USERS;

    mERROR_CHECK(mString_Substring(keyUrl, &subAddress, mARRAYSIZE(users) - 1));
  }
  else
  {
    mRETURN_RESULT(mR_InvalidParameter);
  }

  mString pathString;
  pathString.pAllocator = &mDefaultTempAllocator;
  
  mString valueName;
  valueName.pAllocator = &mDefaultTempAllocator;

  size_t index = (size_t)-1;
  size_t lastSlashIndex = (size_t)-1;
  const mchar_t slashCodePoint = mToChar<2>("\\");

  for (const auto &&_char : subAddress)
  {
    ++index;

    if (_char.codePoint == slashCodePoint)
      lastSlashIndex = index;
  }

  if (lastSlashIndex == (size_t)-1)
  {
    mERROR_CHECK(mString_Create(&pathString, subAddress, &mDefaultTempAllocator));
    mERROR_CHECK(mString_Create(&valueName, "", 1, &mDefaultTempAllocator));
  }
  else
  {
    mERROR_CHECK(mString_Substring(subAddress, &pathString, 0, lastSlashIndex));

    if (lastSlashIndex != subAddress.count)
      mERROR_CHECK(mString_Substring(subAddress, &valueName, lastSlashIndex + 1, subAddress.count - (lastSlashIndex + 2)));
    else
      mERROR_CHECK(mString_Create(&valueName, "", 1, &mDefaultTempAllocator));
  }

  wchar_t *wPathOrValue = nullptr;
  size_t wPathOrValueCount = 0;
  mERROR_CHECK(mString_GetRequiredWideStringCount(pathString, &wPathOrValueCount));

  mAllocator *pAllocator = &mDefaultTempAllocator;
  mDEFER(mAllocator_FreePtr(pAllocator, &wPathOrValue));
  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &wPathOrValue, wPathOrValueCount));

  mERROR_CHECK(mString_ToWideString(pathString, wPathOrValue, wPathOrValueCount));

  wchar_t *wValueName = nullptr;
  size_t wValueNameCount = 0;
  mERROR_CHECK(mString_GetRequiredWideStringCount(valueName, &wValueNameCount));

  mDEFER(mAllocator_FreePtr(pAllocator, &wValueName));
  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &wValueName, wValueNameCount));

  mERROR_CHECK(mString_ToWideString(valueName, wValueName, wValueNameCount));

  HKEY key = nullptr;
  DWORD disposition = 0;

  LSTATUS result = RegCreateKeyExW(parentKey, wPathOrValue, 0, NULL, 0, KEY_ALL_ACCESS, NULL, &key, &disposition);
  mDEFER(if (key != nullptr) RegCloseKey(key));
  mERROR_IF(result != ERROR_SUCCESS, mR_InternalError);

  if (pNewlyCreated != nullptr)
    *pNewlyCreated = (disposition == REG_CREATED_NEW_KEY);

  mERROR_CHECK(mString_GetRequiredWideStringCount(value, &wPathOrValueCount));

  mERROR_CHECK(mAllocator_Reallocate(pAllocator, &wPathOrValue, wPathOrValueCount));
  mERROR_CHECK(mString_ToWideString(value, wPathOrValue, wPathOrValueCount));

  result = RegSetValueExW(key, wValueName, 0, REG_SZ, reinterpret_cast<const BYTE *>(wPathOrValue), (DWORD)wPathOrValueCount * sizeof(wchar_t));
  mERROR_IF(result != ERROR_SUCCESS, mR_InternalError);

  mRETURN_SUCCESS();
}

mFUNCTION(mRegistry_ReadKey, const mString &keyUrl, OUT mString *pValue)
{
  mFUNCTION_SETUP();

  mERROR_IF(pValue == nullptr, mR_ArgumentNull);

  mString subAddress;
  subAddress.pAllocator = &mDefaultTempAllocator;

  const char classesRoot[] = "HKEY_CLASSES_ROOT\\";
  const char currentConfig[] = "HKEY_CURRENT_CONFIG\\";
  const char currentUser[] = "HKEY_CURRENT_USER\\";
  const char localMachine[] = "HKEY_LOCAL_MACHINE\\";
  const char users[] = "HKEY_USERS\\";

  HKEY parentKey = nullptr;

  if (keyUrl.StartsWith(classesRoot))
  {
    parentKey = HKEY_CLASSES_ROOT;

    mERROR_CHECK(mString_Substring(keyUrl, &subAddress, mARRAYSIZE(classesRoot) - 1));
  }
  else if (keyUrl.StartsWith(currentConfig))
  {
    parentKey = HKEY_CURRENT_CONFIG;

    mERROR_CHECK(mString_Substring(keyUrl, &subAddress, mARRAYSIZE(currentConfig) - 1));
  }
  else if (keyUrl.StartsWith(currentUser))
  {
    parentKey = HKEY_CURRENT_USER;

    mERROR_CHECK(mString_Substring(keyUrl, &subAddress, mARRAYSIZE(currentUser) - 1));
  }
  else if (keyUrl.StartsWith(localMachine))
  {
    parentKey = HKEY_LOCAL_MACHINE;

    mERROR_CHECK(mString_Substring(keyUrl, &subAddress, mARRAYSIZE(localMachine) - 1));
  }
  else if (keyUrl.StartsWith(users))
  {
    parentKey = HKEY_USERS;

    mERROR_CHECK(mString_Substring(keyUrl, &subAddress, mARRAYSIZE(users) - 1));
  }
  else
  {
    mRETURN_RESULT(mR_InvalidParameter);
  }

  mString pathString;
  pathString.pAllocator = &mDefaultTempAllocator;

  mString valueName;
  valueName.pAllocator = &mDefaultTempAllocator;

  size_t index = (size_t)-1;
  size_t lastSlashIndex = (size_t)-1;
  const mchar_t slashCodePoint = mToChar<2>("\\");

  for (const auto &&_char : subAddress)
  {
    ++index;

    if (_char.codePoint == slashCodePoint)
      lastSlashIndex = index;
  }

  if (lastSlashIndex == (size_t)-1)
  {
    mERROR_CHECK(mString_Create(&pathString, subAddress, &mDefaultTempAllocator));
    mERROR_CHECK(mString_Create(&valueName, "", 1, &mDefaultTempAllocator));
  }
  else
  {
    mERROR_CHECK(mString_Substring(subAddress, &pathString, 0, lastSlashIndex));

    if (lastSlashIndex != subAddress.count)
      mERROR_CHECK(mString_Substring(subAddress, &valueName, lastSlashIndex + 1, subAddress.count - (lastSlashIndex + 2)));
    else
      mERROR_CHECK(mString_Create(&valueName, "", 1, &mDefaultTempAllocator));
  }

  wchar_t *wPathOrValue = nullptr;
  size_t wPathStringCount = 0;
  mERROR_CHECK(mString_GetRequiredWideStringCount(pathString, &wPathStringCount));

  mAllocator *pAllocator = &mDefaultTempAllocator;
  mDEFER(mAllocator_FreePtr(pAllocator, &wPathOrValue));
  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &wPathOrValue, wPathStringCount));

  mERROR_CHECK(mString_ToWideString(pathString, wPathOrValue, wPathStringCount));

  wchar_t *wValueName = nullptr;
  size_t wValueNameCount = 0;
  mERROR_CHECK(mString_GetRequiredWideStringCount(valueName, &wValueNameCount));

  mDEFER(mAllocator_FreePtr(pAllocator, &wValueName));
  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &wValueName, wValueNameCount));

  mERROR_CHECK(mString_ToWideString(valueName, wValueName, wValueNameCount));

  HKEY key = nullptr;
  
  LSTATUS result = RegOpenKeyExW(parentKey, wPathOrValue, 0, KEY_QUERY_VALUE, &key);
  mDEFER(if (key != nullptr) RegCloseKey(key));
  mERROR_IF(result != ERROR_SUCCESS, mR_ResourceNotFound);

  DWORD type = 0;
  DWORD bytes = 0;

  result = RegQueryValueExW(key, wValueName, NULL, &type, NULL, &bytes);
  mERROR_IF(result == ERROR_FILE_NOT_FOUND, mR_ResourceNotFound);
  mERROR_IF(result != ERROR_SUCCESS, mR_InternalError);

  mERROR_IF(type != REG_SZ, mR_ResourceStateInvalid);

  if (bytes > wPathStringCount * sizeof(wchar_t))
  {
    wPathStringCount = (bytes + 1) / sizeof(wchar_t);
    mERROR_CHECK(mAllocator_Reallocate(pAllocator, &wPathOrValue, wPathStringCount));
  }

  result = RegQueryValueExW(key, wValueName, NULL, NULL, reinterpret_cast<BYTE *>(wPathOrValue), &bytes);
  mERROR_IF(result != ERROR_SUCCESS, mR_InternalError);

  mERROR_CHECK(mString_Create(pValue, wPathOrValue, wPathStringCount, pValue->pAllocator));

  mRETURN_SUCCESS();
}

mFUNCTION(mRegistry_DeleteKey, const mString &keyUrl)
{
  mFUNCTION_SETUP();

  mString subAddress;
  subAddress.pAllocator = &mDefaultTempAllocator;

  const char classesRoot[] = "HKEY_CLASSES_ROOT\\";
  const char currentConfig[] = "HKEY_CURRENT_CONFIG\\";
  const char currentUser[] = "HKEY_CURRENT_USER\\";
  const char localMachine[] = "HKEY_LOCAL_MACHINE\\";
  const char users[] = "HKEY_USERS\\";

  HKEY parentKey = nullptr;

  if (keyUrl.StartsWith(classesRoot))
  {
    parentKey = HKEY_CLASSES_ROOT;

    mERROR_CHECK(mString_Substring(keyUrl, &subAddress, mARRAYSIZE(classesRoot) - 1));
  }
  else if (keyUrl.StartsWith(currentConfig))
  {
    parentKey = HKEY_CURRENT_CONFIG;

    mERROR_CHECK(mString_Substring(keyUrl, &subAddress, mARRAYSIZE(currentConfig) - 1));
  }
  else if (keyUrl.StartsWith(currentUser))
  {
    parentKey = HKEY_CURRENT_USER;

    mERROR_CHECK(mString_Substring(keyUrl, &subAddress, mARRAYSIZE(currentUser) - 1));
  }
  else if (keyUrl.StartsWith(localMachine))
  {
    parentKey = HKEY_LOCAL_MACHINE;

    mERROR_CHECK(mString_Substring(keyUrl, &subAddress, mARRAYSIZE(localMachine) - 1));
  }
  else if (keyUrl.StartsWith(users))
  {
    parentKey = HKEY_USERS;

    mERROR_CHECK(mString_Substring(keyUrl, &subAddress, mARRAYSIZE(users) - 1));
  }
  else
  {
    mRETURN_RESULT(mR_InvalidParameter);
  }

  mString pathString;
  pathString.pAllocator = &mDefaultTempAllocator;

  mString valueName;
  valueName.pAllocator = &mDefaultTempAllocator;

  size_t index = (size_t)-1;
  size_t lastSlashIndex = (size_t)-1;
  const mchar_t slashCodePoint = mToChar<2>("\\");

  for (const auto &&_char : subAddress)
  {
    ++index;

    if (_char.codePoint == slashCodePoint)
      lastSlashIndex = index;
  }

  if (lastSlashIndex == (size_t)-1)
  {
    mERROR_CHECK(mString_Create(&pathString, subAddress, &mDefaultTempAllocator));
    mERROR_CHECK(mString_Create(&valueName, "", 1, &mDefaultTempAllocator));
  }
  else
  {
    mERROR_CHECK(mString_Substring(subAddress, &pathString, 0, lastSlashIndex));

    if (lastSlashIndex != subAddress.count)
      mERROR_CHECK(mString_Substring(subAddress, &valueName, lastSlashIndex + 1, subAddress.count - (lastSlashIndex + 2)));
    else
      mERROR_CHECK(mString_Create(&valueName, "", 1, &mDefaultTempAllocator));
  }

  wchar_t *wPath = nullptr;
  size_t wPathCount = 0;
  mERROR_CHECK(mString_GetRequiredWideStringCount(pathString, &wPathCount));

  mAllocator *pAllocator = &mDefaultTempAllocator;
  mDEFER(mAllocator_FreePtr(pAllocator, &wPath));
  mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &wPath, wPathCount));

  mERROR_CHECK(mString_ToWideString(pathString, wPath, wPathCount));

  HKEY key = nullptr;

  LSTATUS result = RegOpenKeyExW(parentKey, wPath, 0, KEY_ALL_ACCESS, &key);
  mDEFER(if (key != nullptr) RegCloseKey(key));
  mERROR_IF(result != ERROR_SUCCESS, mR_ResourceNotFound);

  if (valueName.bytes <= 1)
  {
    result = RegDeleteTreeW(key, nullptr);
    
    if (result != ERROR_SUCCESS)
    {
      switch (result)
      {
      case ERROR_ACCESS_DENIED:
        mRETURN_RESULT(mR_InsufficientPrivileges);

      case ERROR_PATH_NOT_FOUND:
      case ERROR_FILE_NOT_FOUND:
      case ERROR_NOT_FOUND:
        mRETURN_RESULT(mR_ResourceNotFound);

      default:
        mRETURN_RESULT(mR_InternalError);
      }
    }
  }
  else
  {
    wchar_t *wValueName = nullptr;
    size_t wValueNameCount = 0;
    mERROR_CHECK(mString_GetRequiredWideStringCount(valueName, &wValueNameCount));

    mDEFER(mAllocator_FreePtr(pAllocator, &wValueName));
    mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &wValueName, wValueNameCount));

    mERROR_CHECK(mString_ToWideString(valueName, wValueName, wValueNameCount));

    result = RegDeleteValueW(key, wValueName);

    if (result != ERROR_SUCCESS)
    {
      switch (result)
      {
      case ERROR_ACCESS_DENIED:
        mRETURN_RESULT(mR_InsufficientPrivileges);

      case ERROR_PATH_NOT_FOUND:
      case ERROR_FILE_NOT_FOUND:
      case ERROR_NOT_FOUND:
        mRETURN_RESULT(mR_ResourceNotFound);

      default:
        mRETURN_RESULT(mR_InternalError);
      }
    }
  }

  mRETURN_SUCCESS();
}
