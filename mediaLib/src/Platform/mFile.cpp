#include "mFile.h"

#include <sys\stat.h>

#include <shobjidl.h>
#include <shlobj.h>
#include <knownfolders.h>
#include <Shlwapi.h>

HRESULT mFile_CreateAndInitializeFileOperation_Internal(REFIID riid, void **ppFileOperation);
mFUNCTION(mFile_GetKnownPath_Internal, OUT mString *pString, const GUID &guid);
mFUNCTION(mFileInfo_FromFindDataWStruct_Internal, IN_OUT mFileInfo *pFileInfo, IN const WIN32_FIND_DATAW *pFileData, IN mAllocator *pAllocator);
mFUNCTION(mFileInfo_FromByHandleFileInformationStruct_Internal, IN_OUT mFileInfo *pFileInfo, IN const BY_HANDLE_FILE_INFORMATION *pFileData);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mFile_Exists, const mString &filename, OUT bool *pExists)
{
  mFUNCTION_SETUP();

  mERROR_IF(pExists == nullptr, mR_ArgumentNull);

  std::wstring wfilename;
  mERROR_CHECK(mString_ToWideString(filename, &wfilename));

  mERROR_CHECK(mFile_Exists(wfilename, pExists));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_Exists, const std::wstring &filename, OUT bool *pExists)
{
  mFUNCTION_SETUP();

  mERROR_IF(pExists == nullptr, mR_ArgumentNull);

  *pExists = (PathFileExistsW(filename.c_str()) != FALSE);

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_ReadAllBytes, const mString &filename, IN OPTIONAL mAllocator *pAllocator, OUT mArray<uint8_t> *pBytes)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mFile_ReadAllItems(filename, pAllocator, pBytes));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_ReadAllBytes, const std::wstring &filename, IN OPTIONAL mAllocator *pAllocator, OUT mArray<uint8_t> *pBytes)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mFile_ReadAllItems(filename, pAllocator, pBytes));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_ReadAllText, const std::wstring &filename, IN OPTIONAL mAllocator *pAllocator, OUT std::string *pText, const mFile_Encoding /* encoding = mF_E_UTF8 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pText == nullptr, mR_ArgumentNull);

  size_t count = 0;
  char *text = nullptr;

  mDEFER(mAllocator_FreePtr(nullptr, &text));
  mERROR_CHECK(mFile_ReadRaw(filename, &text, pAllocator, &count));
  text[count] = '\0';
  *pText = std::string(text);
  mERROR_CHECK(mAllocator_FreePtr(pAllocator, &text));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_ReadAllText, const mString &filename, IN OPTIONAL mAllocator *pAllocator, OUT mString *pText, const mFile_Encoding /* encoding = mF_E_UTF8 */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pText == nullptr, mR_ArgumentNull);

  std::wstring wstring;
  mERROR_CHECK(mString_ToWideString(filename, &wstring));

  size_t count = 0;
  char *text = nullptr;

  mDEFER(mAllocator_FreePtr(nullptr, &text));
  mERROR_CHECK(mFile_ReadRaw(wstring, &text, pAllocator, &count));

  mERROR_IF(count == 0, mR_ResourceNotFound);

  text[count] = '\0';
  *pText = mString(text);
  mERROR_CHECK(mAllocator_FreePtr(pAllocator, &text));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_WriteAllBytes, const std::wstring &filename, mArray<uint8_t> &bytes)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mFile_WriteRaw(filename, bytes.pData, bytes.count));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_WriteAllBytes, const mString &filename, mArray<uint8_t> &bytes)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mFile_WriteRaw(filename, bytes.pData, bytes.count));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_WriteAllText, const std::wstring &filename, const std::string &text, const mFile_Encoding /* encoding = mF_E_UTF8 */)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mFile_WriteRaw(filename, text.c_str(), text.length()));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_WriteAllText, const mString &filename, const mString &text, const mFile_Encoding /* encoding = mF_E_UTF8 */)
{
  mFUNCTION_SETUP();

  std::wstring wstring;
  mERROR_CHECK(mString_ToWideString(filename, &wstring));

  mERROR_CHECK(mFile_WriteRaw(wstring, text.c_str(), text.bytes - 1));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_CreateDirectory, const mString &folderPath)
{
  mFUNCTION_SETUP();

  mERROR_IF(folderPath.hasFailed, mR_InvalidParameter);

  mString path;
  mERROR_CHECK(mFile_GetAbsoluteDirectoryPath(&path, folderPath));

  mString pathWithoutLastSlash;
  mERROR_CHECK(mString_Substring(path, &pathWithoutLastSlash, 0, path.Count() - 2));

  std::wstring directoryName;
  mERROR_CHECK(mString_ToWideString(path, &directoryName));

  const int errorCode = SHCreateDirectoryExW(NULL, directoryName.c_str(), NULL);

  switch (errorCode)
  {
  case ERROR_SUCCESS:
    mRETURN_SUCCESS();

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

  std::wstring directoryName;
  mERROR_CHECK(mString_ToWideString(pathWithoutLastSlash, &directoryName));

  HRESULT hr = S_OK;

  wchar_t absolutePath[1024 * 4];
  const DWORD length = GetFullPathNameW(directoryName.c_str(), mARRAYSIZE(absolutePath), absolutePath, nullptr);
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

  std::wstring source;
  mERROR_CHECK(mString_ToWideString(sourceFileName, &source));

  std::wstring target;
  mERROR_CHECK(mString_ToWideString(destinationFileName, &target));

  const BOOL succeeded = CopyFileW(source.c_str(), target.c_str(), !overrideFileIfExistent);
  mERROR_IF(succeeded, mR_Success);

  const DWORD errorCode = GetLastError();

  switch (errorCode)
  {
  case ERROR_ACCESS_DENIED:
    mRETURN_RESULT(mR_ResourceAlreadyExists);

  case ERROR_FILE_NOT_FOUND:
    mRETURN_RESULT(mR_ResourceNotFound);

  default:
    mRETURN_RESULT(mR_Failure);
  }
}

mFUNCTION(mFile_Move, const mString &destinationFileName, const mString &sourceFileName, const bool overrideFileIfExistent /* = true */)
{
  mFUNCTION_SETUP();

  bool destinationFileExists = false;
  mERROR_CHECK(mFile_Exists(destinationFileName, &destinationFileExists));

  if (destinationFileExists)
  {
    mERROR_IF(!overrideFileIfExistent, mR_ResourceAlreadyExists);
    mERROR_CHECK(mFile_Delete(destinationFileName));
  }

  std::wstring source;
  mERROR_CHECK(mString_ToWideString(sourceFileName, &source));

  std::wstring target;
  mERROR_CHECK(mString_ToWideString(destinationFileName, &target));

  const BOOL succeeded = MoveFileW(source.c_str(), target.c_str());
  mERROR_IF(succeeded, mR_Success);

  const DWORD errorCode = GetLastError();
  mUnused(errorCode);

  mRETURN_RESULT(mR_Failure);
}

mFUNCTION(mFile_Delete, const mString &fileName)
{
  mFUNCTION_SETUP();

  std::wstring wfile;
  mERROR_CHECK(mString_ToWideString(fileName, &wfile));

  const BOOL succeeded = DeleteFileW(wfile.c_str());
  mERROR_IF(succeeded, mR_Success);

  const DWORD errorCode = GetLastError();

  switch (errorCode)
  {
  case ERROR_FILE_NOT_FOUND:
    mRETURN_RESULT(mR_ResourceNotFound);

  default:
    mRETURN_RESULT(mR_Failure);
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

  std::wstring wPath;
  mERROR_CHECK(mString_ToWideString(workingDirectory, &wPath));

  if (!SetCurrentDirectoryW(wPath.c_str()))
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

//#if (NTDDI_VERSION >= NTDDI_WIN8)
//  HRESULT hr = S_OK;
//
//  // Requires `pathcch.h` && `Pathcch.lib`.
//  mERROR_IF(FAILED(hr = PathCchRemoveFileSpec(wPath, mARRAYSIZE(wPath))), mR_InternalError); // S_OK, or S_FALSE if nothing was removed.
//#else
  const BOOL result = PathRemoveFileSpecW(wPath); // deprecated since windows 8.
  mUnused(result); // result is true if something was removed, however we don't actually care.
//#endif

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

mFUNCTION(mFile_GetDirectoryContents, const mString &directoryPath, OUT mPtr<mQueue<mFileInfo>> *pFiles, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pFiles == nullptr, mR_ArgumentNull);
  mERROR_IF(directoryPath.hasFailed, mR_InvalidParameter);

  mString actualFolderPath;
  mERROR_CHECK(mString_ToDirectoryPath(&actualFolderPath, directoryPath));

  mERROR_CHECK(mString_Append(actualFolderPath, "*"));

  std::wstring folderPath;
  mERROR_CHECK(mString_ToWideString(actualFolderPath, &folderPath));

  wchar_t absolutePath[1024 * 4];
  const DWORD length = GetFullPathNameW(folderPath.c_str(), mARRAYSIZE(absolutePath), absolutePath, nullptr);
  mERROR_IF(length == 0, mR_InternalError);

  if (*pFiles == nullptr)
    mERROR_CHECK(mQueue_Create(pFiles, pAllocator));
  else
    mERROR_CHECK(mQueue_Clear(*pFiles));

  WIN32_FIND_DATAW fileData;
  const HANDLE handle = FindFirstFileW(absolutePath, &fileData);

  if (handle == INVALID_HANDLE_VALUE)
  {
    const DWORD error = GetLastError();

    switch (error)
    {
    case ERROR_FILE_NOT_FOUND:
      mRETURN_RESULT(mR_ResourceNotFound);

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

  std::wstring wfilename;
  mERROR_CHECK(mString_ToWideString(filename, &wfilename));

  HANDLE fileHandle = CreateFileW(wfilename.c_str(), STANDARD_RIGHTS_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);

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

  mERROR_CHECK(mQueue_Create(pDrives, pAllocator));

  wchar_t driveLabels[1024];
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

mFUNCTION(mFile_GetAbsoluteDirectoryPath, OUT mString *pAbsolutePath, const mString &directoryPath)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAbsolutePath == nullptr, mR_ArgumentNull);
  mERROR_IF(directoryPath.hasFailed, mR_InvalidParameter);

  mString actualFolderPath;
  mERROR_CHECK(mString_ToDirectoryPath(&actualFolderPath, directoryPath));

  std::wstring folderPath;
  mERROR_CHECK(mString_ToWideString(actualFolderPath, &folderPath));

  wchar_t absolutePath[1024 * 4];
  const DWORD length = GetFullPathNameW(folderPath.c_str(), mARRAYSIZE(absolutePath), absolutePath, nullptr);

  mERROR_CHECK(mString_Create(pAbsolutePath, absolutePath, length + 1, directoryPath.pAllocator));

  mRETURN_SUCCESS();
}

mFUNCTION(mFile_GetAbsoluteFilePath, OUT mString *pAbsolutePath, const mString &filePath)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAbsolutePath == nullptr, mR_ArgumentNull);
  mERROR_IF(filePath.hasFailed, mR_InvalidParameter);

  std::wstring folderPath;
  mERROR_CHECK(mString_ToWideString(filePath, &folderPath));

  wchar_t absolutePath[1024 * 4];
  const DWORD length = GetFullPathNameW(folderPath.c_str(), mARRAYSIZE(absolutePath), absolutePath, nullptr);

  mERROR_CHECK(mString_Create(pAbsolutePath, absolutePath, length + 1, filePath.pAllocator));

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
