#ifndef mFile_h__
#define mFile_h__

#include "mediaLib.h"
#include "mArray.h"
#include "mQueue.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "A+934z9PYhluwBWnEIhoe2p1q13IqcW4R6uOt5lbuTbxIrC3XjhrgWPOciisAzSUpD7SGPKFkcXc1tan"
#endif

enum class mFile_Encoding
{
  mF_E_UTF8,
  mF_E_UTF16,
};

#define mFile_FileExists mFile_Exists
mFUNCTION(mFile_Exists, const mString &filename, OUT bool *pExists);
mFUNCTION(mFile_Exists, IN const wchar_t *filename, OUT bool *pExists);

mFUNCTION(mFile_DirectoryExists, const mString &path, OUT bool *pExists);
mFUNCTION(mFile_DirectoryExists, IN const wchar_t *path, OUT bool *pExists);

mFUNCTION(mFile_ReadAllBytes, const mString &filename, IN OPTIONAL mAllocator *pAllocator, OUT mArray<uint8_t> *pBytes);
mFUNCTION(mFile_ReadAllBytes, IN const wchar_t *filename, IN OPTIONAL mAllocator *pAllocator, OUT mArray<uint8_t> *pBytes);
mFUNCTION(mFile_ReadAllText, const mString &filename, IN OPTIONAL mAllocator *pAllocator, OUT mString *pText, const mFile_Encoding encoding = mFile_Encoding::mF_E_UTF8);

mFUNCTION(mFile_WriteAllBytes, IN const wchar_t *filename, mArray<uint8_t> &bytes);
mFUNCTION(mFile_WriteAllBytes, const mString &filename, mArray<uint8_t> &bytes);
mFUNCTION(mFile_WriteAllText, const mString &filename, const mString &text, const mFile_Encoding encoding = mFile_Encoding::mF_E_UTF8);

template <typename T>
mFUNCTION(mFile_ReadAllItems, IN const wchar_t *filename, IN OPTIONAL mAllocator *pAllocator, OUT mArray<T> *pData);

template <typename T>
mFUNCTION(mFile_ReadAllItems, const mString &filename, IN OPTIONAL mAllocator *pAllocator, OUT mArray<T> *pData);

template <typename T>
mFUNCTION(mFile_ReadRaw, IN const wchar_t *filename, OUT T **ppData, IN mAllocator *pAllocator, OUT size_t *pCount);

template <typename T>
mFUNCTION(mFile_ReadRaw, const mString &filename, OUT T **ppData, IN mAllocator *pAllocator, OUT size_t *pCount);

template <typename T>
mFUNCTION(mFile_WriteRaw, IN const wchar_t *filename, IN T *pData, const size_t count);

template <typename T>
mFUNCTION(mFile_WriteRaw, const mString &filename, IN T *pData, const size_t count);

mFUNCTION(mFile_WriteRawBytes, const wchar_t *filename, IN const uint8_t *pData, const size_t bytes);

mFUNCTION(mFile_FailOnInvalidDirectoryPath, const mString &folderPath, OUT OPTIONAL mString *pAbsolutePath);
mFUNCTION(mFile_CreateDirectory, const mString &folderPath);
mFUNCTION(mFile_DeleteFolder, const mString &folderPath);
mFUNCTION(mFile_Copy, const mString &destinationFileName, const mString &sourceFileName, const bool overrideFileIfExistent = false);
mFUNCTION(mFile_Copy, const mString &destinationFileName, const mString &sourceFileName, const std::function<mResult(const double_t progress)> &progressCallback, const bool overrideFileIfExistent);
mFUNCTION(mFile_Move, const mString &destinationFileName, const mString &sourceFileName, const bool overrideFileIfExistent = false);
mFUNCTION(mFile_Move, const mString &destinationFileName, const mString &sourceFileName, const std::function<mResult(const double_t progress)> &progressCallback, const bool overrideFileIfExistent = false);

#define mFile_DeleteFile mFile_Delete
mFUNCTION(mFile_Delete, const mString &filename);

mFUNCTION(mFile_GetTempDirectory, OUT mString *pString);
mFUNCTION(mFile_GetAppDataDirectory, OUT mString *pString);
mFUNCTION(mFile_GetDesktopDirectory, OUT mString *pString);
mFUNCTION(mFile_GetDocumentsDirectory, OUT mString *pString);
mFUNCTION(mFile_GetFontsDirectory, OUT mString *pString);
mFUNCTION(mFile_GetCurrentUserDirectory, OUT mString *pString);
mFUNCTION(mFile_GetProgramFilesDirectory, OUT mString *pString);
mFUNCTION(mFile_GetStartupDirectory, OUT mString *pString);
mFUNCTION(mFile_GetStartMenu, OUT mString *pString);
mFUNCTION(mFile_GetStartMenuPrograms, OUT mString *pString);
mFUNCTION(mFile_GetSystemFolder, OUT mString *pString);
mFUNCTION(mFile_GetSystemFolderX86, OUT mString *pString);
mFUNCTION(mFile_GetOSFolder, OUT mString *pString);
mFUNCTION(mFile_GetStartupDirectory_AllUsers, OUT mString *pString);
mFUNCTION(mFile_GetStartMenu_AllUsers, OUT mString *pString);
mFUNCTION(mFile_GetStartMenuPrograms_AllUsers, OUT mString *pString);

mFUNCTION(mFile_GetWorkingDirectory, OUT mString *pWorkingDirectory);
mFUNCTION(mFile_SetWorkingDirectory, const mString &workingDirectory);
mFUNCTION(mFile_GetCurrentApplicationFilePath, OUT mString *pAppDirectory);

mFUNCTION(mFile_ExtractDirectoryFromPath, OUT mString *pDirectory, const mString &filePath);
mFUNCTION(mFile_ExtractFileExtensionFromPath, OUT mString *pExtension, const mString &filePath);
mFUNCTION(mFile_ExtractFileNameFromPath, OUT mString *pFileName, const mString &filePath, const bool withExtension);

struct mFileInfo
{
  mString name;
  size_t size;
  size_t creationTimeStamp;
  size_t lastAccessTimeStamp;
  size_t lastWriteTimeStamp;
  bool isDirectory, isHidden, isOffline, isSystemResource, isReadonly;
};

mFUNCTION(mFile_GetDirectoryContents, const mString &directoryPath, OUT mPtr<mQueue<mFileInfo>> *pFiles, IN mAllocator *pAllocator);
mFUNCTION(mFile_GetDirectoryContents, const mString &directoryPath, const mString &searchTerm, OUT mPtr<mQueue<mFileInfo>> *pFiles, IN mAllocator *pAllocator);
mFUNCTION(mFile_GetDirectoryContents, const mString &directoryPath, const char *searchTerm, OUT mPtr<mQueue<mFileInfo>> *pFiles, IN mAllocator *pAllocator);
mFUNCTION(mFile_GetDirectoryContents, const mString &directoryPath, const wchar_t *searchTerm, OUT mPtr<mQueue<mFileInfo>> *pFiles, IN mAllocator *pAllocator);
mFUNCTION(mFile_GetDirectoryContents, const mString &directoryPath, const bool recursive, OUT mPtr<mQueue<mFileInfo>> *pFiles, IN mAllocator *pAllocator);

#define mFile_GetFileInfo mFile_GetInfo
mFUNCTION(mFile_GetInfo, const mString &filename, OUT mFileInfo *pFileInfo);

#define mFile_GetFileSize mFile_GetSize
mFUNCTION(mFile_GetSize, const mString &filename, OUT size_t *pSizeBytes);

enum mDriveType
{
  mDT_Unknown,
  mDT_Removable,
  mDT_NonRemovable,
  mDT_Remote,
  mDT_CDRom,
  mDT_RamDisk,
};

struct mDriveInfo
{
  mString drivePath, driveName;
  mDriveType driveType;
};

// Retrieves a list of available drives with additional drive information on the current machine.
// If the queue is null, creates a queue, otherwise clears the queue.
mFUNCTION(mFile_GetDrives, OUT mPtr<mQueue<mDriveInfo>> *pDrives, IN mAllocator *pAllocator);

// Retrieves a list of available drives with additional drive information on the current machine.
// If the queue is null, creates a queue, otherwise clears the queue.
mFUNCTION(mFile_GetDrives, OUT mPtr<mQueue<mString>> *pDrives, IN mAllocator *pAllocator);

mFUNCTION(mFile_GetFreeStorageSpace, const mString &path, OUT size_t *pSize);

mFUNCTION(mFile_GetAbsoluteDirectoryPath, OUT mString *pAbsolutePath, const mString &directoryPath);
mFUNCTION(mFile_GetAbsoluteFilePath, OUT mString *pAbsolutePath, const mString &filePath);

mFUNCTION(mFile_LaunchFile, const mString &filename);
mFUNCTION(mFile_LaunchApplication, const mString &applicationFilename, const mString &arguments, const mString &workingDirectory, const bool elevatePrivileges = false);

mFUNCTION(mFile_CreateShortcut, const mString &path, const mString &targetDestination, const mString &arguments, const mString &workingDirectory, const mString &description, const mString &iconLocation);

mFUNCTION(mFile_GrantAccessToAllUsers, const mString &path);

mFUNCTION(mFile_GetDriveFromFilePath, OUT mString *pDrivePath, const mString &filePath);
mFUNCTION(mFile_GetDriveInfo, const mString &drivePath, OUT mDriveInfo *pDriveInfo, IN mAllocator *pAllocator);
mFUNCTION(mFile_GetFolderSize, const mString &directoryPath, OUT size_t *pFolderSize);

struct mFileWriter
{
#ifdef mPLATFORM_WINDOWS
  HANDLE file;
#else
  FILE *pFile;
#endif
  PUBLIC_FIELD size_t bytesWritten;
};

mFUNCTION(mFileWriter_Create, OUT mPtr<mFileWriter> *pWriter, IN mAllocator *pAllocator, const mString &filename);
mFUNCTION(mFileWriter_Create, OUT mUniqueContainer<mFileWriter> *pWriter, const mString &filename);

mFUNCTION(mFileWriter_Destroy, IN_OUT mPtr<mFileWriter> *pWriter);

mFUNCTION(mFileWriter_WriteRaw, mPtr<mFileWriter> &writer, const uint8_t *pData, const size_t size);

template <typename T>
mFUNCTION(mFileWriter_Write, mPtr<mFileWriter> &writer, const T *pData, const size_t count)
{
  return mFileWriter_WriteRaw(writer, reinterpret_cast<const uint8_t *>(pData), sizeof(T) * count);
}

//////////////////////////////////////////////////////////////////////////

template<typename T>
inline mFUNCTION(mFile_ReadAllItems, IN const wchar_t *filename, IN OPTIONAL mAllocator *pAllocator, OUT mArray<T> *pData)
{
  mFUNCTION_SETUP();

  mERROR_IF(pData == nullptr, mR_ArgumentNull);

  if(pData->pData)
    mERROR_CHECK(mArray_Destroy(pData));

  pData->pAllocator = pAllocator;

  mERROR_CHECK(mFile_ReadRaw(filename, &pData->pData, pAllocator, &pData->count));

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mFile_ReadAllItems, const mString &filename, IN OPTIONAL mAllocator *pAllocator, OUT mArray<T> *pData)
{
  mFUNCTION_SETUP();

  wchar_t wfilename[1024];
  mERROR_CHECK(mString_ToWideString(filename, wfilename, mARRAYSIZE(wfilename)));

  mERROR_CHECK(mFile_ReadAllItems(wfilename, pAllocator, pData));

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mFile_ReadRaw, const wchar_t *filename, OUT T **ppData, IN mAllocator *pAllocator, OUT size_t *pCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppData == nullptr || pCount == nullptr, mR_ArgumentNull);

  FILE *pFile = _wfopen(filename, L"rb");
  mDEFER(if (pFile) { fclose(pFile); });
  mERROR_IF(pFile == nullptr, mR_ResourceNotFound);

  mERROR_IF(0 != fseek(pFile, 0, SEEK_END), mR_IOFailure);
  
  const size_t length = ftell(pFile);
  const size_t count = length / sizeof(T);

  mERROR_IF(0 != fseek(pFile, 0, SEEK_SET), mR_IOFailure);

  mERROR_CHECK(mAllocator_Allocate(pAllocator, (uint8_t **)ppData, length + 1));
  
  mERROR_CHECK(mZeroMemory(&((*ppData)[length]), 1)); // To zero terminate strings. This is out of bounds for all other data types anyways.

  const size_t readLength = fread(*ppData, 1, length, pFile);

  *pCount = readLength / sizeof(T);

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mFile_ReadRaw, const mString &filename, OUT T **ppData, IN mAllocator *pAllocator, OUT size_t *pCount)
{
  mFUNCTION_SETUP();

  wchar_t wfilename[MAX_PATH];
  mERROR_CHECK(mString_ToWideString(filename, wfilename, mARRAYSIZE(wfilename)));

  mERROR_CHECK(mFile_ReadRaw(wfilename, ppData, pAllocator, pCount));

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mFile_WriteRaw, IN const wchar_t *filename, IN T *pData, const size_t count)
{
  mFUNCTION_SETUP();

  mERROR_IF(pData == nullptr || filename == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mFile_WriteRawBytes(filename, reinterpret_cast<const uint8_t *>(pData), count * sizeof(T)));

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mFile_WriteRaw, const mString &filename, IN T *pData, const size_t count)
{
  mFUNCTION_SETUP();

  wchar_t wfilename[MAX_PATH];
  mERROR_CHECK(mString_ToWideString(filename, wfilename, mARRAYSIZE(wfilename)));

  mERROR_CHECK(mFile_WriteRaw(wfilename, pData, count));

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mRegistry_ReadKey, const mString &keyUrl, OUT mString *pValue);
mFUNCTION(mRegistry_ReadKey, const mString &keyUrl, OUT uint32_t *pValue);
mFUNCTION(mRegistry_ReadKey, const mString &keyUrl, OUT uint64_t *pValue);
mFUNCTION(mRegistry_WriteKey, const mString &keyUrl, const mString &value, OUT OPTIONAL bool *pNewlyCreated = nullptr);
mFUNCTION(mRegistry_WriteKey, const mString &keyUrl, const uint32_t value, OUT OPTIONAL bool *pNewlyCreated = nullptr);
mFUNCTION(mRegistry_WriteKey, const mString &keyUrl, const uint64_t value, OUT OPTIONAL bool *pNewlyCreated = nullptr);
mFUNCTION(mRegistry_DeleteKey, const mString &keyUrl);

#endif // mFile_h__
