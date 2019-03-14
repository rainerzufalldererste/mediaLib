#ifndef mFile_h__
#define mFile_h__

#include "mediaLib.h"
#include "mArray.h"
#include "mQueue.h"

enum mFile_Encoding
{
  mF_E_UTF8,
};

mFUNCTION(mFile_Exists, const mString &filename, OUT bool *pExists);
mFUNCTION(mFile_Exists, IN const wchar_t *filename, OUT bool *pExists);

mFUNCTION(mFile_ReadAllBytes, const mString &filename, IN OPTIONAL mAllocator *pAllocator, OUT mArray<uint8_t> *pBytes);
mFUNCTION(mFile_ReadAllBytes, IN const wchar_t *filename, IN OPTIONAL mAllocator *pAllocator, OUT mArray<uint8_t> *pBytes);
mFUNCTION(mFile_ReadAllText, const mString &filename, IN OPTIONAL mAllocator *pAllocator, OUT mString *pText, const mFile_Encoding encoding = mF_E_UTF8);

mFUNCTION(mFile_WriteAllBytes, IN const wchar_t *filename, mArray<uint8_t> &bytes);
mFUNCTION(mFile_WriteAllBytes, const mString &filename, mArray<uint8_t> &bytes);
mFUNCTION(mFile_WriteAllText, const mString &filename, const mString &text, const mFile_Encoding encoding = mF_E_UTF8);

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

mFUNCTION(mFile_CreateDirectory, const mString &folderPath);
mFUNCTION(mFile_DeleteFolder, const mString &folderPath);
mFUNCTION(mFile_Copy, const mString &destinationFileName, const mString &sourceFileName, const bool overrideFileIfExistent = false);
mFUNCTION(mFile_Move, const mString &destinationFileName, const mString &sourceFileName, const bool overrideFileIfExistent = false);
mFUNCTION(mFile_Delete, const mString &filename);

mFUNCTION(mFile_GetTempDirectory, OUT mString *pString);
mFUNCTION(mFile_GetAppDataDirectory, OUT mString *pString);
mFUNCTION(mFile_GetDesktopDirectory, OUT mString *pString);
mFUNCTION(mFile_GetDocumentsDirectory, OUT mString *pString);
mFUNCTION(mFile_GetFontsDirectory, OUT mString *pString);
mFUNCTION(mFile_GetCurrentUserDirectory, OUT mString *pString);
mFUNCTION(mFile_GetProgramFilesDirectory, OUT mString *pString);
mFUNCTION(mFile_GetStartupDirectory, OUT mString *pString);

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
mFUNCTION(mFile_GetDirectoryContents, const mString &directoryPath, const bool recursive, OUT mPtr<mQueue<mFileInfo>> *pFiles, IN mAllocator *pAllocator);
mFUNCTION(mFile_GetInfo, const mString &filename, OUT mFileInfo *pFileInfo);

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

mFUNCTION(mFile_GetDrives, OUT mPtr<mQueue<mDriveInfo>> *pDrives, IN mAllocator *pAllocator);

mFUNCTION(mFile_GetAbsoluteDirectoryPath, OUT mString *pAbsolutePath, const mString &directoryPath);
mFUNCTION(mFile_GetAbsoluteFilePath, OUT mString *pAbsolutePath, const mString &filePath);

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
  const size_t readLength = fread(*ppData, 1, length, pFile);

  *pCount = readLength / sizeof(T);

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mFile_ReadRaw, const mString &filename, OUT T **ppData, IN mAllocator *pAllocator, OUT size_t *pCount)
{
  mFUNCTION_SETUP();

  wchar_t wfilename[1024];
  mERROR_CHECK(mString_ToWideString(filename, wfilename, mARRAYSIZE(wfilename)));

  mERROR_CHECK(mFile_ReadRaw(wfilename, ppData, pAllocator, pCount));

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mFile_WriteRaw, IN const wchar_t *filename, IN T *pData, const size_t count)
{
  mFUNCTION_SETUP();

  mERROR_IF(pData == nullptr, mR_ArgumentNull);

  FILE *pFile = _wfopen(filename, L"wb");
  mDEFER(if (pFile) fclose(pFile););
  mERROR_IF(pFile == nullptr, mR_ResourceNotFound);

  const size_t writeCount = fwrite(pData, 1, sizeof(T) * count, pFile);

  mERROR_IF(writeCount != count * sizeof(T), mR_IOFailure);

  mRETURN_SUCCESS();
}

template<typename T>
inline mFUNCTION(mFile_WriteRaw, const mString &filename, IN T *pData, const size_t count)
{
  mFUNCTION_SETUP();

  wchar_t wfilename[1024];
  mERROR_CHECK(mString_ToWideString(filename, wfilename, mARRAYSIZE(wfilename)));

  mERROR_CHECK(mFile_WriteRaw(wfilename, pData, count));

  mRETURN_SUCCESS();
}

#endif // mFile_h__
