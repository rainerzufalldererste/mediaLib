#include "mMappedFile.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "cUbCTgWkq29BMQJEFgpqrS2+Zndu/6qd9t4aK51HJBMIfIM5y7vwCsQd6m6IKaiDxqxe7kH+9ueACOMn"
#endif

//////////////////////////////////////////////////////////////////////////

struct mMappedFile
{
  HANDLE file, mapping;
};

mFUNCTION(mMappedFile_Destroy_Internal, mMappedFile *pMapping);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mMappedFile_CreateFromFile, OUT mPtr<mMappedFile> *pMappedFile, IN OPTIONAL mAllocator *pAllocator, const mString &filename, const bool hasMappingName, OPTIONAL mString mappingName, const bool isGlobalNamespace, OUT void **ppMapping)
{
  mFUNCTION_SETUP();

  mERROR_IF(pMappedFile == nullptr || ppMapping == nullptr, mR_ArgumentNull);
  mERROR_IF(filename.hasFailed || filename.bytes <= 1, mR_InvalidParameter);
  mERROR_IF(hasMappingName && (mappingName.hasFailed || mappingName.bytes <= 1), mR_InvalidParameter);

  wchar_t wName[MAX_PATH + 1];
  mERROR_CHECK(mString_ToWideString(filename, wName, mARRAYSIZE(wName)));

  HANDLE file = CreateFileW(wName, GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  mERROR_IF(file == nullptr || file == INVALID_HANDLE_VALUE, mR_InternalError);
  mDEFER_ON_ERROR(CloseHandle, file);

  LARGE_INTEGER size;
  mERROR_IF(0 == GetFileSizeEx(file, &size), mR_InternalError);

  wchar_t wMappingName[MAX_PATH + 1];

  if (hasMappingName)
  {
    mString properName;

    if (!isGlobalNamespace)
      mERROR_CHECK(mString_Format(&properName, &mDefaultTempAllocator, "Local\\", mappingName));
    else
      mERROR_CHECK(mString_Format(&properName, &mDefaultTempAllocator, "Global\\", mappingName));

    mERROR_CHECK(mString_ToWideString(properName, wMappingName, mARRAYSIZE(wMappingName)));
  }

  HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READWRITE, size.HighPart, size.LowPart, hasMappingName ? wMappingName : nullptr);

  if (mapping == nullptr)
  {
    switch (GetLastError())
    {
    case ERROR_ACCESS_DENIED:
      mRETURN_RESULT(mR_InsufficientPrivileges);

    default:
      mRETURN_RESULT(mR_InternalError);
    }
  }

  mDEFER_ON_ERROR(CloseHandle, mapping);

  void *pMapping = MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, size.QuadPart);
  mERROR_IF(pMapping == nullptr, mR_InternalError);

  mDEFER_CALL_ON_ERROR(pMappedFile, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Allocate<mMappedFile>(pMappedFile, pAllocator, mMappedFile_Destroy_Internal, 1));

  (*pMappedFile)->file = file;
  (*pMappedFile)->mapping = mapping;

  *ppMapping = pMapping;

  mRETURN_SUCCESS();
}

mFUNCTION(mMappedFile_CreateWithoutFile, OUT mPtr<mMappedFile> *pMappedFile, IN OPTIONAL mAllocator *pAllocator, const mString &mappingName, const bool isGlobalNamespace, const size_t size, OUT void **ppMapping)
{
  mFUNCTION_SETUP();

  mERROR_IF(pMappedFile == nullptr || ppMapping == nullptr, mR_ArgumentNull);
  mERROR_IF(mappingName.hasFailed || mappingName.bytes <= 1, mR_InvalidParameter);
  mERROR_IF(size > INT64_MAX, mR_ArgumentOutOfBounds);

  wchar_t wMappingName[MAX_PATH + 1];
  mString properName;

  if (!isGlobalNamespace)
    mERROR_CHECK(mString_Format(&properName, &mDefaultTempAllocator, "Local\\", mappingName));
  else
    mERROR_CHECK(mString_Format(&properName, &mDefaultTempAllocator, "Global\\", mappingName));

  mERROR_CHECK(mString_ToWideString(properName, wMappingName, mARRAYSIZE(wMappingName)));

  LARGE_INTEGER sizeLI;
  sizeLI.QuadPart = size;

  HANDLE mapping = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, sizeLI.HighPart, sizeLI.LowPart, wMappingName);
  
  if (mapping == nullptr)
  {
    switch (GetLastError())
    {
    case ERROR_ACCESS_DENIED:
      mRETURN_RESULT(mR_InsufficientPrivileges);

    default:
      mRETURN_RESULT(mR_InternalError);
    }
  }
  
  mDEFER_ON_ERROR(CloseHandle, mapping);

  void *pMapping = MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, size);
  mERROR_IF(pMapping == nullptr, mR_InternalError);

  mDEFER_CALL_ON_ERROR(pMappedFile, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Allocate<mMappedFile>(pMappedFile, pAllocator, mMappedFile_Destroy_Internal, 1));

  (*pMappedFile)->file = nullptr;
  (*pMappedFile)->mapping = mapping;

  *ppMapping = pMapping;

  mRETURN_SUCCESS();
}

mFUNCTION(mMappedFile_Destroy, IN_OUT mPtr<mMappedFile> *pMappedFile)
{
  return mSharedPointer_Destroy(pMappedFile);
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mMappedFile_Destroy_Internal, mMappedFile *pMapping)
{
  mFUNCTION_SETUP();

  mERROR_IF(pMapping == nullptr, mR_ArgumentNull);

  if (pMapping->mapping != nullptr)
    CloseHandle(pMapping->mapping);

  if (pMapping->file != nullptr)
    CloseHandle(pMapping->file);

  mRETURN_SUCCESS();
}
