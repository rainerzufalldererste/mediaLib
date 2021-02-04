#include "mDynamicLibrary.h"

#ifdef mPLATFORM_WINDOWS
#pragma warning (push)
#pragma warning (disable: 4091)

extern "C"
{
  #include "MemoryModulePP/include/MemoryModulePP.h"
  #include "MemoryModulePP/src/MemoryModulePP.c"
}
#pragma warning (pop)
#endif

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "Lz+UbdaLT6FQVl8t1AIQYpY8QUFIwuLycmTQzNIplNh5yzymY09ZjkpIH6EUFgJXZPKwsG5qpYwtnV4q"
#endif

#ifdef mPLATFORM_WINDOWS
struct mDynamicLibrary : MEMORYMODULE
{
  // This is a dummy struct. Don't put anything here.
};

struct mDynamicLibraryResource
{
  // This is a dummy struct. Don't put anything here.
};
#endif

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mDynamicLibrary_LoadFromMemory, const uint8_t *pData, const size_t size, OUT mDynamicLibrary **ppHandle)
{
  mFUNCTION_SETUP();

  mERROR_IF(pData == nullptr || ppHandle == nullptr, mR_ArgumentNull);
  mERROR_IF(size == 0, mR_ArgumentOutOfBounds);

#ifdef mPLATFORM_WINDOWS
  mDynamicLibrary *pModule = (mDynamicLibrary *)MemoryLoadLibrary(pData, size);
  mERROR_IF(pModule == nullptr, mR_InternalError);

  *ppHandle = pModule;

  mRETURN_SUCCESS();
#else
  mRETURN_RESULT(mR_NotImplemented);
#endif
}

mFUNCTION(mDynamicLibrary_Free, mDynamicLibrary **ppHandle)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppHandle == nullptr, mR_ArgumentNull);

  if (*ppHandle != nullptr)
    MemoryFreeLibrary((PMEMORYMODULE)*ppHandle);

  mRETURN_SUCCESS();
}

mFUNCTION(mDynamicLibrary_LoadFunction, mDynamicLibrary *pHandle, IN const char *name, OUT void **ppFunction)
{
  mFUNCTION_SETUP();

  mERROR_IF(name == nullptr || ppFunction == nullptr, mR_ArgumentNull);

  void *pFunc = MemoryGetProcAddress((PMEMORYMODULE)pHandle, name);
  mERROR_IF(pFunc == nullptr, mR_ResourceNotFound);

  *ppFunction = pFunc;

  mRETURN_SUCCESS();
}

mFUNCTION(mDynamicLibrary_CallEntryPoint, mDynamicLibrary *pHandle, OUT OPTIONAL int32_t *pExitCode)
{
  mFUNCTION_SETUP();

  mERROR_IF(pHandle == nullptr, mR_ArgumentNull);
  mERROR_IF(pHandle->exeEntry == nullptr, mR_ResourceIncompatible);

  const int32_t exitCode = MemoryCallEntryPoint((PMEMORYMODULE)pHandle);

  if (pExitCode != nullptr)
    *pExitCode = exitCode;

  mRETURN_SUCCESS();
}

mFUNCTION(mDynamicLibrary_FindResource, mDynamicLibrary *pHandle, const mString &name, const mString &type, OUT mDynamicLibraryResource **ppResource)
{
  mFUNCTION_SETUP();

  mERROR_IF(pHandle == nullptr || name.c_str() == nullptr || type.c_str() == nullptr || ppResource == nullptr, mR_ArgumentNull);

  size_t nameRequiredCount, typeRequiredCount;
  mERROR_CHECK(mString_GetRequiredWideStringCount(name, &nameRequiredCount));
  mERROR_CHECK(mString_GetRequiredWideStringCount(type, &typeRequiredCount));

  mPtr<wchar_t> nameW, typeW;
  mERROR_CHECK(mSharedPointer_Allocate(&nameW, &mDefaultTempAllocator, nameRequiredCount));
  mERROR_CHECK(mSharedPointer_Allocate(&typeW, &mDefaultTempAllocator, typeRequiredCount));

  mERROR_CHECK(mString_ToWideString(name, nameW.GetPointer(), nameRequiredCount));
  mERROR_CHECK(mString_ToWideString(type, typeW.GetPointer(), typeRequiredCount));

  HMEMORYRSRC resource = MemoryFindResource((PMEMORYMODULE)pHandle, nameW.GetPointer(), typeW.GetPointer());
  mERROR_IF(resource != nullptr, mR_ResourceNotFound);

  *ppResource = reinterpret_cast<mDynamicLibraryResource *>(resource);

  mRETURN_SUCCESS();
}

mFUNCTION(mDynamicLibrary_FindResource, mDynamicLibrary *pHandle, const mString &name, const mString &type, const int16_t languageCode, OUT mDynamicLibraryResource **ppResource)
{
  mFUNCTION_SETUP();

  mERROR_IF(pHandle == nullptr || name.c_str() == nullptr || type.c_str() == nullptr || ppResource == nullptr, mR_ArgumentNull);

  size_t nameRequiredCount, typeRequiredCount;
  mERROR_CHECK(mString_GetRequiredWideStringCount(name, &nameRequiredCount));
  mERROR_CHECK(mString_GetRequiredWideStringCount(type, &typeRequiredCount));

  mPtr<wchar_t> nameW, typeW;
  mERROR_CHECK(mSharedPointer_Allocate(&nameW, &mDefaultTempAllocator, nameRequiredCount));
  mERROR_CHECK(mSharedPointer_Allocate(&typeW, &mDefaultTempAllocator, typeRequiredCount));

  mERROR_CHECK(mString_ToWideString(name, nameW.GetPointer(), nameRequiredCount));
  mERROR_CHECK(mString_ToWideString(type, typeW.GetPointer(), typeRequiredCount));

  HMEMORYRSRC resource = MemoryFindResourceEx((PMEMORYMODULE)pHandle, nameW.GetPointer(), typeW.GetPointer(), languageCode);
  mERROR_IF(resource != nullptr, mR_ResourceNotFound);

  *ppResource = reinterpret_cast<mDynamicLibraryResource *>(resource);

  mRETURN_SUCCESS();
}

mFUNCTION(mDynamicLibrary_SizeofResource, mDynamicLibrary *pHandle, mDynamicLibraryResource *pResource, OUT size_t *pSize)
{
  mFUNCTION_SETUP();

  mERROR_IF(pHandle == nullptr || pResource == nullptr || pSize == nullptr, mR_ArgumentNull);

  const DWORD size = MemorySizeofResource((PMEMORYMODULE)pHandle, reinterpret_cast<HMEMORYRSRC>(pResource));
  mERROR_IF(size == 0, mR_ResourceNotFound);

  *pSize = (size_t)size;

  mRETURN_SUCCESS();
}

mFUNCTION(mDynamicLibrary_LoadResource, mDynamicLibrary *pHandle, mDynamicLibraryResource *pResource, OUT void **ppData)
{
  mFUNCTION_SETUP();

  mERROR_IF(pHandle == nullptr || pResource == nullptr || ppData == nullptr, mR_ArgumentNull);

  void *pData = MemoryLoadResource((PMEMORYMODULE)pHandle, reinterpret_cast<HMEMORYRSRC>(pResource));
  mERROR_IF(pData == nullptr, mR_ResourceNotFound);

  *ppData = pData;

  mRETURN_SUCCESS();
}

mFUNCTION(mDynamicLibrary_LoadString, mDynamicLibrary *pHandle, const size_t id, IN mAllocator *pAllocator, OUT mString *pString)
{
  mFUNCTION_SETUP();

  mERROR_IF(pHandle == nullptr || pString == nullptr, mR_ArgumentNull);

  const size_t length = MemoryStringRequiredCapacity((PMEMORYMODULE)pHandle, (UINT)id, DEFAULT_LANGUAGE);
  mERROR_IF(length == 0, mR_ResourceNotFound);

  mPtr<wchar_t> string;
  mERROR_CHECK(mSharedPointer_Allocate(&string, &mDefaultTempAllocator, (int32_t)length + 1));

  const size_t result = MemoryLoadString((PMEMORYMODULE)pHandle, (UINT)id, string.GetPointer(), (int32_t)length + 1);
  mERROR_IF(result != length, mR_ResourceStateInvalid);

  mERROR_CHECK(mString_Create(pString, string.GetPointer(), length + 1, pAllocator));

  mRETURN_SUCCESS();
}

mFUNCTION(mDynamicLibrary_LoadString, mDynamicLibrary *pHandle, const size_t id, const int16_t languageCode, IN mAllocator *pAllocator, OUT mString *pString)
{
  mFUNCTION_SETUP();

  mERROR_IF(pHandle == nullptr || pString == nullptr, mR_ArgumentNull);

  const size_t length = MemoryStringRequiredCapacity((PMEMORYMODULE)pHandle, (UINT)id, languageCode);
  mERROR_IF(length == 0, mR_ResourceNotFound);

  mPtr<wchar_t> string;
  mERROR_CHECK(mSharedPointer_Allocate(&string, &mDefaultTempAllocator, (int32_t)length + 1));

  const size_t result = MemoryLoadStringEx((PMEMORYMODULE)pHandle, (UINT)id, string.GetPointer(), (int32_t)length + 1, languageCode);
  mERROR_IF(result != length, mR_ResourceStateInvalid);

  mERROR_CHECK(mString_Create(pString, string.GetPointer(), length + 1, pAllocator));

  mRETURN_SUCCESS();
}
