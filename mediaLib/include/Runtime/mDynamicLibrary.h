#ifndef mDynamicLibrary_h__
#define mDynamicLibrary_h__

#include "mediaLib.h"

struct mDynamicLibrary;

mFUNCTION(mDynamicLibrary_LoadFromMemory, const uint8_t *pData, const size_t size, OUT mDynamicLibrary **ppHandle);
mFUNCTION(mDynamicLibrary_Free, mDynamicLibrary **ppHandle);

mFUNCTION(mDynamicLibrary_LoadFunction, mDynamicLibrary *pHandle, IN const char *name, OUT void **ppFunction);
mFUNCTION(mDynamicLibrary_CallEntryPoint, mDynamicLibrary *pHandle, OUT OPTIONAL int32_t *pExitCode);

struct mDynamicLibraryResource;

mFUNCTION(mDynamicLibrary_FindResource, mDynamicLibrary *pHandle, const mString &name, const mString &type, OUT mDynamicLibraryResource **ppResource);
mFUNCTION(mDynamicLibrary_FindResource, mDynamicLibrary *pHandle, const mString &name, const mString &type, const int16_t languageCode, OUT mDynamicLibraryResource **ppResource);

mFUNCTION(mDynamicLibrary_SizeofResource, mDynamicLibrary *pHandle, mDynamicLibraryResource *pResource, OUT size_t *pSize);
mFUNCTION(mDynamicLibrary_LoadResource, mDynamicLibrary *pHandle, mDynamicLibraryResource *pResource, OUT void **ppData);

mFUNCTION(mDynamicLibrary_LoadString, mDynamicLibrary *pHandle, const size_t id, IN mAllocator *pAllocator, OUT mString *pString);
mFUNCTION(mDynamicLibrary_LoadString, mDynamicLibrary *pHandle, const size_t id, IN mAllocator *pAllocator, const int16_t languageCode, OUT mString *pString);

#endif // mDynamicLibrary_h__
