#include "mediaLib.h"

#ifndef _DEBUG
extern "C"
{
#pragma warning(push, 0)
  #include "rpmalloc/src/rpmalloc.h"
  #include "rpmalloc/src/rpmalloc.c"
#pragma warning(pop)
}
#endif

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "Rx6R99ErVSc74oQfABhd3lN1Z/7LbxaWj5aofNcPw+3mWMReFYhCMscW350rxwq48gtWVS1+nWs+s8Dw"
#endif

extern bool mMemory_RpMallocEnabled = false;
extern bool mMemory_HasActiveAllocations = false;

extern void mMemory_OnProcessStart()
{
#ifndef _DEBUG
  if (mMemory_HasActiveAllocations)
    return;

  HMODULE moduleHandle = nullptr;

  if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPWSTR>(const_cast<void *>(reinterpret_cast<const void *>(mMemory_OnProcessStart))), &moduleHandle) != 0)
    moduleHandle = nullptr;

  mMemory_RpMallocEnabled = (GetModuleHandleW(nullptr) == moduleHandle);

  if (mMemory_RpMallocEnabled && 0 != rpmalloc_initialize())
    mMemory_RpMallocEnabled = false;
#endif
}

extern void mMemory_OnThreadStart()
{
#ifndef _DEBUG
  if (mMemory_RpMallocEnabled)
    rpmalloc_thread_initialize();
#endif
}

extern void mMemory_OnThreadExit()
{
#ifndef _DEBUG
  if (mMemory_RpMallocEnabled)
    rpmalloc_thread_finalize(0);
#endif
}

extern void mMemory_OnProcessExit()
{
#ifndef _DEBUG
  if (mMemory_RpMallocEnabled)
    rpmalloc_finalize();
#endif
}

_Check_return_ _Ret_maybenull_ _Post_writable_byte_size_(size) _CRTALLOCATOR _CRTRESTRICT void *_m_internal_alloc(_In_ const size_t size)
{
#ifndef _DEBUG
  mMemory_HasActiveAllocations = true;

  if (mMemory_RpMallocEnabled)
    return rpmalloc(size);
  else
    return malloc(size);
#else
  return malloc(size);
#endif
}

_Check_return_ _Ret_maybenull_ _Post_writable_byte_size_(size) _CRTALLOCATOR _CRTRESTRICT void *_m_internal_alloc_zero(_In_ const size_t size)
{
#ifndef _DEBUG
  mMemory_HasActiveAllocations = true;

  if (mMemory_RpMallocEnabled)
    return rpcalloc(size, 1);
  else
    return calloc(size, 1);
#else
  return calloc(size, 1);
#endif
}

_Success_(return != 0) _Check_return_ _Ret_maybenull_ _Post_writable_byte_size_(size) _CRTALLOCATOR _CRTRESTRICT void *_m_internal_realloc(_Pre_maybenull_ _Post_invalid_ void *pBlock, _In_ const size_t size)
{
#ifndef _DEBUG
  mMemory_HasActiveAllocations = true;

  if (mMemory_RpMallocEnabled)
    return rprealloc(pBlock, size);
  else
    return realloc(pBlock, size);
#else
  return realloc(pBlock, size);
#endif
}

void __cdecl _m_internal_free(_Pre_maybenull_ _Post_invalid_ void *pBlock)
{
#ifndef _DEBUG
  if (mMemory_RpMallocEnabled)
    rpfree(pBlock);
  else
    free(pBlock);
#else
  free(pBlock);
#endif
}

_Check_return_ _Ret_maybenull_ _Post_writable_byte_size_(size) _CRTALLOCATOR _CRTRESTRICT void *_m_internal_alloc_aligned(_In_ const size_t size, _In_ const size_t alignment)
{
#ifndef _DEBUG
  mMemory_HasActiveAllocations = true;

  if (mMemory_RpMallocEnabled)
    return rpaligned_alloc(alignment, size);
  else
    return _aligned_malloc(size, alignment);
#else
  return _aligned_malloc(size, alignment);
#endif
}

_Check_return_ _Ret_maybenull_ _Post_writable_byte_size_(size) _CRTALLOCATOR _CRTRESTRICT void *_m_internal_alloc_aligned_zero(_In_ const size_t size, _In_ const size_t alignment)
{
#ifndef _DEBUG
  mMemory_HasActiveAllocations = true;

  if (mMemory_RpMallocEnabled)
  {
    return rpaligned_calloc(alignment, size, 1);
  }
  else
  {
    void *pData = _aligned_malloc(size, alignment);

    if (pData != nullptr)
      memset(pData, 0, size);

    return pData;
  }
#else
  void *pData = _aligned_malloc(size, alignment);

  if (pData != nullptr)
    memset(pData, 0, size);

  return pData;
#endif
}

_Success_(return != 0) _Check_return_ _Ret_maybenull_ _Post_writable_byte_size_(size) _CRTALLOCATOR _CRTRESTRICT void *_m_internal_realloc(_Pre_maybenull_ _Post_invalid_ void *pBlock, _In_ const size_t size, _In_ const size_t oldSize, _In_ const size_t alignment)
{
#ifndef _DEBUG
  mMemory_HasActiveAllocations = true;

  if (mMemory_RpMallocEnabled)
    return rpaligned_realloc(pBlock, alignment, size, oldSize, 0);
  else
    return _aligned_realloc(pBlock, size, alignment);
#else
  mUnused(oldSize);

  return _aligned_realloc(pBlock, size, alignment);
#endif
}

void __cdecl _m_internal_free_aligned(_Pre_maybenull_ _Post_invalid_ void *pBlock)
{
#ifndef _DEBUG
  if (mMemory_RpMallocEnabled)
    rpfree(pBlock);
  else
    _aligned_free(pBlock);
#else
  _aligned_free(pBlock);
#endif
}

mFUNCTION(mStringLength, const char *text, const size_t maxCount, OUT size_t *pCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(text == nullptr || pCount == nullptr, mR_ArgumentNull);

  *pCount = strnlen_s(text, maxCount);

  mRETURN_SUCCESS();
}

mFUNCTION(mSprintf, OUT char *buffer, const size_t bufferCount, const char *formatString, ...)
{
  mFUNCTION_SETUP();

  mERROR_IF(buffer == nullptr || formatString == nullptr, mR_ArgumentNull);

  va_list args;
  va_start(args, formatString);
  const int count = vsprintf_s(buffer, bufferCount, formatString, args);
  va_end(args);

  mERROR_IF(count < 0, mR_ArgumentOutOfBounds);

  mRETURN_SUCCESS();
}

#pragma warning(push)
#pragma warning(disable: 5082)
mFUNCTION(mSprintfWithCount, OUT char *buffer, const size_t bufferCount, const char *formatString, OUT size_t *pCount, ...)
{
  mFUNCTION_SETUP();

  mERROR_IF(buffer == nullptr || formatString == nullptr || pCount == nullptr, mR_ArgumentNull);

  *pCount = 0;

  va_list args;
  va_start(args, formatString);
  const int count = vsprintf_s(buffer, bufferCount, formatString, args);
  va_end(args);

  mERROR_IF(count < 0, mR_ArgumentOutOfBounds);

  *pCount = count;

  mRETURN_SUCCESS();
}
#pragma warning(pop)

mFUNCTION(mStringCopy, OUT char *buffer, const size_t bufferCount, const char *source, const size_t sourceCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(buffer == nullptr || source == nullptr, mR_ArgumentNull);

  const errno_t error = strncpy_s(buffer, bufferCount, source, sourceCount);

  switch (error)
  {
  case 0:
    mRETURN_SUCCESS();

  case STRUNCATE:
  case ERANGE:
    mRETURN_RESULT(mR_ArgumentOutOfBounds);

  case EINVAL:
    mRETURN_RESULT(mR_InvalidParameter);

  default:
    mRETURN_RESULT(mR_InternalError);
  }
}

mFUNCTION(mStringConcat, OUT char *buffer, const size_t bufferCount, const char *source, const size_t sourceCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(buffer == nullptr || source == nullptr, mR_ArgumentNull);

  const errno_t error = strncat_s(buffer, bufferCount, source, sourceCount);

  switch (error)
  {
  case 0:
    mRETURN_SUCCESS();

  case STRUNCATE:
  case ERANGE:
    mRETURN_RESULT(mR_ArgumentOutOfBounds);

  case EINVAL:
    mRETURN_RESULT(mR_InvalidParameter);

  default:
    mRETURN_RESULT(mR_InternalError);
  }
}

mFUNCTION(mStringLength, const wchar_t *text, const size_t maxCount, OUT size_t *pCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(text == nullptr || pCount == nullptr, mR_ArgumentNull);

  *pCount = wcsnlen_s(text, maxCount);

  mRETURN_SUCCESS();
}

mFUNCTION(mSprintf, OUT wchar_t *buffer, const size_t bufferCount, const wchar_t *formatString, ...)
{
  mFUNCTION_SETUP();

  mERROR_IF(buffer == nullptr || formatString == nullptr, mR_ArgumentNull);

  va_list args;
  va_start(args, formatString);
  const int result = vswprintf_s(buffer, bufferCount, formatString, args);
  va_end(args);

  mERROR_IF(result < 0, mR_ArgumentOutOfBounds);

  mRETURN_SUCCESS();
}

#pragma warning(push)
#pragma warning(disable: 5082)
mFUNCTION(mSprintfWithCount, OUT wchar_t *buffer, const size_t bufferCount, const wchar_t *formatString, OUT size_t *pCount, ...)
{
  mFUNCTION_SETUP();

  mERROR_IF(buffer == nullptr || formatString == nullptr || pCount == nullptr, mR_ArgumentNull);

  va_list args;
  va_start(args, formatString);
  const int result = vswprintf_s(buffer, bufferCount, formatString, args);
  va_end(args);

  mERROR_IF(result < 0, mR_ArgumentOutOfBounds);

  *pCount = (size_t)result;

  mRETURN_SUCCESS();
}
#pragma warning(pop)

mFUNCTION(mStringCopy, OUT wchar_t *buffer, const size_t bufferCount, const wchar_t *source, const size_t sourceCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(buffer == nullptr || source == nullptr, mR_ArgumentNull);

  const errno_t error = wcsncpy_s(buffer, bufferCount, source, sourceCount);

  switch (error)
  {
  case 0:
    mRETURN_SUCCESS();

  case STRUNCATE:
  case ERANGE:
    mRETURN_RESULT(mR_ArgumentOutOfBounds);

  case EINVAL:
    mRETURN_RESULT(mR_InvalidParameter);

  default:
    mRETURN_RESULT(mR_InternalError);
  }
}

mFUNCTION(mStringConcat, OUT wchar_t *buffer, const size_t bufferCount, const wchar_t *source, const size_t sourceCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(buffer == nullptr || source == nullptr, mR_ArgumentNull);

  const errno_t error = wcsncat_s(buffer, bufferCount, source, sourceCount);

  switch (error)
  {
  case 0:
    mRETURN_SUCCESS();

  case STRUNCATE:
  case ERANGE:
    mRETURN_RESULT(mR_ArgumentOutOfBounds);

  case EINVAL:
    mRETURN_RESULT(mR_InvalidParameter);

  default:
    mRETURN_RESULT(mR_InternalError);
  }
}

mFUNCTION(mStringChar, const char *text, const size_t maxCount, const char character, OUT size_t *pCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(text == nullptr || pCount == nullptr, mR_ArgumentNull);

  *pCount = (size_t)((char *)memchr((void *)text, (int)character, strnlen_s(text, maxCount)) - (char *)text);

  mRETURN_SUCCESS();
}

mFUNCTION(mMemoryAlignHiAny, const size_t size, const size_t align, OUT size_t *pResult)
{
  mFUNCTION_SETUP();

  mERROR_IF(pResult == nullptr, mR_ArgumentNull);

  *pResult = (size + align - 1) / align * align;

  mRETURN_SUCCESS();
}

mFUNCTION(mMemoryAlignLoAny, const size_t size, const size_t align, OUT size_t *pResult)
{
  mFUNCTION_SETUP();

  mERROR_IF(pResult == nullptr, mR_ArgumentNull);

  *pResult = size / align * align;

  mRETURN_SUCCESS();
}

mFUNCTION(mMemoryAlignHi, const size_t size, const size_t align, OUT size_t *pResult)
{
  mFUNCTION_SETUP();

  mERROR_IF(pResult == nullptr, mR_ArgumentNull);

  *pResult = (size + align - 1) & ~(align - 1);

  mRETURN_SUCCESS();
}

mFUNCTION(mMemoryAlignHi, IN const void *pData, const size_t align, OUT void **ppResult)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppResult == nullptr, mR_ArgumentNull);

  *ppResult = (void *)((((size_t)pData) + align - 1) & ~(align - 1));

  mRETURN_SUCCESS();
}

mFUNCTION(mMemoryAlignLo, const size_t size, const size_t align, OUT size_t *pResult)
{
  mFUNCTION_SETUP();

  mERROR_IF(pResult == nullptr, mR_ArgumentNull);

  *pResult = size & ~(align - 1);

  mRETURN_SUCCESS();
}

mFUNCTION(mMemoryAlignLo, IN const void *pData, const size_t align, OUT void **ppResult)
{
  mFUNCTION_SETUP();

  mERROR_IF(ppResult == nullptr, mR_ArgumentNull);

  *ppResult = (void *)(((size_t)pData) & ~(align - 1));

  mRETURN_SUCCESS();
}

mFUNCTION(mMemoryIsAligned, const size_t size, const size_t align, OUT bool *pResult)
{
  mFUNCTION_SETUP();

  mERROR_IF(pResult == nullptr, mR_ArgumentNull);

  size_t allignedSize;
  mERROR_CHECK(mMemoryAlignLo(size, align, &allignedSize));

  *pResult = (size == allignedSize);

  mRETURN_SUCCESS();
}

mFUNCTION(mMemoryIsAligned, IN const void *pData, const size_t align, OUT bool *pResult)
{
  mFUNCTION_SETUP();

  mERROR_IF(pResult == nullptr, mR_ArgumentNull);

  void *pAligned;
  mERROR_CHECK(mMemoryAlignLo(pData, align, &pAligned));

  *pResult = (pData == pAligned);

  mRETURN_SUCCESS();
}
