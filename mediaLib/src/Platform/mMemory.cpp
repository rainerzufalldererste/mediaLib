#include "mediaLib.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "Rx6R99ErVSc74oQfABhd3lN1Z/7LbxaWj5aofNcPw+3mWMReFYhCMscW350rxwq48gtWVS1+nWs+s8Dw"
#endif

mFUNCTION(mStringLength, const char *text, const size_t maxLength, OUT size_t *pLength)
{
  mFUNCTION_SETUP();

  mERROR_IF(text == nullptr || pLength == nullptr, mR_ArgumentNull);

  *pLength = strnlen_s(text, maxLength);

  mRETURN_SUCCESS();
}

mFUNCTION(mSprintf, OUT char *buffer, const size_t bufferLength, const char *formatString, ...)
{
  mFUNCTION_SETUP();

  mERROR_IF(buffer == nullptr || formatString == nullptr, mR_ArgumentNull);

  va_list args;
  va_start(args, formatString);
  const int length = vsprintf_s(buffer, bufferLength, formatString, args);
  va_end(args);

  mERROR_IF(length < 0, mR_ArgumentOutOfBounds);

  mRETURN_SUCCESS();
}

mFUNCTION(mSprintfWithLength, OUT char *buffer, const size_t bufferLength, const char *formatString, OUT size_t *pLength, ...)
{
  mFUNCTION_SETUP();

  mERROR_IF(buffer == nullptr || formatString == nullptr || pLength == nullptr, mR_ArgumentNull);

  *pLength = 0;

  va_list args;
  va_start(args, formatString);
  const int length = vsprintf_s(buffer, bufferLength, formatString, args);
  va_end(args);

  mERROR_IF(length < 0, mR_ArgumentOutOfBounds);

  *pLength = length;

  mRETURN_SUCCESS();
}

mFUNCTION(mStringCopy, OUT char *buffer, const size_t bufferLength, const char *source, const size_t sourceLength)
{
  mFUNCTION_SETUP();

  mERROR_IF(buffer == nullptr || source == nullptr, mR_ArgumentNull);

  const errno_t error = strncpy_s(buffer, bufferLength, source, sourceLength);

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

mFUNCTION(mStringConcat, OUT char *buffer, const size_t bufferLength, const char *source, const size_t sourceLength)
{
  mFUNCTION_SETUP();

  mERROR_IF(buffer == nullptr || source == nullptr, mR_ArgumentNull);

  const errno_t error = strncat_s(buffer, bufferLength, source, sourceLength);

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

mFUNCTION(mStringLength, const wchar_t *text, const size_t maxLength, OUT size_t *pLength)
{
  mFUNCTION_SETUP();

  mERROR_IF(text == nullptr || pLength == nullptr, mR_ArgumentNull);

  *pLength = wcsnlen_s(text, maxLength);

  mRETURN_SUCCESS();
}

mFUNCTION(mSprintf, OUT wchar_t *buffer, const size_t bufferLength, const wchar_t *formatString, ...)
{
  mFUNCTION_SETUP();

  mERROR_IF(buffer == nullptr || formatString == nullptr, mR_ArgumentNull);

  va_list args;
  va_start(args, formatString);
  const int result = vswprintf_s(buffer, bufferLength, formatString, args);
  va_end(args);

  mERROR_IF(result < 0, mR_ArgumentOutOfBounds);

  mRETURN_SUCCESS();
}

mFUNCTION(mSprintfWithLength, OUT wchar_t *buffer, const size_t bufferLength, const wchar_t *formatString, OUT size_t *pLength, ...)
{
  mFUNCTION_SETUP();

  mERROR_IF(buffer == nullptr || formatString == nullptr || pLength == nullptr, mR_ArgumentNull);

  va_list args;
  va_start(args, formatString);
  const int result = vswprintf_s(buffer, bufferLength, formatString, args);
  va_end(args);

  mERROR_IF(result < 0, mR_ArgumentOutOfBounds);

  *pLength = (size_t)result;

  mRETURN_SUCCESS();
}

mFUNCTION(mStringCopy, OUT wchar_t *buffer, const size_t bufferLength, const wchar_t *source, const size_t sourceLength)
{
  mFUNCTION_SETUP();

  mERROR_IF(buffer == nullptr || source == nullptr, mR_ArgumentNull);

  const errno_t error = wcsncpy_s(buffer, bufferLength, source, sourceLength);

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

mFUNCTION(mStringConcet, OUT wchar_t *buffer, const size_t bufferLength, const wchar_t *source, const size_t sourceLength)
{
  mFUNCTION_SETUP();

  mERROR_IF(buffer == nullptr || source == nullptr, mR_ArgumentNull);

  const errno_t error = wcsncat_s(buffer, bufferLength, source, sourceLength);

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

mFUNCTION(mStringChar, const char *text, const size_t maxLength, const char character, OUT size_t *pLength)
{
  mFUNCTION_SETUP();

  mERROR_IF(text == nullptr || pLength == nullptr, mR_ArgumentNull);

  *pLength = (size_t)((char *)memchr((void *)text, (int)character, strnlen_s(text, maxLength)) - (char *)text);

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
