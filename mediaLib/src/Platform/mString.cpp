#include "mediaLib.h"

#include "utf8proc.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "1hqQbvYNxOzsu2fl5hhaGsbVR/yCLjPedhe33Bx5XBcUq3PJV3CT5uU0+KCoMRLwKlS6bvwNWANqmW9k"
#endif

mchar_t mToChar(IN const char *c, const size_t size)
{
  utf8proc_int32_t codePoint;
  utf8proc_iterate(reinterpret_cast<const uint8_t *>(c), size, &codePoint);

  return mchar_t(codePoint);
}

bool mString_IsValidChar(const char *c, const size_t size, OUT OPTIONAL mchar_t *pChar /* = nullptr */, OUT OPTIONAL size_t *pCharSize /* = nullptr */)
{
  if (c == nullptr || size == 0)
    return false;

  utf8proc_int32_t codePoint;
  const ptrdiff_t characterSize = utf8proc_iterate(reinterpret_cast<const uint8_t *>(c), size, &codePoint);

  if (codePoint <= 0 || characterSize <= 0)
    return false;

  if (pChar != nullptr)
    *pChar = (mchar_t)codePoint;

  if (pCharSize != nullptr)
    *pCharSize = (size_t)characterSize;

  return true;
}

//////////////////////////////////////////////////////////////////////////

mString::mString() :
  text(nullptr),
  pAllocator(nullptr),
  bytes(0),
  count(0),
  capacity(0),
  hasFailed(false)
{ }

mString::mString(IN const char *text, const size_t size, IN OPTIONAL mAllocator *pAllocator) :
  text(nullptr),
  pAllocator(nullptr),
  bytes(0),
  count(0),
  capacity(0),
  hasFailed(false)
{
  mResult result = mR_Success;

  mERROR_CHECK_GOTO(mString_Create(this, text, size, pAllocator), result, epilogue);

  return;

epilogue:
  this->~mString();
  hasFailed = true;
}

mString::mString(IN const char *text, IN OPTIONAL mAllocator *pAllocator) : mString(text, strlen(text) + 1, pAllocator)
{ }

mString::mString(const wchar_t *text, const size_t size, IN OPTIONAL mAllocator *pAllocator /* = nullptr */) : mString()
{
  if (mFAILED(mString_Create(this, text, size, pAllocator)))
  {
    this->~mString();
    hasFailed = true;
  }
}

mString::mString(const wchar_t *text, IN OPTIONAL mAllocator *pAllocator /* = nullptr */) : mString(text, wcslen(text) + 1, pAllocator)
{ }

mString::~mString()
{
  if (text != nullptr && pAllocator != mSHARED_POINTER_FOREIGN_RESOURCE)
    mAllocator_FreePtr(pAllocator, &text);

  text = nullptr;
  pAllocator = nullptr;
  bytes = 0;
  count = 0;
  capacity = 0;
  hasFailed = false;
}

mString::mString(const mString &copy) :
  text(nullptr),
  pAllocator(nullptr),
  bytes(0),
  count(0),
  capacity(0),
  hasFailed(false)
{
  if (copy.hasFailed)
    goto epilogue;

  if (copy.capacity == 0)
    return;

  mResult result = mR_Success;

  pAllocator = copy.pAllocator != mSHARED_POINTER_FOREIGN_RESOURCE ? copy.pAllocator : nullptr;

  mERROR_CHECK_GOTO(mAllocator_AllocateZero(pAllocator, &text, copy.bytes), result, epilogue);
  capacity = bytes = copy.bytes;
  count = copy.count;

  mMemcpy(text, copy.text, bytes);

  return;

epilogue:
  this->~mString();
  hasFailed = true;
}

mString::mString(mString &&move) :
  text(move.text),
  pAllocator(move.pAllocator),
  bytes(move.bytes),
  count(move.count),
  capacity(move.capacity),
  hasFailed(move.hasFailed)
{
  move.text = nullptr;
  move.count = 0;
  move.bytes = 0;
  move.pAllocator = nullptr;
  move.capacity = 0;
  move.hasFailed = false;
}

mString & mString::operator=(const mString &copy)
{
  mResult result = mR_Success;

  this->hasFailed = false;

  if (this->capacity < copy.bytes)
  {
    mERROR_CHECK_GOTO(mAllocator_Reallocate(pAllocator, &text, copy.bytes), result, epilogue);
    this->capacity = copy.bytes;
    mMemcpy(text, copy.text, copy.bytes);
  }
  else if(copy.bytes > 0)
  {
    mMemcpy(text, copy.text, copy.bytes);
  }

  this->bytes = copy.bytes;
  this->count = copy.count;

  return *this;

epilogue:
  this->hasFailed = true;
  return *this;
}

mString & mString::operator=(mString &&move)
{
  if (move.text == nullptr)
  {
    this->bytes = 0;
    this->count = 0;
    this->hasFailed = move.hasFailed;

    if (this->text != nullptr)
      this->text[0] = '\0';
  }
  else
  {
    this->~mString();
    new (this) mString(std::move(move));
  }

  return *this;
}

size_t mString::Size() const
{
  return bytes;
}

size_t mString::Count() const
{
  return count;
}

mchar_t mString::operator[](const size_t index) const
{
  utf8proc_int32_t codePoint = UTF8PROC_ERROR_INVALIDOPTS;

  if (index >= count)
    return codePoint;

  size_t offset = 0;

  for (size_t i = 0; i <= index; ++i)
  {
    const ptrdiff_t characterSize = utf8proc_iterate(reinterpret_cast<const uint8_t *>(this->text) + offset, this->bytes - offset, &codePoint);

    if (characterSize < 0 || codePoint < 0)
      return codePoint;

    offset += (size_t)characterSize;

    if (characterSize == 0)
      break;
  }

  return codePoint;
}

mString mString::operator+(const mString &s) const
{
  if (s.bytes <= 1 || s.count <= 1 || s.c_str() == nullptr)
    return *this;
  else if (this->bytes <= 1 || this->count <= 1 || this->c_str() == nullptr)
    return s;

  mString ret;
  mResult result = mR_Success;

  mERROR_CHECK_GOTO(mAllocator_AllocateZero(this->pAllocator, &ret.text, this->bytes + s.bytes - 1), result, epilogue);
  ret.capacity = this->bytes + s.bytes - 1;
  mMemcpy(ret.text, this->text, this->bytes - 1);
  mMemcpy(ret.text + this->bytes - 1, s.text, s.bytes);
  ret.bytes = ret.capacity;
  ret.count = s.count + count - 1;
  ret.pAllocator = pAllocator;

  return ret;

epilogue:
  ret = mString();
  ret.hasFailed = true;
  return ret;
}

mString mString::operator+=(const mString &s)
{
  if (hasFailed || s.hasFailed)
    return *this;

  mResult result = mR_Success;

  mERROR_CHECK_GOTO(mString_Append(*this, s), result, epilogue);

  return *this;

epilogue:
  hasFailed = true;
  return *this;
}

bool mString::operator==(const mString &s) const
{
  bool ret;

  mString_Equals(*this, s, &ret);

  return ret;
}

bool mString::operator!=(const mString &s) const
{
  return !(*this == s);
}

mUtf8StringIterator mString::begin() const
{
  return mUtf8StringIterator(text, bytes);
}

mUtf8StringIterator mString::end() const
{
  return mUtf8StringIterator(nullptr, 0);
}

bool mString::StartsWith(const mString &s) const
{
  bool startsWith = false;

  const mResult result = mString_StartsWith(*this, s, &startsWith);

  return mSUCCEEDED(result) && startsWith;
}

bool mString::EndsWith(const mString &s) const
{
  bool endsWith = false;

  const mResult result = mString_EndsWith(*this, s, &endsWith);

  return mSUCCEEDED(result) && endsWith;
}

bool mString::Contains(const mString &s) const
{
  bool contained = false;

  const mResult result = mString_Contains(*this, s, &contained);

  return mSUCCEEDED(result) && contained;
}

mFUNCTION(mString_Create, OUT mString *pString, IN const char *text, IN OPTIONAL mAllocator *pAllocator /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pString == nullptr, mR_ArgumentNull);

#ifndef GIT_BUILD
  mASSERT(pAllocator != mSHARED_POINTER_FOREIGN_RESOURCE, "Attempting to create a string with FOREIGN RESOURCE allocator.");
#endif

  if (pAllocator == mSHARED_POINTER_FOREIGN_RESOURCE)
    pAllocator = &mDefaultAllocator;

  if (text == nullptr)
  {
    pString->hasFailed = false;

    if (pString->pAllocator != pAllocator)
    {
      mERROR_CHECK(mAllocator_FreePtr(pAllocator, &pString->text));
      pString->capacity = 0;
      pString->count = 0;
      pString->bytes = 0;
    }

    pString->pAllocator = pAllocator;

    if (pString->text != nullptr)
    {
      pString->text[0] = '\0';
      pString->count = 1;
      pString->bytes = 1;
    }

    mRETURN_SUCCESS();
  }

  mERROR_CHECK(mString_Create(pString, text, strlen(text) + 1, pAllocator));

  mRETURN_SUCCESS();
}

mFUNCTION(mString_Create, OUT mString *pString, IN const char *text, size_t size, IN OPTIONAL mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pString == nullptr, mR_ArgumentNull);

#ifndef GIT_BUILD
  mASSERT(pAllocator != mSHARED_POINTER_FOREIGN_RESOURCE, "Attempting to create a string with FOREIGN RESOURCE allocator.");
#endif

  if (pAllocator == mSHARED_POINTER_FOREIGN_RESOURCE)
    pAllocator = &mDefaultAllocator;

  if (text == nullptr)
  {
    pString->hasFailed = false;

    if (pString->pAllocator != pAllocator)
    {
      mERROR_CHECK(mAllocator_FreePtr(pAllocator, &pString->text));
      pString->capacity = 0;
      pString->count = 0;
      pString->bytes = 0;
    }

    pString->pAllocator = pAllocator;

    if (pString->text != nullptr)
    {
      pString->text[0] = '\0';
      pString->count = 1;
      pString->bytes = 1;
    }

    mRETURN_SUCCESS();
  }

  mERROR_CHECK(mStringLength(text, size, &size));
  size++;

  if (pString->pAllocator == pAllocator)
  {
    if (pString->capacity < size)
    {
      mERROR_CHECK(mAllocator_Reallocate(pString->pAllocator, &pString->text, size));
      pString->capacity = pString->bytes = size;
    }

    pString->text[0] = '\0';
    pString->bytes = size;
  }
  else
  {
    pString->~mString();
    *pString = mString();

    pString->pAllocator = pAllocator;

    mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &pString->text, size));
    pString->capacity = pString->bytes = size;
  }

  mMemcpy(pString->text, text, pString->bytes - 1);
  pString->text[pString->bytes - 1] = '\0';

  size_t offset = 0;
  pString->count = 0;

  while (offset < pString->bytes)
  {
    utf8proc_int32_t codePoint;
    ptrdiff_t characterSize;
    mERROR_IF((characterSize = utf8proc_iterate(reinterpret_cast<const uint8_t *>(pString->text) + offset, pString->bytes - offset, &codePoint)) < 0, mR_InternalError);
    mERROR_IF(codePoint < 0, mR_InternalError);
    offset += (size_t)characterSize;
    pString->count++;

    if (characterSize == 0)
      break;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mString_Create, OUT mString *pString, const wchar_t *text, IN OPTIONAL mAllocator *pAllocator /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pString == nullptr, mR_ArgumentNull);

#ifndef GIT_BUILD
  mASSERT(pAllocator != mSHARED_POINTER_FOREIGN_RESOURCE, "Attempting to create a string with FOREIGN RESOURCE allocator.");
#endif

  if (pAllocator == mSHARED_POINTER_FOREIGN_RESOURCE)
    pAllocator = &mDefaultAllocator;

  if (text == nullptr)
  {
    pString->hasFailed = false;

    if (pString->pAllocator != pAllocator)
    {
      mERROR_CHECK(mAllocator_FreePtr(pAllocator, &pString->text));
      pString->capacity = 0;
      pString->count = 0;
      pString->bytes = 0;
    }

    pString->pAllocator = pAllocator;

    if (pString->text != nullptr)
    {
      pString->text[0] = '\0';
      pString->count = 1;
      pString->bytes = 1;
    }

    mRETURN_SUCCESS();
  }

  const size_t size = wcslen(text) + 1;

  if (pString->pAllocator == pAllocator)
  {
    *pString = mString();
    pString->pAllocator = pAllocator;

    if (pString->capacity < size * mString_MaxUtf16CharInUtf8Chars)
    {
      mERROR_CHECK(mAllocator_Reallocate(pString->pAllocator, &pString->text, size * mString_MaxUtf16CharInUtf8Chars));
      pString->capacity = size * mString_MaxUtf16CharInUtf8Chars;
    }
  }
  else
  {
    pString->~mString();
    *pString = mString();

    pString->pAllocator = pAllocator;

    mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &pString->text, size * mString_MaxUtf16CharInUtf8Chars));
    pString->capacity = size * mString_MaxUtf16CharInUtf8Chars;
  }

  if (0 == (pString->bytes = WideCharToMultiByte(CP_UTF8, 0, text, (int32_t)size, pString->text, (int32_t)pString->capacity, nullptr, false)))
  {
    DWORD error = GetLastError();
    mUnused(error);

    mRETURN_RESULT(mR_InvalidParameter);
  }

  size_t offset = 0;
  pString->count = 0;

  while (offset < pString->bytes)
  {
    utf8proc_int32_t codePoint;
    ptrdiff_t characterSize;
    mERROR_IF((characterSize = utf8proc_iterate(reinterpret_cast<const uint8_t *>(pString->text) + offset, pString->bytes - offset, &codePoint)) < 0, mR_InternalError);
    mERROR_IF(codePoint < 0, mR_InternalError);
    offset += (size_t)characterSize;
    pString->count++;

    if (characterSize == 0)
      break;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mString_Create, OUT mString *pString, const wchar_t *text, const size_t bufferSize, IN OPTIONAL mAllocator *pAllocator /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pString == nullptr, mR_ArgumentNull);
  
#ifndef GIT_BUILD
  mASSERT(pAllocator != mSHARED_POINTER_FOREIGN_RESOURCE, "Attempting to create a string with FOREIGN RESOURCE allocator.");
#endif

  if (pAllocator == mSHARED_POINTER_FOREIGN_RESOURCE)
    pAllocator = &mDefaultAllocator;

  if (text == nullptr)
  {
    pString->hasFailed = false;

    if (pString->pAllocator != pAllocator)
    {
      mERROR_CHECK(mAllocator_FreePtr(pAllocator, &pString->text));
      pString->capacity = 0;
      pString->count = 0;
      pString->bytes = 0;
    }

    pString->pAllocator = pAllocator;

    if (pString->text != nullptr)
    {
      pString->text[0] = '\0';
      pString->count = 1;
      pString->bytes = 1;
    }

    mRETURN_SUCCESS();
  }

  const size_t size = wcsnlen_s(text, bufferSize) + 1;

  if (pString->pAllocator == pAllocator)
  {
    *pString = mString();
    pString->pAllocator = pAllocator;

    if (pString->capacity < size * mString_MaxUtf16CharInUtf8Chars)
    {
      mERROR_CHECK(mAllocator_Reallocate(pString->pAllocator, &pString->text, size * mString_MaxUtf16CharInUtf8Chars));
      pString->capacity = size * mString_MaxUtf16CharInUtf8Chars;
    }
  }
  else
  {
    pString->~mString();
    *pString = mString();

    pString->pAllocator = pAllocator;

    mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &pString->text, size * mString_MaxUtf16CharInUtf8Chars));
    pString->capacity = size * mString_MaxUtf16CharInUtf8Chars;
  }

  if (0 == (pString->bytes = WideCharToMultiByte(CP_UTF8, 0, text, (int32_t)(size - 1), pString->text, (int32_t)pString->capacity, nullptr, false)))
  {
    DWORD error = GetLastError();
    mUnused(error);

    mRETURN_RESULT(mR_InvalidParameter);
  }

  pString->text[pString->bytes] = '\0';
  pString->bytes++;

  size_t offset = 0;
  pString->count = 0;

  while (offset < pString->bytes)
  {
    utf8proc_int32_t codePoint;
    ptrdiff_t characterSize;
    mERROR_IF((characterSize = utf8proc_iterate(reinterpret_cast<const uint8_t *>(pString->text) + offset, pString->bytes - offset, &codePoint)) < 0, mR_InternalError);
    mERROR_IF(codePoint < 0, mR_InternalError);
    offset += (size_t)characterSize;
    pString->count++;

    if (characterSize == 0)
      break;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mString_Create, OUT mString *pString, const mString &from, IN OPTIONAL mAllocator *pAllocator /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pString == nullptr, mR_ArgumentNull);

#ifndef GIT_BUILD
  mASSERT(pAllocator != mSHARED_POINTER_FOREIGN_RESOURCE, "Attempting to create a string with FOREIGN RESOURCE allocator.");
#endif

  if (pAllocator == mSHARED_POINTER_FOREIGN_RESOURCE)
    pAllocator = &mDefaultAllocator;

  if (from.bytes <= 1)
  {
    if (pString->bytes > 1)
    {
      pString->bytes = 1;
      pString->count = 1;
      pString->text[0] = '\0';
    }

    mRETURN_SUCCESS();
  }

  if (pString->pAllocator == pAllocator)
  {
    *pString = mString();
    pString->pAllocator = pAllocator;

    if (pString->capacity < from.bytes)
    {
      mERROR_CHECK(mAllocator_Reallocate(pString->pAllocator, &pString->text, from.bytes));
      pString->capacity = pString->bytes = from.bytes;
    }
    else
    {
      pString->bytes = from.bytes;
    }
  }
  else
  {
    pString->~mString();
    *pString = mString();

    pString->pAllocator = pAllocator;

    mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &pString->text, from.bytes));
    pString->capacity = pString->bytes = from.bytes;
  }

  mMemcpy(pString->text, from.text, pString->bytes);

  pString->count = from.count;

  mRETURN_SUCCESS();
}

mFUNCTION(mString_Destroy, IN_OUT mString *pString)
{
  mFUNCTION_SETUP();

  mERROR_IF(pString == nullptr, mR_ArgumentNull);

  if (pString->text != nullptr)
    mERROR_CHECK(mAllocator_FreePtr(pString->pAllocator, &pString->text));

  pString->capacity = 0;
  pString->bytes = 0;
  pString->count = 0;
  pString->hasFailed = false;
  pString->pAllocator = nullptr;

  mRETURN_SUCCESS();
}

mFUNCTION(mString_Reserve, mString &string, const size_t size)
{
  mFUNCTION_SETUP();

  if (string.capacity < size)
  {
    mERROR_CHECK(mAllocator_Reallocate(string.pAllocator, &string.text, size));
    string.capacity = size;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mString_GetByteSize, const mString &string, OUT size_t *pSize)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSize == nullptr, mR_ArgumentNull);

  *pSize = string.bytes;

  mRETURN_SUCCESS();
}

mFUNCTION(mString_GetCount, const mString &string, OUT size_t *pLength)
{
  mFUNCTION_SETUP();

  mERROR_IF(pLength == nullptr, mR_ArgumentNull);

  *pLength = string.count;

  mRETURN_SUCCESS();
}

mFUNCTION(mString_ToWideString, const mString &string, OUT wchar_t *pWideString, const size_t bufferCount)
{
  size_t _unused;

  return mString_ToWideString(string, pWideString, bufferCount, &_unused);
}

mFUNCTION(mString_ToWideString, const mString &string, OUT wchar_t *pWideString, const size_t bufferCount, OUT size_t *pWideStringCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(pWideString == nullptr || pWideStringCount == nullptr, mR_ArgumentNull);
  mERROR_IF(string.hasFailed, mR_ResourceInvalid);

  if (string.text == nullptr)
  {
    mERROR_IF(bufferCount == 0, mR_ArgumentOutOfBounds);
    
    pWideString[0] = L'\0';
    *pWideStringCount = 1;
  }
  else
  {
    int32_t length = 0;

    if (0 >= (length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, string.text, (int32_t)string.bytes, pWideString, (int32_t)bufferCount)))
    {
      const DWORD error = GetLastError();

      switch (error)
      {
      case ERROR_INSUFFICIENT_BUFFER:
        mRETURN_RESULT(mR_IndexOutOfBounds);

      case ERROR_NO_UNICODE_TRANSLATION:
        mRETURN_RESULT(mR_InvalidParameter);

      case ERROR_INVALID_FLAGS:
      case ERROR_INVALID_PARAMETER:
      default:
        mRETURN_RESULT(mR_InternalError);
      }
    }

    *pWideStringCount = length;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mString_GetRequiredWideStringCount, const mString &string, OUT size_t *pWideStringCount)
{
  mFUNCTION_SETUP();

  mERROR_IF(pWideStringCount == nullptr, mR_ArgumentNull);
  mERROR_IF(string.hasFailed, mR_ResourceInvalid);

  if (string.text == nullptr)
  {
    *pWideStringCount = 1;
  }
  else
  {
    int32_t length = 0;

    if (0 >= (length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, string.text, (int32_t)string.bytes, nullptr, 0)))
    {
      const DWORD error = GetLastError();

      switch (error)
      {
      case ERROR_INSUFFICIENT_BUFFER:
        mRETURN_RESULT(mR_IndexOutOfBounds);

      case ERROR_NO_UNICODE_TRANSLATION:
        mRETURN_RESULT(mR_InvalidParameter);

      case ERROR_INVALID_FLAGS:
      case ERROR_INVALID_PARAMETER:
      default:
        mRETURN_RESULT(mR_InternalError);
      }
    }

    *pWideStringCount = length;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mString_Substring, const mString &text, OUT mString *pSubstring, const size_t startCharacter)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSubstring == nullptr, mR_ArgumentNull);
  mERROR_IF(startCharacter >= text.count, mR_IndexOutOfBounds);

  mERROR_CHECK(mString_Substring(text, pSubstring, startCharacter, text.count - startCharacter - 1));

  mRETURN_SUCCESS();
}

mFUNCTION(mString_Substring, const mString &text, OUT mString *pSubstring, const size_t startCharacter, const size_t length)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSubstring == nullptr, mR_ArgumentNull);
  mERROR_IF(startCharacter >= text.count || startCharacter + length >= text.count, mR_IndexOutOfBounds);
  
  size_t byteOffset = 0;
  size_t character = 0;

  while (character < startCharacter)
  {
    utf8proc_int32_t codePoint;
    ptrdiff_t characterSize;
    mERROR_IF((characterSize = utf8proc_iterate(reinterpret_cast<const uint8_t *>(text.text) + byteOffset, text.bytes - byteOffset, &codePoint)) < 0, mR_InternalError);
    mERROR_IF(codePoint < 0, mR_InternalError);
    byteOffset += (size_t)characterSize;
    character++;
  }

  *pSubstring = mString();
  char utf8char[4];

  while (character - startCharacter < length)
  {
    utf8proc_int32_t codePoint;
    ptrdiff_t characterSize;
    mERROR_IF((characterSize = utf8proc_iterate(reinterpret_cast<const uint8_t *>(text.text) + byteOffset, text.bytes - byteOffset, &codePoint)) < 0, mR_InternalError);
    mERROR_IF(codePoint < 0, mR_InternalError);
    character++;

    switch (characterSize)
    {
    case 4:
      utf8char[3] = *((char *)text.text + byteOffset + 3);
    case 3:
      utf8char[2] = *((char *)text.text + byteOffset + 2);
    case 2:
      utf8char[1] = *((char *)text.text + byteOffset + 1);
    case 1:
      utf8char[0] = *((char *)text.text + byteOffset + 0);
      break;
    }

    byteOffset += (size_t)characterSize;
    mERROR_CHECK(mString_Append(*pSubstring, utf8char, characterSize));
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mString_Append, mString &text, const mString &appendedText)
{
  mFUNCTION_SETUP();

  if (text.bytes > 0 && appendedText.bytes > 0)
  {
    if (text.capacity < text.bytes + appendedText.bytes - 1)
    {
      size_t newCapacity;

      if (text.capacity * 2 >= text.bytes + appendedText.bytes - 1)
        newCapacity = text.capacity * 2;
      else
        newCapacity = text.bytes + appendedText.bytes - 1;

      mERROR_CHECK(mAllocator_Reallocate(text.pAllocator, &text.text, newCapacity));
      text.capacity = newCapacity;
    }

    mMemcpy(text.text + text.bytes - 1, appendedText.text, appendedText.bytes);

    text.count += appendedText.count - 1;
    text.bytes += appendedText.bytes - 1;
  }
  else
  {
    if (appendedText.bytes == 0)
    {
      mRETURN_SUCCESS();
    }
    else
    {
      text.hasFailed = false;

      if (text.capacity < appendedText.bytes)
      {
        mERROR_CHECK(mAllocator_Reallocate(text.pAllocator, &text.text, appendedText.bytes));
        text.capacity = appendedText.bytes;
        mMemcpy(text.text, appendedText.text, appendedText.bytes);
      }
      else if (appendedText.bytes > 0)
      {
        mMemcpy(text.text, appendedText.text, appendedText.bytes);
      }

      text.bytes = appendedText.bytes;
      text.count = appendedText.count;

      mRETURN_SUCCESS();
    }
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mString_Append, mString &text, const char *appendedText)
{
  mFUNCTION_SETUP();

  mERROR_IF(appendedText == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mString_Append(text, appendedText, strlen(appendedText)));

  mRETURN_SUCCESS();
}

mFUNCTION(mString_Append, mString &text, const char *appendedText, const size_t size)
{
  mFUNCTION_SETUP();

  mERROR_IF(appendedText == nullptr, mR_ArgumentNull);

  size_t count, bytes;
  mERROR_CHECK(mInplaceString_GetCount_Internal(appendedText, size, &count, &bytes));

  const bool appendedTextNotTerminated = *(appendedText + bytes - 1) != '\0';

  if (text.bytes > 0 && bytes > 0)
  {
    if (text.capacity < text.bytes + bytes - 1 + appendedTextNotTerminated)
    {
      size_t newCapacity;

      if (text.capacity * 2 >= text.bytes + bytes - 1 + appendedTextNotTerminated)
        newCapacity = text.capacity * 2;
      else
        newCapacity = text.bytes + bytes - 1 + appendedTextNotTerminated;

      mERROR_CHECK(mAllocator_Reallocate(text.pAllocator, &text.text, newCapacity));
      text.capacity = newCapacity;
    }

    mMemcpy(text.text + text.bytes - 1, appendedText, bytes);

    if (!appendedTextNotTerminated)
    {
      text.count += count - 1;
      text.bytes += bytes - 1;
    }
    else
    {
      text.text[text.bytes - 1 + bytes] = '\0';

      text.count += count;
      text.bytes += bytes;
    }
  }
  else
  {
    if (bytes == 0)
    {
      mRETURN_SUCCESS();
    }
    else
    {
      text.hasFailed = false;

      if (text.capacity < bytes + appendedTextNotTerminated)
      {
        mERROR_CHECK(mAllocator_Reallocate(text.pAllocator, &text.text, bytes + appendedTextNotTerminated));
        text.capacity = bytes + appendedTextNotTerminated;
        
        mMemcpy(text.text, appendedText, bytes);
      }
      else if (bytes > 0)
      {
        mMemcpy(text.text, appendedText, bytes);
      }

      if (!appendedTextNotTerminated)
      {
        text.bytes = bytes;
        text.count = count;
      }
      else
      {
        text.text[bytes] = '\0';

        text.bytes = bytes + 1;
        text.count = count + 1;
      }

      mRETURN_SUCCESS();
    }
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mString_AppendUnsignedInteger, mString &text, const uint64_t value)
{
  mFUNCTION_SETUP();

  char txt[64];
  _ui64toa(value, txt, 10);

  mERROR_CHECK(mString_Append(text, txt, sizeof(txt)));

  mRETURN_SUCCESS();
}

mFUNCTION(mString_AppendInteger, mString &text, const int64_t value)
{
  mFUNCTION_SETUP();

  char txt[64];
  _i64toa(value, txt, 10);

  mERROR_CHECK(mString_Append(text, txt, sizeof(txt)));

  mRETURN_SUCCESS();
}

mFUNCTION(mString_AppendBool, mString &text, const bool value)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mString_Append(text, value ? "true" : "false"));

  mRETURN_SUCCESS();
}

mFUNCTION(mString_AppendDouble, mString &text, const double_t value)
{
  mFUNCTION_SETUP();

  char txt[128];
  mERROR_IF(0 > sprintf_s(txt, "%f", value), mR_InternalError);

  mERROR_CHECK(mString_Append(text, txt, sizeof(txt)));

  mRETURN_SUCCESS();
}

mFUNCTION(mString_ToDirectoryPath, OUT mString *pString, const mString &text)
{
  mFUNCTION_SETUP();

  mERROR_IF(pString == nullptr, mR_ArgumentNull);

  if (text.text == nullptr || text.bytes <= 1 || text.count <= 1)
  {
    *pString = "";
    mRETURN_SUCCESS();
  }

  if (pString->pAllocator != text.pAllocator && pString->capacity > 0)
  {
    mERROR_CHECK(mAllocator_FreePtr(pString->pAllocator, &pString->text));
    pString->bytes = 0;
    pString->capacity = 0;
    pString->count = 0;
  }

  pString->pAllocator = text.pAllocator;

  mERROR_CHECK(mString_Reserve(*pString, text.bytes + 1));
  *pString = text;
  mERROR_IF(pString->hasFailed, mR_InternalError);

  size_t offset = 0;
  bool lastWasSlash = false;

  for (size_t i = 0; i < pString->count; ++i)
  {
    utf8proc_int32_t codePoint;
    ptrdiff_t characterSize;
    mERROR_IF((characterSize = utf8proc_iterate(reinterpret_cast<const uint8_t *>(pString->text) + offset, pString->bytes - offset, &codePoint)) < 0, mR_InternalError);
    mERROR_IF(codePoint < 0, mR_InternalError);

    if (characterSize == 1)
    {
      if (lastWasSlash && (pString->text[offset] == '/' || pString->text[offset] == '\\') && i != 1)
      {
        pString->bytes -= characterSize;
        mERROR_CHECK(mMemmove(&pString->text[offset], &pString->text[offset + characterSize], pString->bytes - offset));
        --pString->count;
        --i;
        continue;
      }
      else if (pString->text[offset] == '/')
      {
        pString->text[offset] = '\\';
        lastWasSlash = true;
      }
      else if (pString->text[offset] == '\\')
      {
        lastWasSlash = true;
      }
      else if (pString->text[offset] == '\0')
      {
        ; // don't change `lastWasSlash`.
      }
      else
      {
        lastWasSlash = false;
      }
    }
    else
    {
      lastWasSlash = false;
    }

    offset += (size_t)characterSize;

    if (characterSize == 0)
      break;
  }

  if (!lastWasSlash)
  {
    pString->text[offset - 1] = '\\';
    pString->text[offset] = '\0';
    ++pString->bytes;
    ++pString->count;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mString_ToFilePath, OUT mString *pString, const mString &text)
{
  mFUNCTION_SETUP();

  mERROR_IF(pString == nullptr, mR_ArgumentNull);

  if (text.text == nullptr || text.bytes <= 1 || text.count <= 1)
  {
    *pString = "";
    mRETURN_SUCCESS();
  }

  *pString = text;
  mERROR_IF(pString->hasFailed, mR_InternalError);

  size_t offset = 0;
  bool lastWasSlash = false;

  for (size_t i = 0; i < pString->count; ++i)
  {
    utf8proc_int32_t codePoint;
    ptrdiff_t characterSize;
    mERROR_IF((characterSize = utf8proc_iterate(reinterpret_cast<const uint8_t *>(pString->text) + offset, pString->bytes - offset, &codePoint)) < 0, mR_InternalError);
    mERROR_IF(codePoint < 0, mR_InternalError);

    if (characterSize == 1)
    {
      if (lastWasSlash && (pString->text[offset] == '/' || pString->text[offset] == '\\') && i != 1)
      {
        pString->bytes -= characterSize;
        mERROR_CHECK(mMemmove(&pString->text[offset], &pString->text[offset + characterSize], pString->bytes - offset));
        --pString->count;
        continue;
      }
      else if (pString->text[offset] == '/')
      {
        pString->text[offset] = '\\';
        lastWasSlash = true;
      }
      else if (pString->text[offset] == '\\')
      {
        lastWasSlash = true;
      }
      else
      {
        lastWasSlash = false;
      }
    }
    else
    {
      lastWasSlash = false;
    }

    offset += (size_t)characterSize;

    if (characterSize == 0)
      break;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mString_Equals, const mString &stringA, const mString &stringB, bool *pAreEqual)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAreEqual == nullptr, mR_ArgumentNull);

  if ((stringA.bytes <= 1) && (stringB.bytes <= 1)) // "" should equal an uninitialized mString.
  {
    *pAreEqual = true;
    mRETURN_SUCCESS();
  }

  if (stringA.bytes != stringB.bytes || stringA.count != stringB.count)
  {
    *pAreEqual = false;
    mRETURN_SUCCESS();
  }

  size_t offset = 0;
  
  for (size_t i = 0; i < stringA.count; ++i)
  {
    utf8proc_int32_t codePointA;
    ptrdiff_t characterSizeA = utf8proc_iterate(reinterpret_cast<const uint8_t *>(stringA.text) + offset, stringA.bytes - offset, &codePointA);

    utf8proc_int32_t codePointB;
    utf8proc_iterate(reinterpret_cast<const uint8_t *>(stringB.text) + offset, stringA.bytes - offset, &codePointB);

    if (codePointA != codePointB)
    {
      *pAreEqual = false;
      mRETURN_SUCCESS();
    }

    offset += characterSizeA;

    if (characterSizeA == 0)
      break;
  }

  *pAreEqual = true;

  mRETURN_SUCCESS();
}

mFUNCTION(mString_ForEachChar, const mString &string, const std::function<mResult(mchar_t, char *, size_t)> &function)
{
  mFUNCTION_SETUP();

  mERROR_IF(string.hasFailed, mR_InvalidParameter);
  mERROR_IF(function == nullptr, mR_ArgumentNull);

  size_t offset = 0;

  while (offset + 1 < string.bytes)
  {
    utf8proc_int32_t codePoint;
    ptrdiff_t characterSize;
    mERROR_IF((characterSize = utf8proc_iterate(reinterpret_cast<const uint8_t *>(string.text) + offset, string.bytes - offset, &codePoint)) < 0, mR_InternalError);
    mERROR_IF(codePoint < 0, mR_InternalError);

    if (characterSize == 0)
      break;

    const mResult result = function((mchar_t)codePoint, string.text + offset, (size_t)characterSize);
    
    if (mFAILED(result))
    {
      if (result == mR_Break)
        break;
      else
        mRETURN_RESULT(result);
    }

    offset += (size_t)characterSize;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mString_StartsWith, const mString &stringA, const mString &start, OUT bool *pStartsWith)
{
  mFUNCTION_SETUP();

  mERROR_IF(pStartsWith == nullptr, mR_ArgumentNull);
  
  *pStartsWith = false;

  mERROR_IF(stringA.hasFailed || start.hasFailed, mR_InvalidParameter);

  if (stringA.bytes <= 1 || start.bytes <= 1)
  {
    *pStartsWith = start.bytes <= 1;
    mRETURN_SUCCESS();
  }

  mERROR_IF(start.count > stringA.count, mR_Success); // pStartsWith is already false.

  size_t offset = 0;

  for (size_t i = 0; i < start.count - 1; ++i) // Exclude null char.
  {
    utf8proc_int32_t codePointA;
    ptrdiff_t characterSizeA = utf8proc_iterate(reinterpret_cast<const uint8_t *>(stringA.text) + offset, stringA.bytes - offset, &codePointA);

    utf8proc_int32_t codePointB;
    utf8proc_iterate(reinterpret_cast<const uint8_t *>(start.text) + offset, stringA.bytes - offset, &codePointB);

    mERROR_IF(codePointA != codePointB, mR_Success); // pStartsWith is already false.

    offset += characterSizeA;

    if (characterSizeA == 0)
      break;
  }

  *pStartsWith = true;

  mRETURN_SUCCESS();
}

mFUNCTION(mString_EndsWith, const mString &stringA, const mString &end, OUT bool *pEndsWith)
{
  mFUNCTION_SETUP();

  mERROR_IF(pEndsWith == nullptr, mR_ArgumentNull);

  *pEndsWith = false;

  mERROR_IF(stringA.hasFailed || end.hasFailed, mR_InvalidParameter);

  if (stringA.bytes <= 1 || end.bytes <= 1)
  {
    *pEndsWith = end.bytes <= 1;
    mRETURN_SUCCESS();
  }

  mERROR_IF(end.count > stringA.count, mR_Success);

  size_t stringOffset = 0;

  for (size_t i = 0; i < stringA.count - end.count; ++i) // Exclude null char.
  {
    utf8proc_int32_t codePoint;
    const ptrdiff_t characterSize = utf8proc_iterate(reinterpret_cast<const uint8_t *>(stringA.text) + stringOffset, stringA.bytes - stringOffset, &codePoint);
    stringOffset += characterSize;

    if (characterSize == 0)
      break;
  }

  size_t endOffset = 0;

  for (size_t i = 0; i < end.count - 1; ++i) // Exclude null char.
  {
    utf8proc_int32_t codePointA;
    const ptrdiff_t characterSizeA = utf8proc_iterate(reinterpret_cast<const uint8_t *>(stringA.text) + stringOffset, stringA.bytes - stringOffset, &codePointA);

    utf8proc_int32_t codePointB;
    utf8proc_iterate(reinterpret_cast<const uint8_t *>(end.text) + endOffset, stringA.bytes - endOffset, &codePointB);

    mERROR_IF(codePointA != codePointB, mR_Success); // pEndsWith is already false.

    stringOffset += characterSizeA;
    endOffset += characterSizeA;

    if (characterSizeA == 0)
      break;
  }

  *pEndsWith = true;

  mRETURN_SUCCESS();
}

// `pKmp` should be zeroed.
void mString_GetKmp(const mString &string, IN_OUT mchar_t *pString, IN_OUT size_t *pKmp)
{
  pKmp[0] = 0;

  const size_t length = string.count - 1;
  size_t offset = 0;

  for (size_t i = 0; i < length; i++)
  {
    utf8proc_int32_t codePoint;
    const ptrdiff_t characterSize = utf8proc_iterate(reinterpret_cast<const uint8_t *>(string.text) + offset, string.bytes - offset, &codePoint);
    offset += characterSize;

    pString[i] = codePoint;

    if (characterSize == 0)
      return;
  }

  for (size_t i = 1, k = 0; i < length; ++i)
  {
    while (k > 0 && pString[k] != pString[i])
      k = pKmp[k - 1];
    
    if (pString[k] == pString[i])
      ++k;
    
    pKmp[i] = k;
  }
}

 void mString_FindNextWithKMP(const mString &string, const size_t offsetChar, const size_t offsetBytes, IN const mchar_t *pFind, const size_t findCountWithoutNull, IN const size_t *pKmp, OUT size_t *pStartChar, OUT size_t *pOffset, OUT bool *pContained)
{
   *pContained = false;

   const size_t chars = string.count - offsetChar - 1;

   if (findCountWithoutNull == 0 || findCountWithoutNull > chars)
     return;
   
   size_t offset = offsetBytes;

   for (size_t i = 0, k = 0; i < chars; i++)
   {
     utf8proc_int32_t codePoint;
     const ptrdiff_t characterSize = utf8proc_iterate(reinterpret_cast<const uint8_t *>(string.text) + offset, string.bytes - offset, &codePoint);
     offset += characterSize;

     while (k > 0 && pFind[k] != codePoint)
       k = pKmp[k - 1];
     
     if (pFind[k] == codePoint)
       ++k;
     
     if (k == findCountWithoutNull)
     {
       *pStartChar = offsetChar + i - k + 1;
       *pOffset = offset;
       *pContained = true;

       return;
     }
   }
}

mFUNCTION(mString_FindFirst, const mString &string, const mString &find, OUT size_t *pStartChar, OUT bool *pContained)
{
  mFUNCTION_SETUP();

  mERROR_IF(pContained == nullptr || pStartChar == nullptr, mR_ArgumentNull);
  
  *pContained = false;

  mERROR_IF(string.hasFailed || find.hasFailed, mR_InvalidParameter);
  mERROR_IF(pStartChar == nullptr || pContained == nullptr, mR_InvalidParameter);
  
  if (string.count <= 1 || find.count <= 1)
  {
    *pStartChar = 0;
    *pContained = find.count <= 1;
    mRETURN_SUCCESS();
  }

  // TODO: Add a brute force variant for small strings.

  size_t *pKmp = nullptr;
  mchar_t *pChars = nullptr;

  mDEFER_CALL_2(mAllocator_FreePtr, &mDefaultTempAllocator, &pKmp);
  mDEFER_CALL_2(mAllocator_FreePtr, &mDefaultTempAllocator, &pChars);
  mERROR_CHECK(mAllocator_AllocateZero(&mDefaultTempAllocator, &pKmp, find.count - 1));
  mERROR_CHECK(mAllocator_Allocate(&mDefaultTempAllocator, &pChars, find.count - 1));

  mString_GetKmp(find, pChars, pKmp);

  size_t _unusedOffset;
  mString_FindNextWithKMP(string, 0, 0, pChars, find.count - 1, pKmp, pStartChar, &_unusedOffset, pContained);

  mRETURN_SUCCESS();
}

mFUNCTION(mString_Contains, const mString &string, const mString &contained, OUT bool *pContains)
{
  mFUNCTION_SETUP();

  mERROR_IF(pContains == nullptr, mR_ArgumentNull);

  *pContains = false;

  mERROR_IF(string.hasFailed || contained.hasFailed, mR_InvalidParameter);

  if (string.count <= 1 || contained.count <= 1)
  {
    *pContains = contained.count <= 1;
    mRETURN_SUCCESS();
  }

  // TODO: Add a brute force variant for small strings.

  size_t *pKmp = nullptr;
  mchar_t *pChars = nullptr;

  mDEFER_CALL_2(mAllocator_FreePtr, &mDefaultTempAllocator, &pKmp);
  mDEFER_CALL_2(mAllocator_FreePtr, &mDefaultTempAllocator, &pChars);
  mERROR_CHECK(mAllocator_AllocateZero(&mDefaultTempAllocator, &pKmp, contained.count - 1));
  mERROR_CHECK(mAllocator_Allocate(&mDefaultTempAllocator, &pChars, contained.count - 1));

  mString_GetKmp(contained, pChars, pKmp);

  size_t _unusedStartChar, _unusedOffset;
  mString_FindNextWithKMP(string, 0, 0, pChars, contained.count - 1, pKmp, &_unusedStartChar, &_unusedOffset, pContains);

  mRETURN_SUCCESS();
}

mFUNCTION(mString_TrimStart, const mString &string, const mchar_t trimmedChar, OUT mString *pTrimmedString)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTrimmedString == nullptr, mR_ArgumentNull);
  mERROR_IF(string.hasFailed || trimmedChar == 0, mR_InvalidParameter);
  
  size_t byteOffset = 0;
  size_t charOffset = 0;

  for (; charOffset < string.count - 1; charOffset++)
  {
    utf8proc_int32_t codePoint;
    const ptrdiff_t characterSize = utf8proc_iterate(reinterpret_cast<const uint8_t *>(string.text) + byteOffset, string.bytes - byteOffset, &codePoint);
    
    if (codePoint != trimmedChar || characterSize == 0)
      break;

    byteOffset += characterSize;
  }

  mERROR_CHECK(mString_Create(pTrimmedString, "", pTrimmedString->pAllocator));
  mERROR_CHECK(mString_Reserve(*pTrimmedString, string.bytes - byteOffset));
  pTrimmedString->text[0] = '\0';

  if (charOffset < string.count - 1)
  {
    pTrimmedString->bytes = string.bytes - byteOffset;
    pTrimmedString->count = string.count - charOffset;
    mERROR_CHECK(mMemcpy(pTrimmedString->text, string.text + byteOffset, string.bytes - byteOffset));
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mString_TrimEnd, const mString &string, const mchar_t trimmedChar, OUT mString *pTrimmedString)
{
  mFUNCTION_SETUP();

  mERROR_IF(pTrimmedString == nullptr, mR_ArgumentNull);
  mERROR_IF(string.hasFailed || trimmedChar == 0, mR_InvalidParameter);

  size_t byteOffset = 0;
  
  size_t firstMatchingChar = 0;
  size_t firstMatchingByte = 0;

  for (size_t charOffset = 0; charOffset < string.count - 1; charOffset++)
  {
    utf8proc_int32_t codePoint;
    const ptrdiff_t characterSize = utf8proc_iterate(reinterpret_cast<const uint8_t *>(string.text) + byteOffset, string.bytes - byteOffset, &codePoint);
    byteOffset += characterSize;

    if (codePoint != trimmedChar)
    {
      firstMatchingChar = charOffset + 1;
      firstMatchingByte = byteOffset;
    }

    if (characterSize == 0)
      break;
  }

  mERROR_CHECK(mString_Create(pTrimmedString, "", pTrimmedString->pAllocator));
  mERROR_CHECK(mString_Reserve(*pTrimmedString, firstMatchingByte + 1));
  pTrimmedString->text[0] = '\0';

  if (firstMatchingByte > 0)
  {
    pTrimmedString->bytes = firstMatchingByte + 1;
    pTrimmedString->count = firstMatchingChar + 1;
    mERROR_CHECK(mMemcpy(pTrimmedString->text, string.text, firstMatchingByte));
    pTrimmedString->text[firstMatchingByte] = '\0';
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mString_RemoveChar, const mString &string, const mchar_t remove, OUT mString *pResult)
{
  mFUNCTION_SETUP();

  mERROR_IF(pResult == nullptr, mR_ArgumentNull);
  mERROR_IF(string.hasFailed || remove == 0, mR_InvalidParameter);

  mERROR_CHECK(mString_Create(pResult, "", pResult->pAllocator));
  mERROR_CHECK(mString_Reserve(*pResult, string.bytes));
  pResult->text[0] = '\0';

  bool lastWasMatch = false;
  size_t firstNoMatchByte = 0;
  size_t sourceOffset = 0;
  size_t destinationOffset = 0;
  size_t destinationCharCount = 0;
  size_t destinationCharSize = 0;
  ptrdiff_t characterSize = 0;

  for (size_t charOffset = 0; charOffset < string.count - 1; charOffset++)
  {
    utf8proc_int32_t codePoint;
    characterSize = utf8proc_iterate(reinterpret_cast<const uint8_t *>(string.text) + sourceOffset, string.bytes - sourceOffset, &codePoint);
    sourceOffset += characterSize;

    if (characterSize == 0)
      break;

    if (codePoint == remove)
    {
      const size_t size = (sourceOffset - characterSize) - firstNoMatchByte;

      if (!lastWasMatch && size > 0)
      {
        mERROR_CHECK(mMemcpy(pResult->text + destinationOffset, string.text + firstNoMatchByte, size));
        destinationOffset += size;
      }
      
      firstNoMatchByte = sourceOffset;
      lastWasMatch = true;
    }
    else
    {
      lastWasMatch = false;
      ++destinationCharCount;
      destinationCharSize += characterSize;
    }
  }

  if (!lastWasMatch)
  {
    const size_t size = sourceOffset - firstNoMatchByte;
    mERROR_CHECK(mMemcpy(pResult->text + destinationOffset, string.text + firstNoMatchByte, size));
    destinationOffset += size;
  }

  pResult->text[destinationOffset] = '\0';
  pResult->count = destinationCharCount + 1;
  pResult->bytes = destinationCharSize + 1;

  mRETURN_SUCCESS();
}

mFUNCTION(mString_RemoveString, const mString &string, const mString &remove, OUT mString *pResult)
{
  mFUNCTION_SETUP();

  mERROR_IF(pResult == nullptr, mR_ArgumentNull);
  mERROR_IF(string.hasFailed || remove.hasFailed, mR_InvalidParameter);

  if (remove.bytes <= 1 || string.bytes <= 1 || remove.count > string.count)
  {
    mERROR_CHECK(mString_Create(pResult, string, pResult->pAllocator));
    mRETURN_SUCCESS();
  }

  mERROR_CHECK(mString_Create(pResult, "", pResult->pAllocator));
  mERROR_CHECK(mString_Reserve(*pResult, string.bytes));

  size_t *pKmp = nullptr;
  mchar_t *pChars = nullptr;

  mDEFER_CALL_2(mAllocator_FreePtr, &mDefaultTempAllocator, &pKmp);
  mDEFER_CALL_2(mAllocator_FreePtr, &mDefaultTempAllocator, &pChars);
  mERROR_CHECK(mAllocator_AllocateZero(&mDefaultTempAllocator, &pKmp, remove.count - 1));
  mERROR_CHECK(mAllocator_Allocate(&mDefaultTempAllocator, &pChars, remove.count - 1));

  mString_GetKmp(remove, pChars, pKmp);

  size_t charOffsetAfter, byteOffsetAfter;
  size_t charOffset = 0;
  size_t byteOffset = 0;
  size_t destinationOffset = 0;
  size_t destinationNotBytes = 0;
  size_t destinationNotCount = 0;
  bool contained = false;

  while (charOffset < string.count - 1)
  {
    mString_FindNextWithKMP(string, charOffset, byteOffset, pChars, remove.count - 1, pKmp, &charOffsetAfter, &byteOffsetAfter, &contained);

    if (!contained)
    {
      const size_t size = string.bytes - 1 - byteOffset;
      mERROR_CHECK(mMemcpy(pResult->text + destinationOffset, string.text + byteOffset, size));
      destinationOffset += size;

      break;
    }
    else if (byteOffsetAfter - (remove.bytes - 1) != byteOffset)
    {
      const size_t size = byteOffsetAfter - (remove.bytes - 1) - byteOffset;
      mERROR_CHECK(mMemcpy(pResult->text + destinationOffset, string.text + byteOffset, size));
      destinationOffset += size;
    }

    charOffset = charOffsetAfter;
    byteOffset = byteOffsetAfter;

    destinationNotBytes += remove.bytes - 1;
    destinationNotCount += remove.count - 1;
  }

  pResult->text[destinationOffset] = '\0';
  pResult->bytes = string.bytes - destinationNotBytes;
  pResult->count = string.count - destinationNotCount;

  mRETURN_SUCCESS();
}

mFUNCTION(mString_Replace, const mString &string, const mString &replace, const mString &with, OUT mString *pResult)
{
  mFUNCTION_SETUP();

  mERROR_IF(pResult == nullptr, mR_ArgumentNull);
  mERROR_IF(string.hasFailed || replace.hasFailed || with.hasFailed, mR_InvalidParameter);

  if (replace.bytes <= 1 || string.bytes <= 1 || replace.count > string.count)
  {
    mERROR_CHECK(mString_Create(pResult, string, pResult->pAllocator));
    mRETURN_SUCCESS();
  }

  mERROR_CHECK(mString_Create(pResult, "", pResult->pAllocator));
  mERROR_CHECK(mString_Reserve(*pResult, string.bytes));

  size_t *pKmp = nullptr;
  mchar_t *pChars = nullptr;

  mDEFER_CALL_2(mAllocator_FreePtr, &mDefaultTempAllocator, &pKmp);
  mDEFER_CALL_2(mAllocator_FreePtr, &mDefaultTempAllocator, &pChars);
  mERROR_CHECK(mAllocator_AllocateZero(&mDefaultTempAllocator, &pKmp, replace.count - 1));
  mERROR_CHECK(mAllocator_Allocate(&mDefaultTempAllocator, &pChars, replace.count - 1));

  mString_GetKmp(replace, pChars, pKmp);

  size_t charOffsetAfter, byteOffsetAfter;
  size_t charOffset = 0;
  size_t byteOffset = 0;
  size_t destinationOffset = 0;
  size_t destinationNotBytes = 0;
  size_t destinationNotCount = 0;
  size_t destinationAddedBytes = 0;
  size_t destinationAddedCount = 0;
  bool contained = false;

  while (charOffset < string.count - 1)
  {
    mString_FindNextWithKMP(string, charOffset, byteOffset, pChars, replace.count - 1, pKmp, &charOffsetAfter, &byteOffsetAfter, &contained);

    if (!contained)
    {
      const size_t size = string.bytes - 1 - byteOffset;

      if (destinationOffset + size >= pResult->capacity)
        mERROR_CHECK(mString_Reserve(*pResult, mMax(pResult->capacity * 2, pResult->capacity + destinationOffset + size)));

      mERROR_CHECK(mMemcpy(pResult->text + destinationOffset, string.text + byteOffset, size));
      destinationOffset += size;

      break;
    }
    else if (byteOffsetAfter - (replace.bytes - 1) != byteOffset)
    {
      const size_t size = byteOffsetAfter - (replace.bytes - 1) - byteOffset;

      if (destinationOffset + size + with.bytes >= pResult->capacity)
        mERROR_CHECK(mString_Reserve(*pResult, mMax(pResult->capacity * 2, pResult->capacity + destinationOffset + size + with.bytes)));

      mERROR_CHECK(mMemcpy(pResult->text + destinationOffset, string.text + byteOffset, size));
      destinationOffset += size;

      mERROR_CHECK(mMemcpy(pResult->text + destinationOffset, with.text, with.bytes - 1));
      destinationOffset += with.bytes - 1;

      destinationAddedBytes += with.bytes - 1;
      destinationAddedCount += with.count - 1;
    }
    else
    {
      if (destinationOffset + with.bytes >= pResult->capacity)
        mERROR_CHECK(mString_Reserve(*pResult, mMax(pResult->capacity * 2, pResult->capacity + destinationOffset + with.bytes)));

      mERROR_CHECK(mMemcpy(pResult->text + destinationOffset, with.text, with.bytes - 1));
      destinationOffset += with.bytes - 1;

      destinationAddedBytes += with.bytes - 1;
      destinationAddedCount += with.count - 1;
    }

    charOffset = charOffsetAfter;
    byteOffset = byteOffsetAfter;

    destinationNotBytes += replace.bytes - 1;
    destinationNotCount += replace.count - 1;
  }

  pResult->text[destinationOffset] = '\0';
  pResult->bytes = string.bytes - destinationNotBytes + destinationAddedBytes;
  pResult->count = string.count - destinationNotCount + destinationAddedCount;

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mInplaceString_GetCount_Internal, IN const char *text, const size_t maxSize, OUT size_t *pCount, OUT size_t *pSize)
{
  mFUNCTION_SETUP();

  size_t offset = 0;
  size_t count = 0;

  while (offset < maxSize)
  {
    utf8proc_int32_t codePoint;
    ptrdiff_t characterSize;
    mERROR_IF((characterSize = utf8proc_iterate(reinterpret_cast<const uint8_t *>(text) + offset, maxSize - offset, &codePoint)) < 0, mR_InternalError);
    mERROR_IF(codePoint < 0, mR_InternalError);
    offset += (size_t)characterSize;
    count++;

    if (characterSize == 0 || codePoint == 0)
    {
      *pCount = count;
      *pSize = offset;
      break;
    }
  }

  *pCount = count;
  *pSize = offset;

  mRETURN_SUCCESS();
}

bool mInplaceString_StringsAreEqual_Internal(IN const char *textA, IN const char *textB, const size_t bytes, const size_t count)
{
  size_t offset = 0;

  for (size_t i = 0; i < count; ++i)
  {
    utf8proc_int32_t codePointA;
    ptrdiff_t characterSizeA = utf8proc_iterate(reinterpret_cast<const uint8_t *>(textA)  + offset, bytes - offset, &codePointA);

    utf8proc_int32_t codePointB;
    utf8proc_iterate(reinterpret_cast<const uint8_t *>(textB)  + offset, bytes - offset, &codePointB);

    if (codePointA != codePointB)
    {
      return false;
    }

    offset += characterSizeA;

    if (characterSizeA == 0)
      break;
  }

  return true;
}

//////////////////////////////////////////////////////////////////////////

mUtf8StringIterator::mUtf8StringIterator(char *string) :
  mUtf8StringIterator(string, strlen(string) + 1)
{ }

mUtf8StringIterator::mUtf8StringIterator(char *string, size_t bytes) :
  string(string),
  bytes(bytes)
{ }

mUtf8StringIterator & mUtf8StringIterator::begin()
{
  return *this;
}

mUtf8StringIterator mUtf8StringIterator::end()
{
  return *this;
}

bool mUtf8StringIterator::operator!=(const mUtf8StringIterator &)
{
  if (position + 1 >= bytes)
    return false;

  characterSize = utf8proc_iterate(reinterpret_cast<const uint8_t *>(string)  + position, bytes - position, &codePoint);
  position += (size_t)characterSize;
  charCount++;

  if (characterSize == 0 || codePoint == 0)
    return false;

  return true;
}

mUtf8StringIterator & mUtf8StringIterator::operator++()
{
  return *this;
}

mIteratedString mUtf8StringIterator::operator*()
{
  return mIteratedString(string + position - characterSize, codePoint, characterSize, charCount - 1, position - characterSize);
}

mIteratedString::mIteratedString(char *character, const mchar_t codePoint, const size_t characterSize, const size_t index, const size_t offset) :
  character(character),
  codePoint(codePoint),
  characterSize(characterSize),
  index(index),
  offset(offset)
{ }
