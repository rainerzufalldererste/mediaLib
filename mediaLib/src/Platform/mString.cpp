#include "mString.h"
#include "utf8proc.h"

mchar_t mToChar(IN const char *c, const size_t size)
{
  utf8proc_int32_t codePoint;
  utf8proc_iterate((uint8_t *)c, size, &codePoint);

  return mchar_t(codePoint);
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
  if (text != nullptr)
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

  pAllocator = copy.pAllocator;

  mERROR_CHECK_GOTO(mAllocator_AllocateZero(pAllocator, &text, copy.bytes), result, epilogue);
  capacity = bytes = copy.bytes;
  count = copy.count;

  mERROR_CHECK_GOTO(mAllocator_Copy(pAllocator, text, copy.text, bytes), result, epilogue);

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
    mERROR_CHECK_GOTO(mAllocator_Copy(pAllocator, text, copy.text, copy.bytes), result, epilogue);
  }
  else if(copy.bytes > 0)
  {
    mERROR_CHECK_GOTO(mAllocator_Copy(pAllocator, text, copy.text, copy.bytes), result, epilogue);
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
  this->~mString();
  new (this) mString(std::move(move));
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
    const ptrdiff_t characterSize = utf8proc_iterate((uint8_t *)this->text + offset, this->bytes - offset, &codePoint);

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
  mERROR_CHECK_GOTO(mAllocator_Copy(this->pAllocator, ret.text, this->text, this->bytes - 1), result, epilogue);
  mERROR_CHECK_GOTO(mAllocator_Copy(s.pAllocator, ret.text + this->bytes - 1, s.text, s.bytes), result, epilogue);
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

const char *mString::c_str() const
{
  return text;
}

mFUNCTION(mString_Create, OUT mString *pString, IN const char *text, IN OPTIONAL mAllocator *pAllocator /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pString == nullptr, mR_ArgumentNull);

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
    *pString = mString();
    pString->pAllocator = pAllocator;

    if (pString->capacity < size)
    {
      mERROR_CHECK(mAllocator_Reallocate(pString->pAllocator, &pString->text, size));
      pString->capacity = pString->bytes = size;
    }
  }
  else
  {
    pString->~mString();
    *pString = mString();

    pString->pAllocator = pAllocator;

    mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &pString->text, size));
    pString->capacity = pString->bytes = size;
  }

  mERROR_CHECK(mAllocator_Copy(pString->pAllocator, pString->text, text, pString->bytes));

  size_t offset = 0;
  pString->count = 0;

  while (offset < pString->bytes)
  {
    utf8proc_int32_t codePoint;
    ptrdiff_t characterSize;
    mERROR_IF((characterSize = utf8proc_iterate((uint8_t *)pString->text + offset, pString->bytes - offset, &codePoint)) < 0, mR_InternalError);
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

    if (pString->capacity < size * sizeof(wchar_t))
    {
      mERROR_CHECK(mAllocator_Reallocate(pString->pAllocator, &pString->text, size * sizeof(wchar_t)));
      pString->capacity = size * sizeof(wchar_t);
    }
  }
  else
  {
    pString->~mString();
    *pString = mString();

    pString->pAllocator = pAllocator;

    mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &pString->text, size * sizeof(wchar_t)));
    pString->capacity = size * sizeof(wchar_t);
  }

  if (0 == (pString->bytes = WideCharToMultiByte(CP_UTF8, 0, text, (int)size, pString->text, (int)pString->capacity, nullptr, false)))
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
    mERROR_IF((characterSize = utf8proc_iterate((uint8_t *)pString->text + offset, pString->bytes - offset, &codePoint)) < 0, mR_InternalError);
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

    if (pString->capacity < size * sizeof(wchar_t) * 2)
    {
      mERROR_CHECK(mAllocator_Reallocate(pString->pAllocator, &pString->text, size * sizeof(wchar_t) * 2));
      pString->capacity = size * sizeof(wchar_t) * 2;
    }
  }
  else
  {
    pString->~mString();
    *pString = mString();

    pString->pAllocator = pAllocator;

    mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &pString->text, size * sizeof(wchar_t) * 2));
    pString->capacity = size * sizeof(wchar_t) * 2;
  }
  
  if (0 == (pString->bytes = WideCharToMultiByte(CP_UTF8, 0, text, (int)size, pString->text, (int)pString->capacity, nullptr, false)))
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
    mERROR_IF((characterSize = utf8proc_iterate((uint8_t *)pString->text + offset, pString->bytes - offset, &codePoint)) < 0, mR_InternalError);
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
  }
  else
  {
    pString->~mString();
    *pString = mString();

    pString->pAllocator = pAllocator;

    mERROR_CHECK(mAllocator_AllocateZero(pAllocator, &pString->text, from.bytes));
    pString->capacity = pString->bytes = from.bytes;
  }

  mERROR_CHECK(mAllocator_Copy(pString->pAllocator, pString->text, from.text, pString->bytes));

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

mFUNCTION(mString_ToWideString, const mString &string, OUT wchar_t *pWideString, const size_t bufferSize)
{
  mFUNCTION_SETUP();

  mERROR_IF(pWideString == nullptr, mR_ArgumentNull);
  
  if (0 == MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, string.text, (int)string.bytes, pWideString, (int)bufferSize))
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

  mRETURN_SUCCESS();
}

mFUNCTION(mString_ToWideString, const mString &string, OUT wchar_t *pWideString, const size_t bufferSize, OUT size_t *pWideStringLength)
{
  mFUNCTION_SETUP();

  mERROR_IF(pWideStringLength == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mString_ToWideString(string, pWideString, bufferSize));

  *pWideStringLength = wcsnlen_s(pWideString, bufferSize);

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
    mERROR_IF((characterSize = utf8proc_iterate((uint8_t *)text.text + byteOffset, text.bytes - byteOffset, &codePoint)) < 0, mR_InternalError);
    mERROR_IF(codePoint < 0, mR_InternalError);
    byteOffset += (size_t)characterSize;
    character++;
  }

  *pSubstring = mString();
  char utf8char[5] = { 0, 0, 0, 0, 0 };

  while (character - startCharacter < length)
  {
    utf8proc_int32_t codePoint;
    ptrdiff_t characterSize;
    mERROR_IF((characterSize = utf8proc_iterate((uint8_t *)text.text + byteOffset, text.bytes - byteOffset, &codePoint)) < 0, mR_InternalError);
    mERROR_IF(codePoint < 0, mR_InternalError);
    character++;
    *(uint32_t *)utf8char = 0;

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
    mERROR_CHECK(mString_Append(*pSubstring, utf8char));
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
      if (text.capacity * 2 >= text.bytes + appendedText.bytes - 1)
      {
        const size_t newCapacity = text.capacity *= 2;

        mERROR_CHECK(mAllocator_Reallocate(text.pAllocator, &text.text, newCapacity));
        text.capacity = newCapacity;
      }
      else
      {
        const size_t newCapacity = text.bytes + appendedText.bytes - 1;

        mERROR_CHECK(mAllocator_Reallocate(text.pAllocator, &text.text, newCapacity));
        text.capacity = newCapacity;
      }
    }
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
        mERROR_CHECK(mAllocator_Copy(text.pAllocator, text.text, appendedText.text, appendedText.bytes));
      }
      else if (appendedText.bytes > 0)
      {
        mERROR_CHECK(mAllocator_Copy(text.pAllocator, text.text, appendedText.text, appendedText.bytes));
      }

      text.bytes = appendedText.bytes;
      text.count = appendedText.count;

      mRETURN_SUCCESS();
    }
  }

  mERROR_CHECK(mAllocator_Copy(text.pAllocator, text.text + text.bytes - 1, appendedText.text, appendedText.bytes));

  text.count += appendedText.count - 1;
  text.bytes += appendedText.bytes - 1;

  mRETURN_SUCCESS();
}

mFUNCTION(mString_AppendUnsignedInteger, mString &text, const uint64_t value)
{
  mFUNCTION_SETUP();

  mString format;
  mERROR_CHECK(mString_CreateFormat(&format, text.pAllocator, "%" PRIu64, value));
  mERROR_CHECK(mString_Append(text, format));

  mRETURN_SUCCESS();
}

mFUNCTION(mString_AppendInteger, mString &text, const int64_t value)
{
  mFUNCTION_SETUP();

  mString format;
  mERROR_CHECK(mString_CreateFormat(&format, text.pAllocator, "%" PRIi64, value));
  mERROR_CHECK(mString_Append(text, format));

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

  mString format;
  mERROR_CHECK(mString_CreateFormat(&format, text.pAllocator, "%f", value));
  mERROR_CHECK(mString_Append(text, format));

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
    mERROR_IF((characterSize = utf8proc_iterate((uint8_t *)pString->text + offset, pString->bytes - offset, &codePoint)) < 0, mR_InternalError);
    mERROR_IF(codePoint < 0, mR_InternalError);

    if (characterSize == 1)
    {
      if (lastWasSlash && (pString->text[offset] == '/' || pString->text[offset] == '\\'))
      {
        pString->bytes -= characterSize;
        mERROR_CHECK(mAllocator_Move(pString->pAllocator, &pString->text[offset], &pString->text[offset + characterSize], pString->bytes - offset));
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
        ; // don't chang `lastWasSlash`.
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
    mERROR_IF((characterSize = utf8proc_iterate((uint8_t *)pString->text + offset, pString->bytes - offset, &codePoint)) < 0, mR_InternalError);
    mERROR_IF(codePoint < 0, mR_InternalError);

    if (characterSize == 1)
    {
      if (lastWasSlash && (pString->text[offset] == '/' || pString->text[offset] == '\\'))
      {
        pString->bytes -= characterSize;
        mERROR_CHECK(mAllocator_Move(pString->pAllocator, &pString->text[offset], &pString->text[offset + characterSize], pString->bytes - offset));
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

  if (stringA.bytes != stringB.bytes || stringA.count != stringB.count)
  {
    *pAreEqual = false;
    mRETURN_SUCCESS();
  }

  size_t offset = 0;
  
  for (size_t i = 0; i < stringA.count; ++i)
  {
    utf8proc_int32_t codePointA;
    ptrdiff_t characterSizeA = utf8proc_iterate((uint8_t *)stringA.text + offset, stringA.bytes - offset, &codePointA);

    utf8proc_int32_t codePointB;
    utf8proc_iterate((uint8_t *)stringB.text + offset, stringA.bytes - offset, &codePointB);

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
    mERROR_IF((characterSize = utf8proc_iterate((uint8_t *)string.text + offset, string.bytes - offset, &codePoint)) < 0, mR_InternalError);
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

mFUNCTION(mInplaceString_GetCount_Internal, IN const char *text, const size_t maxSize, OUT size_t *pCount, OUT size_t *pSize)
{
  mFUNCTION_SETUP();

  size_t offset = 0;
  size_t count = 0;

  while (offset + 1 < maxSize)
  {
    utf8proc_int32_t codePoint;
    ptrdiff_t characterSize;
    mERROR_IF((characterSize = utf8proc_iterate((uint8_t *)text + offset, maxSize - offset, &codePoint)) < 0, mR_InternalError);
    mERROR_IF(codePoint < 0, mR_InternalError);
    offset += (size_t)characterSize;
    count++;

    if (characterSize == 0 || codePoint == 0)
    {
      count--;
      offset--;
      break;
    }
  }

  *pCount = count + 1;
  *pSize = offset + 1;

  mRETURN_SUCCESS();
}

bool mInplaceString_StringsAreEqual_Internal(IN const char *textA, IN const char *textB, const size_t bytes, const size_t count)
{
  size_t offset = 0;

  for (size_t i = 0; i < count; ++i)
  {
    utf8proc_int32_t codePointA;
    ptrdiff_t characterSizeA = utf8proc_iterate((uint8_t *)textA + offset, bytes - offset, &codePointA);

    utf8proc_int32_t codePointB;
    utf8proc_iterate((uint8_t *)textB + offset, bytes - offset, &codePointB);

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

  characterSize = utf8proc_iterate((uint8_t *)string + position, bytes - position, &codePoint);
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
