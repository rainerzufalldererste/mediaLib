#ifndef mFormat_h__
#define mFormat_h__

#include "mediaLib.h"
#include "mString.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "SNB8RRXrKNE7m+OqRwekSM/LU9EAlIIyWOgZP4BAGyI+cd9g1RtO7Oz6jo6uHQKbvdIo50g3ZL8+0fwE"
#endif

template <typename T>
struct mIsFloatFormattable_t
{
  static constexpr bool value = std::is_floating_point<T>::value;
};

template <typename T>
struct mIsIntegerFormattable_t
{
  static constexpr bool value = std::is_integral<T>::value && !std::is_same<T, bool>::value;
};

template <typename T>
struct mIsListFormattable_t
{
  static constexpr bool value = false;
  typedef void value_type;
};

template <typename T>
struct mIsVectorFormattable_t
{
  static constexpr bool value = false;
  typedef void value_type;
  static constexpr size_t count = 0;
};

template <typename T>
struct mIsMapFormattable_t
{
  static constexpr bool value = false;
  typedef void key_type;
  typedef void value_type;
};

template <typename T>
struct mIsFormattable_t
{
  static constexpr bool value = mIsIntegerFormattable_t<T>::value || mIsFloatFormattable_t<T>::value || mIsListFormattable_t<T>::value || mIsVectorFormattable_t<T>::value || mIsMapFormattable_t<T>::value;
};

template <>
struct mIsFormattable_t<bool>
{
  static constexpr bool value = true;
};

template <>
struct mIsFormattable_t<char *>
{
  static constexpr bool value = true;
};

template <>
struct mIsFormattable_t<wchar_t *>
{
  static constexpr bool value = true;
};

template <>
struct mIsFormattable_t<const char *>
{
  static constexpr bool value = true;
};

template <>
struct mIsFormattable_t<const wchar_t *>
{
  static constexpr bool value = true;
};

template <size_t count>
struct mIsFormattable_t<char[count]>
{
  static constexpr bool value = true;
};

template <size_t count>
struct mIsFormattable_t<wchar_t[count]>
{
  static constexpr bool value = true;
};

template <typename T, size_t count>
struct mIsListFormattable_t<T[count]>
{
  static constexpr bool value = !std::is_same<T, char>::value && !std::is_same<T, wchar_t>::value && mIsFormattable_t<T>::value;
  typedef T value_type;
};

template <typename T>
inline constexpr bool mIsFormattable(const T &)
{
  return mIsFormattable_t<T>::value;
}

template <typename T>
inline constexpr bool mIsIntegerFormattable(const T &)
{
  return mIsIntegerFormattable_t<T>::value;
}

template <typename T>
inline constexpr bool mIsFloatFormattable(const T &)
{
  return mIsFloatFormattable_t<T>::value;
}

template <typename T>
inline constexpr bool mIsListFormattable(const T &)
{
  return mIsListFormattable_t<T>::value && mIsFormattable_t<mIsListFormattable_t<T>::value_type>::value;
}

template <typename T>
inline constexpr bool mIsVectorFormattable(const T &)
{
  return mIsVectorFormattable_t<T>::value && mIsFormattable_t<mIsVectorFormattable_t<T>::value_type>::value;
}

template <typename T>
inline constexpr bool mIsMapFormattable(const T &)
{
  return mIsMapFormattable_t<T>::value && mIsFormattable_t<mIsMapFormattable_t<T>::key_type>::value && mIsFormattable_t<mIsMapFormattable_t<T>::value_type>::value;
}

enum mFormatSignOption
{
  mFSO_NegativeOnly,
  mFSO_NegativeOrFill,
  mFSO_Both,
  mFSO_Never
};

enum mFormatBaseOption
{
  mFBO_Decimal = 10,
  mFBO_Hexadecimal = 0x10,
  mFBO_Binary = 0b10
};

enum mFormatAlign
{
  mFA_Left,
  mFA_Right,
  mFA_Center
};

enum mFormatDigitGroupingOption
{
  mFDGO_Thousand, // SI, commonly used throughout the western world. (1.234.567 / 1,234,567 / 1 234 567 / 1'234'567)
  mFDGO_TenThousand, // For Chinese Numerals. (1,2345,6789 / 1亿2345万6789 / 1億2345萬6789)
  mFDGO_Indian // For the Indian numbering system. (12,34,56,789)
};

// For Hexadecimal & Binary Values, Strings.
enum mFormatOverflowBehaviour
{
  mFOB_AlignRight,
  mFOB_AlignLeft
};

struct mFormatState
{
  mAllocator *pAllocator = &mDefaultAllocator;
  char *textStart = nullptr;
  size_t textCapacity = 0;
  size_t textPosition = 0;
  bool inFormatStatement = false;
  char fillCharacter = ' ';
  bool fillCharacterIsZero = false;
  size_t decimalSeparatorLength = 1;
  char digitGroupingChars[5] = { ',', '\0' };
  bool groupDigits = false;
  size_t digitGroupingCharLength = 1;
  mFormatDigitGroupingOption digitGroupingOption = mFDGO_Thousand;
  char decimalSeparatorChars[5] = { '.', '\0' };
  char listMapStartChar = '{';
  char listMapEndChar = '}';
  char listMapSeparatorChar = ',';
  char mapEntryStartChar = '[';
  char mapEntryEndChar = ']';
  char mapEntryKeyValueSeparator = ':';
  char vectorStartChar = '[';
  char vectorEndChar = ']';
  char vectorSeparatorChar = ',';
  bool listMapSpaceAfterStart = true;
  bool listMapSpaceBeforeEnd = true;
  bool mapEntrySpaceAfterStart = true;
  bool mapEntrySpaceBeforeEnd = true;
  bool vectorSpaceAfterStart = false;
  bool vectorSpaceBeforeEnd = false;
  bool listMapSpaceAfterSeparator = true;
  bool vectorSpaceAfterSeparator = true;
  bool mapSpaceAfterKeyValueSeparator = true;
  size_t minChars = 0;
  size_t maxChars = INT64_MAX; // yes, not uint64_t max.
  size_t fractionalDigits = 5;
  mFormatSignOption signOption = mFSO_NegativeOnly;
  bool alignSign = true;
  mFormatBaseOption integerBaseOption = mFBO_Decimal;
  bool hexadecimalUpperCase = true;
  mFormatAlign stringAlign = mFA_Left;
  mFormatAlign numberAlign = mFA_Right;
  bool stringOverflowEllipsis = true;
  size_t stringOverflowEllipsisLength = 3;
  size_t stringOverflowEllipsisCount = 3;
  char stringOverflowEllipsisChars[8] = { '.', '.', '.', ' ', '\0' };
  mFormatOverflowBehaviour numberOverflow = mFOB_AlignRight;
  size_t listMapMaxLength = 25;
  size_t listMapContinuationLength = 3; // doesn't include a null terminator.
  char listMapContinuation[5] = { '.', '.', '.', ' ', '\0' };
  size_t infinityCount = 8;
  size_t infinityBytes = 8;
  char infinityChars[32] = "Infinity";
  size_t negativeInfinityCount = 9;
  size_t negativeInfinityBytes = 9;
  char negativeInfinityChars[32] = "-Infinity";
  size_t nanCount = 3;
  size_t nanBytes = 3;
  char nanChars[32] = "NaN";
  char exponentChar = 'e';
  bool scientificNotation = false;
  size_t trueCount = 4;
  size_t trueBytes = 4;
  char trueChars[32] = "true";
  size_t falseCount = 5;
  size_t falseBytes = 5;
  char falseChars[32] = "false";
  bool adaptiveFractionalDigits = true;
  bool adaptiveFloatScientificNotation = false;
  int16_t adaptiveScientificNotationPositiveExponentThreshold = 10;
  int16_t adaptiveScientificNotationNegativeExponentThreshold = -4;

  mFormatState() = default;
  mFormatState(const mFormatState &copy) = default;

  // This should only be used
  inline mFormatState operator = (const mFormatState &copy)
  {
    new (this) mFormatState(copy);

    textStart = nullptr;
    textCapacity = 0;
    textPosition = 0;
    inFormatStatement = false;
    pAllocator = &mDefaultAllocator;
  }

  inline void SetTo(const mFormatState &copy)
  {
    bool previousInFormatStatement = inFormatStatement;
    mAllocator *pPreviousAllocator = pAllocator;
    char *previousTextStart = textStart;
    size_t previousTextCapacity = textCapacity;
    size_t previousTextPosition = textPosition;

    new (this) mFormatState(copy);

    inFormatStatement = previousInFormatStatement;
    pAllocator = pPreviousAllocator;
    textStart = previousTextStart;
    textCapacity = previousTextCapacity;
    textPosition = previousTextPosition;
  }
};

mFormatState & mFormat_GetState();
void mFormatState_ResetCulture();

#pragma warning (push)
#pragma warning (disable: 4702)

template <typename T, typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value && !std::is_same<bool, T>::value && !std::is_same<char, T>::value && !std::is_enum<T>::value>::type* = nullptr>
inline size_t mFormat_GetMaxBytes(const T &value, const mFormatState &fs)
{
  size_t signChars = 0;
  size_t numberChars;

  switch (fs.integerBaseOption)
  {
  default:
  case mFBO_Decimal:
  {
    switch (fs.signOption)
    {
    case mFSO_Both:
    case mFSO_NegativeOrFill:
    case mFSO_NegativeOnly:
      signChars = 1;
      break;
    }

    const T abs = value < 0 ? value : -value; // because otherwise the minimum value couldn't be converted to a valid signed equivalent.

    mIF_CONSTEXPR (sizeof(value) == 1)
      numberChars = 3;
    else mIF_CONSTEXPR (sizeof(value) == 2)
      numberChars = 5;
    else mIF_CONSTEXPR (sizeof(value) == 4)
      numberChars = 10;
    else
      numberChars = 19;

    if (fs.groupDigits)
    {
      size_t groupingChars;

      switch (fs.digitGroupingOption)
      {
      default:
      case mFDGO_Thousand:
        groupingChars = (numberChars - 1) / 3;
        break;

      case mFDGO_TenThousand:
        groupingChars = (numberChars - 1) / 4;
        break;

      case mFDGO_Indian:
        groupingChars = (size_t)!!((numberChars - 1) / 3);

        if (groupingChars)
          groupingChars += (numberChars - 4) / 2;
        
        break;
      }

      return signChars + mClamp(numberChars * fs.digitGroupingCharLength, fs.minChars, fs.maxChars);
    }
    else
    {
      return signChars + mClamp(numberChars, fs.minChars, fs.maxChars);
    }

    break;
  }

  case mFBO_Hexadecimal:
  case mFBO_Binary:
  {
    return mFormat_GetCount((typename mUnsignedEquivalent<T>::type)value, fs);
  }
  }
}

template <typename T, typename std::enable_if<std::is_enum<T>::value>::type* = nullptr>
inline size_t mFormat_GetMaxBytes(const T &value, const mFormatState &fs)
{
  return mFormat_GetMaxBytes((typename mEnumEquivalentIntegerType<T>::type)value, fs);
}

template <typename T, typename std::enable_if<std::is_enum<T>::value>::type* = nullptr>
inline size_t mFormat_GetCount(const T &value, const mFormatState &fs)
{
  return mFormat_GetCount((typename mEnumEquivalentIntegerType<T>::type)value, fs);
}

template <typename T, typename std::enable_if<std::is_enum<T>::value>::type* = nullptr>
inline size_t _mFormat_Append(const T &value, const mFormatState &fs, char *text)
{
  return _mFormat_Append((typename mEnumEquivalentIntegerType<T>::type)value, fs, text);
}

inline size_t _mFormat_GetDigitGroupingCharCount(const size_t numberChars, const mFormatState &fs)
{
  if (numberChars == 0)
    return 0;

  switch (fs.digitGroupingOption)
  {
  default:
  case mFDGO_Thousand:
    return (numberChars - 1) / 3;

  case mFDGO_TenThousand:
    return (numberChars - 1) / 4;

  case mFDGO_Indian:
    size_t groupingChars = (size_t)!!((numberChars - 1) / 3);

    if (groupingChars)
      groupingChars += (numberChars - 4) / 2;

    return groupingChars;
  }
}

template <typename T, typename std::enable_if<std::is_integral<T>::value && std::is_unsigned<T>::value && !std::is_same<bool, T>::value>::type* = nullptr>
inline size_t mFormat_GetMaxBytes(const T &value, const mFormatState &fs)
{
  size_t signChars = 0;
  size_t numberChars;

  switch (fs.integerBaseOption)
  {
  default:
  case mFBO_Decimal:
  {
    switch (fs.signOption)
    {
    case mFSO_Both:
    case mFSO_NegativeOrFill:
      signChars = 1;
      break;
    }

    mIF_CONSTEXPR (sizeof(value) == 1)
      numberChars = 3;
    else mIF_CONSTEXPR (sizeof(value) == 2)
      numberChars = 5;
    else mIF_CONSTEXPR (sizeof(value) == 4)
      numberChars = 10;
    else
      numberChars = 20;

    if (fs.groupDigits)
      return signChars + mClamp(numberChars + _mFormat_GetDigitGroupingCharCount(numberChars, fs) * fs.digitGroupingCharLength, fs.minChars, fs.maxChars);
    else
      return signChars + mClamp(numberChars, fs.minChars, fs.maxChars);

    break;
  }

  case mFBO_Hexadecimal:
  {
    return sizeof(value) * 2;
  }

  case mFBO_Binary:
  {
    return sizeof(value) * 8;
  }
  }
}

template <typename T, typename std::enable_if<std::is_same<bool, T>::value>::type* = nullptr>
inline size_t mFormat_GetMaxBytes(const T &, const mFormatState &fs)
{
  return mMax(fs.trueBytes, fs.falseBytes);
}

template <typename T, typename std::enable_if<std::is_same<bool, T>::value>::type* = nullptr>
inline size_t mFormat_GetCount(const T &value, const mFormatState &fs)
{
  return value ? fs.trueChars : fs.falseChars;
}

inline size_t mFormat_GetMaxBytes(const char &, const mFormatState &)
{
  return 1;
}

inline size_t mFormat_GetCount(const char &value, const mFormatState &)
{
  if (value == 0)
    return 0;

  return 1;
}

inline size_t mFormat_GetMaxBytes(const wchar_t &, const mFormatState &)
{
  return 3; // i.e. 0x2026 will become 0xE2 0x80 0xA6 in UTF-8
}

inline size_t mFormat_GetCount(const mString &value, const mFormatState &fs)
{
  if (value.bytes <= 1)
    return 0;

  return mClamp(value.count - 1, fs.minChars, fs.maxChars);
}

template <typename T, typename std::enable_if<std::is_same<T, mString>::value>::type* = nullptr>
inline size_t mFormat_GetMaxBytes(const T &value, const mFormatState &fs)
{
  if (value.bytes <= 1)
    return 0;

  return mClamp(value.bytes - 1, fs.minChars * 4, fs.maxChars * 4);
}

template <size_t TCount>
inline size_t mFormat_GetCount(const mInplaceString<TCount> &value, const mFormatState &fs)
{
  if (value.bytes <= 1)
    return 0;

  return mClamp(value.count - 1, fs.minChars, fs.maxChars);
}

template <size_t TCount>
inline size_t mFormat_GetMaxBytes(const mInplaceString<TCount> &value, const mFormatState &fs)
{
  if (value.bytes <= 1)
    return 0;

  return mClamp(value.bytes - 1, fs.minChars * 4, fs.maxChars * 4);
}

size_t _mFormat_GetStringCount(const char *value, const size_t length);

inline size_t mFormat_GetCount(const char *value, const mFormatState &fs)
{
  if (value == nullptr)
    return 0;

  return mClamp(_mFormat_GetStringCount(value, strlen(value)), fs.minChars, fs.maxChars);
}

template <typename T, typename std::enable_if<std::is_same<T, char *>::value>::type* = nullptr>
inline size_t mFormat_GetMaxBytes(T value, const mFormatState &fs)
{
  if (value == nullptr)
    return 0;

  return mClamp(strlen(value), fs.minChars * 4, fs.maxChars * 4);
}

template <typename T, typename std::enable_if<std::is_same<T, const char *>::value>::type* = nullptr>
inline size_t mFormat_GetMaxBytes(T value, const mFormatState &fs)
{
  if (value == nullptr)
    return 0;

  return mClamp(strlen(value), fs.minChars * 4, fs.maxChars * 4);
}

template <typename T, typename std::enable_if<std::is_same<T, const wchar_t *>::value>::type* = nullptr>
inline size_t mFormat_GetMaxBytes(const T value, const mFormatState &fs)
{
  if (value == nullptr)
    return 0;

  return mClamp(wcslen(value) * mString_MaxUtf16CharInUtf8Chars + 1, fs.minChars * 4, fs.maxChars * 4);
}

template <typename T, typename std::enable_if<std::is_same<T, wchar_t *>::value>::type* = nullptr>
inline size_t mFormat_GetMaxBytes(const T value, const mFormatState &fs)
{
  if (value == nullptr)
    return 0;

  return mClamp(wcslen(value) * mString_MaxUtf16CharInUtf8Chars + 1, fs.minChars * 4, fs.maxChars * 4);
}

template <size_t TCount>
inline size_t mFormat_GetCount(char (&value)[TCount], const mFormatState &fs)
{
  return mClamp(_mFormat_GetStringCount(value, TCount), fs.minChars, fs.maxChars);
}

template <size_t TCount>
inline size_t mFormat_GetMaxBytes(char (&)[TCount], const mFormatState &fs)
{
  return mClamp(TCount, fs.minChars * 4, fs.maxChars * 4);
}

template <size_t TCount>
inline size_t mFormat_GetMaxBytes(wchar_t (&)[TCount], const mFormatState &fs)
{
  return mClamp(TCount * mString_MaxUtf16CharInUtf8Chars + 1, fs.minChars * 4, fs.maxChars * 4);
}

template <size_t TCount>
inline size_t mFormat_GetCount(const char (&value)[TCount], const mFormatState &fs)
{
  return mClamp(_mFormat_GetStringCount(value, TCount), fs.minChars, fs.maxChars);
}

template <size_t TCount>
inline size_t mFormat_GetMaxBytes(const char (&)[TCount], const mFormatState &fs)
{
  return mClamp(TCount, fs.minChars * 4, fs.maxChars * 4);
}

template <size_t TCount>
inline size_t mFormat_GetMaxBytes(const wchar_t (&)[TCount], const mFormatState &fs)
{
  return mClamp(TCount * mString_MaxUtf16CharInUtf8Chars + 1, fs.minChars * 4, fs.maxChars * 4);
}

template <typename T, typename std::enable_if<std::is_integral<T>::value && !std::is_signed<T>::value && !std::is_same<bool, T>::value>::type* = nullptr>
inline size_t mFormat_GetCount(const T &value, const mFormatState &fs)
{
  size_t signChars = 0;
  size_t numberChars = 0;

  switch (fs.integerBaseOption)
  {
  default:
  case mFBO_Decimal:
  {
    switch (fs.signOption)
    {
    case mFSO_Both:
    case mFSO_NegativeOrFill:
      signChars = 1;
      break;
    }

    mIF_CONSTEXPR (sizeof(value) == 1)
      goto one_byte_decimal;
    else mIF_CONSTEXPR (sizeof(value) == 2)
      goto two_bytes_decimal;
    else mIF_CONSTEXPR (sizeof(value) == 4)
      goto four_bytes_decimal;

    if (value >= 10000000000000000000) { numberChars = 20; break; }
    if (value >= 1000000000000000000) { numberChars = 19; break; }
    if (value >= 100000000000000000) { numberChars = 18; break; }
    if (value >= 10000000000000000) { numberChars = 17; break; }
    if (value >= 1000000000000000) { numberChars = 16; break; }
    if (value >= 100000000000000) { numberChars = 15; break; }
    if (value >= 10000000000000) { numberChars = 14; break; }
    if (value >= 1000000000000) { numberChars = 13; break; }
    if (value >= 100000000000) { numberChars = 12; break; }
    if (value >= 10000000000) { numberChars = 11; break; }

    goto four_bytes_decimal;
  four_bytes_decimal:

    if (value >= 1000000000) { numberChars = 10; break; }
    if (value >= 100000000) { numberChars = 9; break; }
    if (value >= 10000000) { numberChars = 8; break; }
    if (value >= 1000000) { numberChars = 7; break; }
    if (value >= 100000) { numberChars = 6; break; }

    goto two_bytes_decimal;
  two_bytes_decimal:

    if (value >= 10000) { numberChars = 5; break; }
    if (value >= 1000) { numberChars = 4; break; }

    goto one_byte_decimal;
  one_byte_decimal:

    if (value >= 100) { numberChars = 3; break; }
    if (value >= 10) { numberChars = 2; break; }

    numberChars = 1;
    break;
  }

  case mFBO_Hexadecimal:
  {
    numberChars = 0;
    T tmp = value;

    while (tmp)
    {
      numberChars++;
      tmp >>= 4;
    }

    return mClamp(numberChars, fs.minChars, fs.maxChars);
  }

  case mFBO_Binary:
  {
    numberChars = 0;
    T tmp = value;

    while (tmp)
    {
      numberChars++;
      tmp >>= 1;
    }

    return mClamp(numberChars, fs.minChars, fs.maxChars);
  }
  }

  if (fs.groupDigits)
    return mClamp(signChars + numberChars + _mFormat_GetDigitGroupingCharCount(numberChars, fs) * fs.digitGroupingCharLength, fs.minChars, fs.maxChars);
  else
    return mClamp(signChars + numberChars, fs.minChars, fs.maxChars);
}

template <typename T, typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value && !std::is_same<bool, T>::value>::type* = nullptr>
inline size_t mFormat_GetCount(const T &value, const mFormatState &fs)
{
  size_t signChars = 0;
  size_t numberChars = 0;

  switch (fs.integerBaseOption)
  {
  default:
  case mFBO_Decimal:
  {
    switch (fs.signOption)
    {
    case mFSO_Both:
    case mFSO_NegativeOrFill:
      signChars = 1;
      break;

    case mFSO_NegativeOnly:
      if (value < 0)
        signChars = 1;
      break;
    }

    const T negativeAbs = value < 0 ? value : -value; // because otherwise the minimum value couldn't be converted to a valid signed equivalent.

    mIF_CONSTEXPR (sizeof(value) == 1)
      goto one_byte_decimal;
    else mIF_CONSTEXPR (sizeof(value) == 2)
      goto two_bytes_decimal;
    else mIF_CONSTEXPR (sizeof(value) == 4)
      goto four_bytes_decimal;

    if (negativeAbs <= -1000000000000000000) { numberChars = 19; break; }
    if (negativeAbs <= -100000000000000000) { numberChars = 18; break; }
    if (negativeAbs <= -10000000000000000) { numberChars = 17; break; }
    if (negativeAbs <= -1000000000000000) { numberChars = 16; break; }
    if (negativeAbs <= -100000000000000) { numberChars = 15; break; }
    if (negativeAbs <= -10000000000000) { numberChars = 14; break; }
    if (negativeAbs <= -1000000000000) { numberChars = 13; break; }
    if (negativeAbs <= -100000000000) { numberChars = 12; break; }
    if (negativeAbs <= -10000000000) { numberChars = 11; break; }

    goto four_bytes_decimal;
  four_bytes_decimal:

    if (negativeAbs <= -1000000000) { numberChars = 10; break; }
    if (negativeAbs <= -100000000) { numberChars = 9; break; }
    if (negativeAbs <= -10000000) { numberChars = 8; break; }
    if (negativeAbs <= -1000000) { numberChars = 7; break; }
    if (negativeAbs <= -100000) { numberChars = 6; break; }

    goto two_bytes_decimal;
  two_bytes_decimal:

    if (negativeAbs <= -10000) { numberChars = 5; break; }
    if (negativeAbs <= -1000) { numberChars = 4; break; }

    goto one_byte_decimal;
  one_byte_decimal:

    if (negativeAbs <= -100) { numberChars = 3; break; }
    if (negativeAbs <= -10) { numberChars = 2; break; }

    numberChars = 1;
    break;
  }

  case mFBO_Hexadecimal:
  case mFBO_Binary:
  {
    return mFormat_GetCount((mUnsignedEquivalent<T>::type)value, fs);
  }
  }

  if (fs.groupDigits)
    return mClamp(signChars + numberChars + _mFormat_GetDigitGroupingCharCount(numberChars, fs) * fs.digitGroupingCharLength, fs.minChars, fs.maxChars);
  else
    return mClamp(signChars + numberChars, fs.minChars, fs.maxChars);
}

#pragma warning (pop)

inline size_t mFormat_GetMaxBytes(const float_t, const mFormatState &fs)
{
  constexpr size_t maxDigits = sizeof("340282346638528859811704183484516925440") - 1;

  if (fs.scientificNotation)
    return mClamp(1 + 1 + fs.decimalSeparatorLength + fs.fractionalDigits + 1 + 1 + 10, fs.minChars, mMax(fs.maxChars, 6ULL)); // sign + digit + decimalSeparator + decimalDigits + e + sign + exponent.
  else
    return mClamp(1 /* sign */ + maxDigits + _mFormat_GetDigitGroupingCharCount(maxDigits, fs) * fs.digitGroupingCharLength + fs.decimalSeparatorLength + fs.fractionalDigits, fs.minChars, fs.maxChars);
}

inline size_t mFormat_GetMaxBytes(const double_t, const mFormatState &fs)
{
  constexpr size_t maxDigits = sizeof("179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368") - 1;

  if (fs.scientificNotation)
    return mClamp(1 + 1 + fs.decimalSeparatorLength + fs.fractionalDigits + 1 + 1 + 10, fs.minChars, mMax(fs.maxChars, 7ULL)); // sign + digit + decimalSeparator + decimalDigits + e + sign + exponent.
  else
    return mClamp(1 /* sign */ + maxDigits + _mFormat_GetDigitGroupingCharCount(maxDigits, fs) * fs.digitGroupingCharLength + fs.decimalSeparatorLength + fs.fractionalDigits, fs.minChars, fs.maxChars);
}

template <typename T>
size_t _mFormat_AppendVector(const T *pFirstValue, const size_t count, const mFormatState &fs, char *text)
{
  size_t ret = 1;

  *text = fs.vectorStartChar;
  text++;

  if (fs.vectorSpaceAfterStart)
  {
    ret++;
    *text = ' ';
    text++;
  }

  for (size_t i = 0; i < count; i++)
  {
    if (i != 0)
    {
      *text = fs.vectorSeparatorChar;
      text++;
      ret++;

      if (fs.vectorSpaceAfterSeparator)
      {
        ret++;
        *text = ' ';
        text++;
      }
    }

    const size_t length = _mFormat_Append(*pFirstValue, fs, text);

    text += length;
    ret += length;
    pFirstValue++;
  }

  if (fs.vectorSpaceBeforeEnd)
  {
    ret++;
    *text = ' ';
    text++;
  }

  *text = fs.vectorEndChar;
  text++;
  ret++;

  return ret;
}

template <typename T>
inline size_t mFormat_GetMaxBytes(const mVec2t<T> &, const mFormatState &fs)
{
  constexpr size_t dimensions = 2;

  return 1 + (size_t)fs.vectorSpaceAfterStart + dimensions * mFormat_GetMaxBytes((T)0, fs) + (dimensions - 1) * ((size_t)fs.vectorSpaceAfterSeparator + 1) + (size_t)fs.vectorSpaceBeforeEnd + 1;
}

template <typename T>
inline size_t _mFormat_Append(const mVec2t<T> &value, const mFormatState &fs, char *text)
{
  return _mFormat_AppendVector(&value.x, 2, fs, text);
}

template <typename T>
inline size_t mFormat_GetMaxBytes(const mVec3t<T> &, const mFormatState &fs)
{
  constexpr size_t dimensions = 3;

  return 1 + (size_t)fs.vectorSpaceAfterStart + dimensions * mFormat_GetMaxBytes((T)0, fs) + (dimensions - 1) * ((size_t)fs.vectorSpaceAfterSeparator + 1) + (size_t)fs.vectorSpaceBeforeEnd + 1;
}

template <typename T>
inline size_t _mFormat_Append(const mVec3t<T> &value, const mFormatState &fs, char *text)
{
  return _mFormat_AppendVector(&value.x, 3, fs, text);
}

template <typename T>
inline size_t mFormat_GetMaxBytes(const mVec4t<T> &, const mFormatState &fs)
{
  constexpr size_t dimensions = 4;

  return 1 + (size_t)fs.vectorSpaceAfterStart + dimensions * mFormat_GetMaxBytes((T)0, fs) + (dimensions - 1) * ((size_t)fs.vectorSpaceAfterSeparator + 1) + (size_t)fs.vectorSpaceBeforeEnd + 1;
}

template <typename T>
inline size_t _mFormat_Append(const mVec4t<T> &value, const mFormatState &fs, char *text)
{
  return _mFormat_AppendVector(&value.x, 4, fs, text);
}

inline size_t mFormat_GetMaxBytes(const mVector &, const mFormatState &fs)
{
  constexpr size_t dimensions = 4;

  return 1 + (size_t)fs.vectorSpaceAfterStart + dimensions * mFormat_GetMaxBytes(0.f, fs) + (dimensions - 1) * ((size_t)fs.vectorSpaceAfterSeparator + 1) + (size_t)fs.vectorSpaceBeforeEnd + 1;
}

inline size_t _mFormat_Append(const mVector &value, const mFormatState &fs, char *text)
{
  return _mFormat_AppendVector(&value.x, 4, fs, text);
}

inline size_t _mFormat_Append(const char value, const mFormatState &fs, char *text)
{
  if (fs.maxChars == 0 || value == 0)
    return 0;

  *text = value;
  return 1;
}

size_t _mFormat_Append(const int64_t value, const mFormatState &fs, char *text);
size_t _mFormat_Append(const uint64_t value, const mFormatState &fs, char *text);
size_t _mFormat_Append(const float_t value, const mFormatState &fs, char *text);
size_t _mFormat_Append(const double_t value, const mFormatState &fs, char *text);
size_t _mFormat_Append(const wchar_t value, const mFormatState &fs, char *text);

size_t _mFormat_AppendWStringWithLength(const wchar_t *string, const size_t charCount, const mFormatState &fs, char *text);

template <typename T, typename std::enable_if<std::is_same<T, wchar_t *>::value>::type* = nullptr>
inline size_t _mFormat_Append(const T value, const mFormatState &fs, char *text)
{
  if (value == nullptr)
    return 0;

  return _mFormat_AppendWStringWithLength(value, wcslen(value), fs, text);
}

template <typename T, typename std::enable_if<std::is_same<T, const wchar_t *>::value>::type* = nullptr>
inline size_t _mFormat_Append(const T value, const mFormatState &fs, char *text)
{
  if (value == nullptr)
    return 0;

  return _mFormat_AppendWStringWithLength(value, wcslen(value), fs, text);
}

template <size_t TCount>
inline size_t _mFormat_Append(wchar_t(&value)[TCount], const mFormatState &fs, char *text)
{
  return _mFormat_AppendWStringWithLength(value, wcsnlen_s(value, TCount), fs, text);
}

template <size_t TCount>
inline size_t _mFormat_Append(const wchar_t(&value)[TCount], const mFormatState &fs, char *text)
{
  return _mFormat_AppendWStringWithLength(value, wcsnlen_s(value, TCount), fs, text);
}

size_t _mFormat_AppendBool(const bool value, const mFormatState &fs, char *text);

template <typename T, typename std::enable_if<std::is_same<T, bool>::value>::type* = nullptr>
inline size_t _mFormat_Append(const T value, const mFormatState &fs, char *text)
{
  return _mFormat_AppendBool(value, fs, text);
}

inline size_t _mFormat_Append(const int32_t value, const mFormatState &fs, char *text) { return _mFormat_Append((int64_t)value, fs, text); }
inline size_t _mFormat_Append(const int16_t value, const mFormatState &fs, char *text) { return _mFormat_Append((int64_t)value, fs, text); }
inline size_t _mFormat_Append(const int8_t value, const mFormatState &fs, char *text) { return _mFormat_Append((int64_t)value, fs, text); }
inline size_t _mFormat_Append(const uint32_t value, const mFormatState &fs, char *text) { return _mFormat_Append((uint64_t)value, fs, text); }
inline size_t _mFormat_Append(const uint16_t value, const mFormatState &fs, char *text) { return _mFormat_Append((uint64_t)value, fs, text); }
inline size_t _mFormat_Append(const uint8_t value, const mFormatState &fs, char *text) { return _mFormat_Append((uint64_t)value, fs, text); }

size_t _mFormat_AppendStringWithLength(const char *value, const size_t length, const mFormatState &fs, char *text);

template <size_t TCount>
inline size_t _mFormat_Append(char (&value)[TCount], const mFormatState &fs, char *text)
{
  if (TCount == 0)
    return 0;

  return _mFormat_AppendStringWithLength(value, strnlen(value, TCount - 1), fs, text);
}

template <size_t TCount>
inline size_t _mFormat_Append(const char(&value)[TCount], const mFormatState &fs, char *text)
{
  if (TCount == 0)
    return 0;

  return _mFormat_AppendStringWithLength(value, strnlen(value, TCount - 1), fs, text);
}

template <typename T, typename std::enable_if<std::is_same<T, char *>::value>::type* = nullptr>
inline size_t _mFormat_Append(const T value, const mFormatState &fs, char *text)
{
  if (value == nullptr)
    return 0;

  const size_t length = strlen(value);

  return _mFormat_AppendStringWithLength(value, length, fs, text);
}

template <typename T, typename std::enable_if<std::is_same<T, const char *>::value>::type* = nullptr>
inline size_t _mFormat_Append(const T value, const mFormatState &fs, char *text)
{
  if (value == nullptr)
    return 0;

  const size_t length = strlen(value);

  return _mFormat_AppendStringWithLength(value, length, fs, text);
}

size_t _mFormat_AppendMString(const mString &value, const mFormatState &fs, char *text);
size_t _mFormat_AppendInplaceString(const char *string, const size_t count, const size_t length, const mFormatState &fs, char *text);

template <typename T, typename std::enable_if<std::is_same<T, mString>::value>::type* = nullptr>
inline size_t _mFormat_Append(const T &value, const mFormatState &fs, char *text)
{
  return _mFormat_AppendMString(value, fs, text);
}

template <size_t TCount>
inline size_t _mFormat_Append(const mInplaceString<TCount> &value, const mFormatState &fs, char *text)
{
  return _mFormat_AppendInplaceString(value.c_str(), value.count, value.bytes, fs, text);
}

template <typename T>
size_t _mFormat_GetMaxBytes(const mFormatState &fs, const T &param)
{
  return mFormat_GetMaxBytes(param, fs);
}

template <typename T, typename... Args>
size_t _mFormat_GetMaxBytes(const mFormatState &fs, const T &param, Args && ... args)
{
  return mFormat_GetMaxBytes(param, fs) + _mFormat_GetMaxBytes(fs, args...);
}

template <typename T, typename... Args>
size_t _mFormat_Append_Internal(const mFormatState &fs, char *text, const T &param)
{
  return _mFormat_Append(param, fs, text);
}

template <typename T, typename... Args>
size_t _mFormat_Append_Internal(const mFormatState &fs, char *text, const T &param, Args && ... args)
{
  const size_t offset = _mFormat_Append(param, fs, text);

  text += offset;

  return offset + _mFormat_Append_Internal(fs, text, args...);
}

// mFormat is a general purpose replacement for sprintf() (to char *). The buffer that is returned will be overwritten whenever mFormat is called by the same thread. Please don't pass around the returned pointer. The returned pointer does not need to be `free`d.
template <typename... Args>
inline const char *mFormat(Args && ...args)
{
  mFormatState &fs = mFormat_GetState();

  if (fs.inFormatStatement)
  {
    mFAIL_DEBUG("Recursive mFormat is not supported.");
    return "";
  }

  fs.inFormatStatement = true;
  mDEFER_SINGLE_CAPTURE(fs, fs.inFormatStatement = false);

  fs.textPosition = 0;

  const size_t maxCapacityRequired = _mFormat_GetMaxBytes(fs, args...) + 1;

  if (fs.textCapacity < maxCapacityRequired)
  {
    const size_t nextCapacity = (maxCapacityRequired + (maxCapacityRequired - fs.textCapacity) * 2 + 1023) & ~(size_t)1023;

    if (mFAILED(mAllocator_Reallocate(fs.pAllocator, &fs.textStart, nextCapacity)))
      return "<ERROR: MEMORY_ALLOCATION_FAILURE>";

    fs.textCapacity = nextCapacity;
  }

  const size_t size = _mFormat_Append_Internal(fs, fs.textStart, args...);

  fs.textStart[size] = '\0';
  fs.textPosition = size + 1;

  return fs.textStart;
}

template <typename ...Args>
inline size_t mFormat_GetMaxRequiredBytes(Args && ...args)
{
  return _mFormat_GetMaxBytes(mFormat_GetState(), args...) + 1;
}

template <typename ...Args>
inline mResult mFormatTo(char *destination, const size_t capacity, Args && ...args)
{
  mFUNCTION_SETUP();

  mERROR_IF(destination == nullptr, mR_ArgumentNull);

  mFormatState &fs = mFormat_GetState();

  const size_t maxCapacityRequired = _mFormat_GetMaxBytes(fs, args...) + 1;
  const bool fitsInPlace = maxCapacityRequired <= capacity;

  if (fitsInPlace)
  {
    const size_t length = _mFormat_Append_Internal(fs, destination, args...);
    destination[length] = '\0';
  }
  else
  {
    const char *result = mFormat(args...);
    const size_t length = fs.textPosition;

    mERROR_IF(result == nullptr || length == 0, mR_InternalError);
    mERROR_IF(capacity < length, mR_ArgumentOutOfBounds);

    mMemcpy(destination, result, length);
  }

  mRETURN_SUCCESS();
}

template <typename T>
void _mFormat_ApplyFormat(mFormatState &fs)
{
  T::ApplyFormat(fs);
}

template <typename T, typename T2, typename ... Args>
void _mFormat_ApplyFormat(mFormatState &fs)
{
  T::ApplyFormat(fs);

  _mFormat_ApplyFormat<T2, Args...>(fs);
}

template <typename T, typename ... Args>
struct _mFormatType_Wrapper
{
  const T &value;

  _mFormatType_Wrapper(const T &value) : value(value) { }
};

template <typename T, typename ... Args>
struct _mFormatTypeInstance_Wrapper
{
  const T value;

  _mFormatTypeInstance_Wrapper(const T value) : value(value) { }
};

#define _M_FORMAT_DEFINE_SPECIALIZED_ALIAS(name, type) \
  template <typename ... Args> struct name : public _mFormatTypeInstance_Wrapper<type, Args...> \
  { inline name(const type v) : _mFormatTypeInstance_Wrapper<type, Args...>(v) {} }

_M_FORMAT_DEFINE_SPECIALIZED_ALIAS(mFInt, int64_t);
_M_FORMAT_DEFINE_SPECIALIZED_ALIAS(mFUInt, uint64_t);
_M_FORMAT_DEFINE_SPECIALIZED_ALIAS(mFFloat, float_t);
_M_FORMAT_DEFINE_SPECIALIZED_ALIAS(mFDouble, double_t);
_M_FORMAT_DEFINE_SPECIALIZED_ALIAS(mFBool, bool);

template <typename T, typename ... Args>
_mFormatType_Wrapper<T, Args...> mFString(const T &string, const Args && ...)
{
  return _mFormatType_Wrapper<T, Args...>(string);
}

template <typename T, typename ... Args>
_mFormatType_Wrapper<T, Args...> mFVector(const T &vector, const Args && ...)
{
  return _mFormatType_Wrapper<T, Args...>(vector);
}

template <typename T, typename ... Args>
size_t mFormat_GetMaxBytes(const _mFormatType_Wrapper<T, Args...> &value, const mFormatState &fs)
{
  mFormatState localFS(fs);

  _mFormat_ApplyFormat<Args...>(localFS);

  return mFormat_GetMaxBytes(value.value, localFS);
}

template <typename T, typename ... Args>
size_t mFormat_GetCount(const _mFormatType_Wrapper<T, Args...> &value, const mFormatState &fs)
{
  mFormatState localFS(fs);

  _mFormat_ApplyFormat<Args...>(localFS);

  return mFormat_GetCount(value.value, localFS);
}

template <typename T, typename ... Args>
size_t _mFormat_Append(const _mFormatType_Wrapper<T, Args...> &value, const mFormatState &fs, char *text)
{
  mFormatState localFS(fs);

  _mFormat_ApplyFormat<Args...>(localFS);

  return _mFormat_Append(value.value, localFS, text);
}

template <typename T, typename ... Args>
size_t mFormat_GetMaxBytes(const _mFormatTypeInstance_Wrapper<T, Args...> &value, const mFormatState &fs)
{
  mFormatState localFS(fs);

  _mFormat_ApplyFormat<Args...>(localFS);

  return mFormat_GetMaxBytes(value.value, localFS);
}

template <typename T, typename ... Args>
size_t mFormat_GetCount(const _mFormatTypeInstance_Wrapper<T, Args...> &value, const mFormatState &fs)
{
  mFormatState localFS(fs);

  _mFormat_ApplyFormat<Args...>(localFS);

  return mFormat_GetCount(value.value, localFS);
}

template <typename T, typename ... Args>
size_t _mFormat_Append(const _mFormatTypeInstance_Wrapper<T, Args...> &value, const mFormatState &fs, char *text)
{
  mFormatState localFS(fs);

  _mFormat_ApplyFormat<Args...>(localFS);

  return _mFormat_Append(value.value, localFS, text);
}

template <size_t maxDigits>
struct mFMaxDigits
{
  static void ApplyFormat(mFormatState &fs) { fs.maxChars = maxDigits; }
};

#define mFMaxChars mFMaxDigits

template <size_t minDigits>
struct mFMinDigits
{
  static void ApplyFormat(mFormatState &fs) { fs.minChars = minDigits; }
};

#define mFMinChars mFMinDigits

template <size_t fractionalDigits>
struct mFFractionalDigits
{
  static void ApplyFormat(mFormatState &fs) { fs.fractionalDigits = fractionalDigits; }
};

struct mFHex
{
  static void ApplyFormat(mFormatState &fs) { fs.integerBaseOption = mFBO_Hexadecimal; }
};

struct mFHexUppercase
{
  static void ApplyFormat(mFormatState &fs) { fs.hexadecimalUpperCase = true; }
};

struct mFHexLowercase
{
  static void ApplyFormat(mFormatState &fs) { fs.hexadecimalUpperCase = false; }
};

struct mFBinary
{
  static void ApplyFormat(mFormatState &fs) { fs.integerBaseOption = mFBO_Binary; }
};

struct mFDecimal
{
  static void ApplyFormat(mFormatState &fs) { fs.integerBaseOption = mFBO_Decimal; }
};

struct mFAlignNumRight
{
  static void ApplyFormat(mFormatState &fs) { fs.numberAlign = mFA_Right; }
};

struct mFAlignNumLeft
{
  static void ApplyFormat(mFormatState &fs) { fs.numberAlign = mFA_Left; }
};

struct mFAlignNumCenter
{
  static void ApplyFormat(mFormatState &fs) { fs.numberAlign = mFA_Center; }
};

struct mFAlignStringRight
{
  static void ApplyFormat(mFormatState &fs) { fs.stringAlign = mFA_Right; }
};

struct mFAlignStringLeft
{
  static void ApplyFormat(mFormatState &fs) { fs.stringAlign = mFA_Left; }
};

struct mFAlignStringCenter
{
  static void ApplyFormat(mFormatState &fs) { fs.stringAlign = mFA_Center; }
};

struct mFSignBoth
{
  static void ApplyFormat(mFormatState &fs) { fs.signOption = mFSO_Both; }
};

struct mFSignNever
{
  static void ApplyFormat(mFormatState &fs) { fs.signOption = mFSO_Never; }
};

struct mFSignNegativeOrFill
{
  static void ApplyFormat(mFormatState &fs) { fs.signOption = mFSO_NegativeOrFill; }
};

struct mFSignNegativeOnly
{
  static void ApplyFormat(mFormatState &fs) { fs.signOption = mFSO_NegativeOnly; }
};

struct mFSignAligned
{
  static void ApplyFormat(mFormatState &fs) { fs.alignSign = true; }
};

struct mFSignNotAligned
{
  static void ApplyFormat(mFormatState &fs) { fs.alignSign = false; }
};

struct mFFillZeroes
{
  static void ApplyFormat(mFormatState &fs) { fs.fillCharacter = '0'; fs.fillCharacterIsZero = true; }
};

struct mFFillWhitespace
{
  static void ApplyFormat(mFormatState &fs) { fs.fillCharacter = ' '; fs.fillCharacterIsZero = false; }
};

struct mFEllipsis
{
  static void ApplyFormat(mFormatState &fs) { fs.stringOverflowEllipsis = true; }
};

struct mFNoEllipsis
{
  static void ApplyFormat(mFormatState &fs) { fs.stringOverflowEllipsis = false; }
};

struct mFGroupDigits
{
  static void ApplyFormat(mFormatState &fs) { fs.groupDigits = true; }
};

struct mFNoExponent
{
  static void ApplyFormat(mFormatState &fs) { fs.scientificNotation = false; }
};

struct mFExponent
{
  static void ApplyFormat(mFormatState &fs) { fs.scientificNotation = true; }
};

struct mFExponentAdaptive
{
  static void ApplyFormat(mFormatState &fs) { fs.adaptiveFloatScientificNotation = true; }
};

struct mFExponentNotAdaptive
{
  static void ApplyFormat(mFormatState &fs) { fs.adaptiveFloatScientificNotation = false; }
};

struct mFFractionalDigitsAdaptive
{
  static void ApplyFormat(mFormatState &fs) { fs.adaptiveFractionalDigits = true; }
};

struct mFFractionalDigitsFixed
{
  static void ApplyFormat(mFormatState &fs) { fs.adaptiveFractionalDigits = false; }
};

#define _mFORMAT_EXPAND(x) x

#ifndef _MSC_VER

#define _mFORMAT_GET_NTH_ARG(_1, _2, _3, _4, _5, _6, _7, _8, _9, N, ...) N
#define _mFORMAT_FOR_EACH_0(_call, ...)
#define _mFORMAT_FOR_EACH_1(_call, x) _call(x)
#define _mFORMAT_FOR_EACH_2(_call, x, ...) _call(x) , _mFORMAT_FOR_EACH_1(_call, __VA_ARGS__)
#define _mFORMAT_FOR_EACH_3(_call, x, ...) _call(x) , _mFORMAT_FOR_EACH_2(_call, __VA_ARGS__)
#define _mFORMAT_FOR_EACH_4(_call, x, ...) _call(x) , _mFORMAT_FOR_EACH_3(_call, __VA_ARGS__)
#define _mFORMAT_FOR_EACH_5(_call, x, ...) _call(x) , _mFORMAT_FOR_EACH_4(_call, __VA_ARGS__)
#define _mFORMAT_FOR_EACH_6(_call, x, ...) _call(x) , _mFORMAT_FOR_EACH_5(_call, __VA_ARGS__)
#define _mFORMAT_FOR_EACH_7(_call, x, ...) _call(x) , _mFORMAT_FOR_EACH_6(_call, __VA_ARGS__)
#define _mFORMAT_FOR_EACH_8(_call, x, ...) _call(x) , _mFORMAT_FOR_EACH_7(_call, __VA_ARGS__)

#define mFORMAT_MACRO_COMMA_FOR_EACH(x, ...) _mFORMAT_GET_NTH_ARG(IGNORED, ##__VA_ARGS__, _mFORMAT_FOR_EACH_8, _mFORMAT_FOR_EACH_7, _mFORMAT_FOR_EACH_6, _mFORMAT_FOR_EACH_5, _mFORMAT_FOR_EACH_4, _mFORMAT_FOR_EACH_3, _mFORMAT_FOR_EACH_2, _mFORMAT_FOR_EACH_1, _mFORMAT_FOR_EACH_0)(x, ##__VA_ARGS__)

#define _mFORMAT_ARG_COUNT(...) _mFORMAT_ARG_COUNT_INTERNAL(0, ## __VA_ARGS__, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
#define _mFORMAT_ARG_COUNT_INTERNAL(_0, _1_, _2_, _3_, _4_, _5_, _6_, _7_, _8_, _9_, _10_, _11_, _12_, _13_, _14_, _15_, _16_, _17_, _18_, _19_, _20_, _21_, _22_, _23_, _24_, _25_, _26_, _27_, _28_, _29_, _30_, _31_, _32_, _33_, _34_, _35_, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54, _55, _56, _57, _58, _59, _60, _61, _62, _63, _64, _65, _66, _67, _68, _69, _70, count, ...) count

#else

#define _mFORMAT_FOR_EACH_1(what, x, ...) what(x)
#define _mFORMAT_FOR_EACH_2(what, x, ...)\
  what(x),\
  _mFORMAT_EXPAND(_mFORMAT_FOR_EACH_1(what,  __VA_ARGS__))
#define _mFORMAT_FOR_EACH_3(what, x, ...)\
  what(x),\
  _mFORMAT_EXPAND(_mFORMAT_FOR_EACH_2(what, __VA_ARGS__))
#define _mFORMAT_FOR_EACH_4(what, x, ...)\
  what(x),\
  _mFORMAT_EXPAND(_mFORMAT_FOR_EACH_3(what,  __VA_ARGS__))
#define _mFORMAT_FOR_EACH_5(what, x, ...)\
  what(x),\
  _mFORMAT_EXPAND(_mFORMAT_FOR_EACH_4(what,  __VA_ARGS__))
#define _mFORMAT_FOR_EACH_6(what, x, ...)\
  what(x),\
  _mFORMAT_EXPAND(_mFORMAT_FOR_EACH_5(what,  __VA_ARGS__))
#define _mFORMAT_FOR_EACH_7(what, x, ...)\
  what(x),\
  _mFORMAT_EXPAND(_mFORMAT_FOR_EACH_6(what,  __VA_ARGS__))
#define _mFORMAT_FOR_EACH_8(what, x, ...)\
  what(x),\
  _mFORMAT_EXPAND(_mFORMAT_FOR_EACH_7(what,  __VA_ARGS__))
#define _mFORMAT_FOR_EACH_NARG(...) _mFORMAT_FOR_EACH_NARG_(__VA_ARGS__, _mFORMAT_FOR_EACH_RSEQ_N())
#define _mFORMAT_FOR_EACH_NARG_(...) _mFORMAT_EXPAND(_mFORMAT_FOR_EACH_ARG_N(__VA_ARGS__))
#define _mFORMAT_FOR_EACH_ARG_N(_1, _2, _3, _4, _5, _6, _7, _8, N, ...) N
#define _mFORMAT_FOR_EACH_RSEQ_N() 8, 7, 6, 5, 4, 3, 2, 1, 0
#define _mFORMAT_FOR_EACH_(N, what, ...) _mFORMAT_EXPAND(mCONCAT_LITERALS_INTERNAL(_mFORMAT_FOR_EACH_, N)(what, __VA_ARGS__))
#define mFORMAT_MACRO_COMMA_FOR_EACH(what, ...) _mFORMAT_FOR_EACH_(_mFORMAT_FOR_EACH_NARG(__VA_ARGS__), what, __VA_ARGS__)

#define _mFORMAT_ARG_COUNT(...)  _mFORMAT_ARG_COUNT_INTERNAL_EXPAND_ARGS_PRIVATE(_mFORMAT_ARG_COUNT_INTERNAL_ARGS_AUGMENTER(__VA_ARGS__))

#define _mFORMAT_ARG_COUNT_INTERNAL_ARGS_AUGMENTER(...) unused, __VA_ARGS__
#define _mFORMAT_ARG_COUNT_INTERNAL_EXPAND(x) x
#define _mFORMAT_ARG_COUNT_INTERNAL_EXPAND_ARGS_PRIVATE(...) _mFORMAT_ARG_COUNT_INTERNAL_EXPAND(_mFORMAT_ARG_COUNT_INTERNAL_GET_ARG_COUNT_PRIVATE(__VA_ARGS__, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0))
#define _mFORMAT_ARG_COUNT_INTERNAL_GET_ARG_COUNT_PRIVATE(_1_, _2_, _3_, _4_, _5_, _6_, _7_, _8_, _9_, _10_, _11_, _12_, _13_, _14_, _15_, _16_, _17_, _18_, _19_, _20_, _21_, _22_, _23_, _24_, _25_, _26_, _27_, _28_, _29_, _30_, _31_, _32_, _33_, _34_, _35_, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54, _55, _56, _57, _58, _59, _60, _61, _62, _63, _64, _65, _66, _67, _68, _69, _70, count, ...) count

#endif

#define _mFORMAT_SHORT_Hex mFHex
#define _mFORMAT_SHORT_x mFHexLowercase
#define _mFORMAT_SHORT_X mFHexUppercase
#define _mFORMAT_SHORT_Bin mFBinary
#define _mFORMAT_SHORT_Frac(k) mFFractionalDigits< k >
#define _mFORMAT_SHORT_Min(k) mFMinDigits< k >
#define _mFORMAT_SHORT_Max(k) mFMaxDigits< k >
#define _mFORMAT_SHORT_Fill0 mFFillZeroes
#define _mFORMAT_SHORT_AllFrac mFFractionalDigitsFixed
#define _mFORMAT_SHORT_Exp mFExponent
#define _mFORMAT_SHORT_SBoth mFSignBoth
#define _mFORMAT_SHORT_Center mFAlignCenter
#define _mFORMAT_SHORT_Left mFAlignLeft
#define _mFORMAT_SHORT_Right mFAlignRight
#define _mFORMAT_SHORT_Group mFGroupDigits

#define _mFORMAT_TO_SHORT_FORMAT(x) mCONCAT_LITERALS(_mFORMAT_SHORT_, x)

#define _mFORMAT_XX_UNRAVEL(...) mFORMAT_MACRO_COMMA_FOR_EACH(_mFORMAT_TO_SHORT_FORMAT, __VA_ARGS__)

#define mFX_COMMA_OR_EMPTY_0
#define mFX_COMMA_OR_EMPTY_1 ,
#define mFX_COMMA_OR_EMPTY_2 ,
#define mFX_COMMA_OR_EMPTY_3 ,
#define mFX_COMMA_OR_EMPTY_4 ,
#define mFX_COMMA_OR_EMPTY_5 ,
#define mFX_COMMA_OR_EMPTY_6 ,
#define mFX_COMMA_OR_EMPTY_7 ,
#define mFX_COMMA_OR_EMPTY_8 ,
#define mFX_COMMA_OR_EMPTY_9 ,

#define mFI(...) mFInt< _mFORMAT_XX_UNRAVEL(__VA_ARGS__) >
#define mFU(...) mFUInt< _mFORMAT_XX_UNRAVEL(__VA_ARGS__) >
#define mFF(...) mFFloat< _mFORMAT_XX_UNRAVEL(__VA_ARGS__) >
#define mFD(...) mFDouble< _mFORMAT_XX_UNRAVEL(__VA_ARGS__) >
#define mFX(...) mFUInt<_mFORMAT_XX_UNRAVEL(Hex mCONCAT_LITERALS(mFX_COMMA_OR_EMPTY_, _mFORMAT_ARG_COUNT(__VA_ARGS__)) __VA_ARGS__) >

#endif // mFormat_h__
