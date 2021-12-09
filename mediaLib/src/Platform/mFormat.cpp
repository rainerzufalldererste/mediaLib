#include "dragonbox/dragonbox.h"
#include "dragonbox/dragonbox_to_chars.h"
#include "dragonbox/dragonbox_to_chars.cpp"

#include "mFormat.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "zIP7a6/ohlaNnadYcj56rd9FqUa68fs2K7o1TVIv1/8fFTsZGKDizjsFVr8Bs5D5AbH3ffMBhEyimt4a"
#endif

static mFormatState mFormat_GlobalState;
thread_local mFormatState mFormat_LocalState = mFormat_GlobalState;

mFormatState & mFormat_GetState()
{
  return mFormat_LocalState;
}

mFormatState & mFormat_GetGlobalState()
{
  return mFormat_GlobalState;
}

void mFormatState_ResetCulture()
{
  mFormat_GlobalState.decimalSeparatorLength = mFormat_LocalState.decimalSeparatorLength = 1;
  mFormat_GlobalState.decimalSeparatorChars[0] = mFormat_LocalState.decimalSeparatorChars[0] = '.';
  mFormat_GlobalState.digitGroupingCharLength = mFormat_LocalState.digitGroupingCharLength = 1;
  mFormat_GlobalState.digitGroupingChars[0] = mFormat_LocalState.digitGroupingChars[0] = ',';
  mFormat_GlobalState.digitGroupingOption = mFormat_LocalState.digitGroupingOption = mFDGO_Thousand;
  
  mFormat_GlobalState.nanChars[0] = 'N';
  mFormat_GlobalState.nanChars[1] = 'a';
  mFormat_GlobalState.nanChars[2] = 'N';
  mFormat_GlobalState.nanBytes = mFormat_GlobalState.nanCount = 3;

  mFormat_GlobalState.infinityChars[0] = 'I';
  mFormat_GlobalState.infinityChars[1] = 'n';
  mFormat_GlobalState.infinityChars[2] = 'f';
  mFormat_GlobalState.infinityChars[3] = 'i';
  mFormat_GlobalState.infinityChars[4] = 'n';
  mFormat_GlobalState.infinityChars[5] = 'i';
  mFormat_GlobalState.infinityChars[6] = 't';
  mFormat_GlobalState.infinityChars[7] = 'y';
  mFormat_GlobalState.infinityBytes = mFormat_GlobalState.infinityCount = 8;
  
  mFormat_GlobalState.negativeInfinityChars[0] = '-';
  mFormat_GlobalState.negativeInfinityChars[1] = 'I';
  mFormat_GlobalState.negativeInfinityChars[2] = 'n';
  mFormat_GlobalState.negativeInfinityChars[3] = 'f';
  mFormat_GlobalState.negativeInfinityChars[4] = 'i';
  mFormat_GlobalState.negativeInfinityChars[5] = 'n';
  mFormat_GlobalState.negativeInfinityChars[6] = 'i';
  mFormat_GlobalState.negativeInfinityChars[7] = 't';
  mFormat_GlobalState.negativeInfinityChars[8] = 'y';
  mFormat_GlobalState.negativeInfinityBytes = mFormat_GlobalState.negativeInfinityCount = 9;

  mFormat_LocalState.SetTo(mFormat_GlobalState);
}

size_t _mFormat_GetStringCount(const char *value, const size_t length)
{
  size_t count, size;

  if (mSUCCEEDED(mInplaceString_GetCount_Internal(value, length, &count, &size)))
    return count;

  return 0;
}

static const class _mFormat_LocaleSetter
{
public:
  _mFormat_LocaleSetter()
  {
    new (&mFormat_GlobalState) mFormatState();

    wchar_t wbuffer[128];
    int32_t size = 0;

    size = GetLocaleInfoEx(LOCALE_NAME_USER_DEFAULT, LOCALE_SDECIMAL, wbuffer, (int32_t)mARRAYSIZE(wbuffer));
    mFormat_GlobalState.decimalSeparatorLength = WideCharToMultiByte(CP_UTF8, 0, wbuffer, size, mFormat_GlobalState.decimalSeparatorChars, (int32_t)mARRAYSIZE(mFormat_GlobalState.decimalSeparatorChars), nullptr, false);
    mFormat_GlobalState.decimalSeparatorLength -= (size_t)!!mFormat_GlobalState.decimalSeparatorLength;

    size = GetLocaleInfoEx(LOCALE_NAME_USER_DEFAULT, LOCALE_STHOUSAND, wbuffer, (int32_t)mARRAYSIZE(wbuffer));
    mFormat_GlobalState.digitGroupingCharLength = WideCharToMultiByte(CP_UTF8, 0, wbuffer, size, mFormat_GlobalState.digitGroupingChars, (int32_t)mARRAYSIZE(mFormat_GlobalState.digitGroupingChars), nullptr, false);
    mFormat_GlobalState.digitGroupingCharLength -= (size_t)!!mFormat_GlobalState.digitGroupingCharLength;

    char buffer[32];
    size = GetLocaleInfoA(LOCALE_NAME_USER_DEFAULT, LOCALE_SGROUPING, buffer, (int32_t)mARRAYSIZE(buffer));

    if (buffer[0] == '4')
      mFormat_GlobalState.digitGroupingOption = mFDGO_TenThousand;
    else if (size > 2 && buffer[2] == '2')
      mFormat_GlobalState.digitGroupingOption = mFDGO_Indian;
    else
      mFormat_GlobalState.digitGroupingOption = mFDGO_Thousand;

    size = GetLocaleInfoEx(LOCALE_NAME_USER_DEFAULT, LOCALE_SNEGINFINITY, wbuffer, (int32_t)mARRAYSIZE(wbuffer));
    mFormat_GlobalState.negativeInfinityBytes = WideCharToMultiByte(CP_UTF8, 0, wbuffer, size, mFormat_GlobalState.negativeInfinityChars, (int32_t)mARRAYSIZE(mFormat_GlobalState.negativeInfinityChars), nullptr, false);
    mFormat_GlobalState.negativeInfinityCount = _mFormat_GetStringCount(mFormat_GlobalState.negativeInfinityChars, mFormat_GlobalState.negativeInfinityBytes);
    mFormat_GlobalState.negativeInfinityBytes -= (size_t)!!mFormat_GlobalState.negativeInfinityBytes;

    size = GetLocaleInfoEx(LOCALE_NAME_USER_DEFAULT, LOCALE_SPOSINFINITY, wbuffer, (int32_t)mARRAYSIZE(wbuffer));
    mFormat_GlobalState.infinityBytes = WideCharToMultiByte(CP_UTF8, 0, wbuffer, size, mFormat_GlobalState.infinityChars, (int32_t)mARRAYSIZE(mFormat_GlobalState.infinityChars), nullptr, false);
    mFormat_GlobalState.infinityCount = _mFormat_GetStringCount(mFormat_GlobalState.infinityChars, mFormat_GlobalState.infinityBytes);
    mFormat_GlobalState.infinityBytes -= (size_t)!!mFormat_GlobalState.infinityBytes;

    size = GetLocaleInfoEx(LOCALE_NAME_USER_DEFAULT, LOCALE_SNAN, wbuffer, (int32_t)mARRAYSIZE(wbuffer));
    mFormat_GlobalState.nanBytes = WideCharToMultiByte(CP_UTF8, 0, wbuffer, size, mFormat_GlobalState.nanChars, (int32_t)mARRAYSIZE(mFormat_GlobalState.nanChars), nullptr, false);
    mFormat_GlobalState.nanCount = _mFormat_GetStringCount(mFormat_GlobalState.nanChars, mFormat_GlobalState.nanBytes);
    mFormat_GlobalState.nanBytes -= (size_t)!!mFormat_GlobalState.nanBytes;
  }

} _mFormat_SetLocale;

//////////////////////////////////////////////////////////////////////////

size_t _mFormat_Append_Decimal(const bool negative, const char signChar, const size_t signChars, const size_t digits, const char *buffer, const mFormatState &fs, char *text);
void _mFormat_Append_DecimalDigitsWithGroupDigitsNoSign_Internal(const size_t numberBytes, char *text, const char *buffer, const mFormatState &fs);
void _mFormat_Append_DisplayWithAlignNoGroupingWithSign_Internal(const size_t totalBytes, const size_t signChars, const char signChar, const size_t numberBytes, char *text, const char *buffer, const mFormatState &fs);
size_t _mFormat_Append_DisplayWithAlign_Internal(const size_t totalBytes, const size_t maxChars, char *text, const char *buffer, const mFormatState &fs, const bool isNumber);
size_t _mFormat_Append_DisplayWithAlign_Internal(const size_t totalBytes, char *text, const char *buffer, const mFormatState &fs);
size_t _mFormat_Append_DecimalFloat(const bool negative, const char signChar, const size_t signChars, const size_t digits, char *buffer, const int64_t exponent, const mFormatState &fs, char *text);
size_t _mFormat_Append_DecimalFloatScientific(const bool negative, const char signChar, const size_t signChars, const size_t digits, const char *buffer, const size_t decimalSeparatorPosition, const char *exponentBuffer, const size_t exponentLength, const size_t fractionalDigits, const mFormatState &fs, char *text);

//////////////////////////////////////////////////////////////////////////

static constexpr char _mFormat_DecimalLUT[] = 
{
  '0', '0', '0', '1', '0', '2', '0', '3', '0', '4',
  '0', '5', '0', '6', '0', '7', '0', '8', '0', '9',
  '1', '0', '1', '1', '1', '2', '1', '3', '1', '4',
  '1', '5', '1', '6', '1', '7', '1', '8', '1', '9',
  '2', '0', '2', '1', '2', '2', '2', '3', '2', '4',
  '2', '5', '2', '6', '2', '7', '2', '8', '2', '9',
  '3', '0', '3', '1', '3', '2', '3', '3', '3', '4',
  '3', '5', '3', '6', '3', '7', '3', '8', '3', '9',
  '4', '0', '4', '1', '4', '2', '4', '3', '4', '4',
  '4', '5', '4', '6', '4', '7', '4', '8', '4', '9',
  '5', '0', '5', '1', '5', '2', '5', '3', '5', '4',
  '5', '5', '5', '6', '5', '7', '5', '8', '5', '9',
  '6', '0', '6', '1', '6', '2', '6', '3', '6', '4',
  '6', '5', '6', '6', '6', '7', '6', '8', '6', '9',
  '7', '0', '7', '1', '7', '2', '7', '3', '7', '4',
  '7', '5', '7', '6', '7', '7', '7', '8', '7', '9',
  '8', '0', '8', '1', '8', '2', '8', '3', '8', '4',
  '8', '5', '8', '6', '8', '7', '8', '8', '8', '9',
  '9', '0', '9', '1', '9', '2', '9', '3', '9', '4',
  '9', '5', '9', '6', '9', '7', '9', '8', '9', '9'
};

size_t _mFormat_Append(const int64_t value, const mFormatState &fs, char *text)
{
  switch (fs.integerBaseOption)
  {
  default:
  case mFBO_Decimal:
  {
    size_t signChars = 0;
    size_t numberBytes = 1;
    char signChar = '-';

    switch (fs.signOption)
    {
    case mFSO_Both:
    {
      signChars = 1;

      if (value >= 0)
        signChar = '+';

      break;
    }

    case mFSO_NegativeOrFill:
    {
      signChars = 1;

      if (value >= 0)
      {
        if (fs.fillCharacterIsZero)
          signChar = ' ';
        else
          signChar = fs.fillCharacter;
      }

      break;
    }

    case mFSO_NegativeOnly:
    {
      if (value < 0)
        signChars = 1;

      break;
    }
    }

    char buffer[19];
    char *pBuffer = &buffer[mARRAYSIZE(buffer) - 1];

    int64_t negativeAbs = value < 0 ? value : -value; // because otherwise the minimum value couldn't be converted to a valid signed equivalent.

    *pBuffer = (char)('0' + -(negativeAbs % 10));
    pBuffer--;

    negativeAbs /= -10;

    if (negativeAbs != 0)
    {
      uint64_t tmp = (uint64_t)negativeAbs;

      while (tmp >= 100)
      {
        memcpy(pBuffer - 1, _mFormat_DecimalLUT + (tmp % 100) * 2, 2);
        pBuffer -= 2;
        tmp /= 100;
        numberBytes += 2;
      }

      if (tmp >= 10)
      {
        *pBuffer = (char)('0' + (tmp % 10));
        pBuffer--;
        tmp /= 10;
        numberBytes++;
      }

      if (tmp != 0)
      {
        *pBuffer = (char)('0' + tmp);
        numberBytes++;
      }
      else
      {
        pBuffer++;
      }
    }
    else
    {
      pBuffer++;
    }

    return _mFormat_Append_Decimal(value < 0, signChar, signChars, numberBytes, pBuffer, fs, text);
  }

  case mFBO_Hexadecimal:
  case mFBO_Binary:
  {
    return _mFormat_Append((size_t)value, fs, text);
  }
  }
}

size_t _mFormat_Append(const uint64_t value, const mFormatState &fs, char *text)
{
  switch (fs.integerBaseOption)
  {
  default:
  case mFBO_Decimal:
  {
    size_t signChars = 0;
    size_t numberBytes = 0;
    char signChar = '+';

    switch (fs.signOption)
    {
    case mFSO_Both:
      signChars = 1;
      break;

    case mFSO_NegativeOrFill:
      signChars = 1;

      if (fs.fillCharacterIsZero)
        signChar = ' ';
      else
        signChar = fs.fillCharacter;

      break;
    }

    char buffer[20];
    char *pBuffer = &buffer[mARRAYSIZE(buffer) - 1];
    uint64_t tmp = value;
    
    while (tmp >= 100)
    {
      memcpy(pBuffer - 1, _mFormat_DecimalLUT + (tmp % 100) * 2, 2);
      pBuffer -= 2;
      tmp /= 100;
      numberBytes += 2;
    }

    if (tmp >= 10)
    {
      *pBuffer = (char)('0' + (tmp % 10));
      pBuffer--;
      tmp /= 10;
      numberBytes++;
    }

    if (tmp != 0 || numberBytes == 0)
    {
      *pBuffer = (char)('0' + tmp);
      numberBytes++;
    }
    else
    {
      pBuffer++;
    }

    return _mFormat_Append_Decimal(false, signChar, signChars, numberBytes, pBuffer, fs, text);
  }

  case mFBO_Hexadecimal:
  {
    char buffer[16];
    char *pBuffer = &buffer[mARRAYSIZE(buffer) - 1];
    const size_t lowerCaseCorrectionValue = (!fs.hexadecimalUpperCase) * ('a' - 'A') - 0xA;
    uint64_t tmp = value;
    size_t numberBytes = 0;

    while (tmp >= 0x10)
    {
      const size_t digit = (tmp & 0xF);

      *pBuffer = (char)(digit <= 9 ? ('0' + digit) : ('A' + digit + lowerCaseCorrectionValue));
      pBuffer--;
      tmp >>= 4;
      numberBytes++;
    }

    if (tmp != 0 || numberBytes == 0)
    {
      *pBuffer = (char)(tmp <= 9 ? ('0' + tmp) : ('A' + tmp + lowerCaseCorrectionValue));
      numberBytes++;
    }
    else
    {
      pBuffer++;
    }

    return _mFormat_Append_DisplayWithAlign_Internal(numberBytes, numberBytes, text, pBuffer, fs, true);
  }

  case mFBO_Binary:
  {
    char buffer[64];
    char *pBuffer = &buffer[mARRAYSIZE(buffer) - 1];

    uint64_t tmp = value;
    size_t numberBytes = 0;

    while (tmp >= 0b10)
    {
      *pBuffer = (char)('0' + (tmp & 0b1));
      pBuffer--;
      tmp >>= 1;
      numberBytes++;
    }

    if (tmp != 0 || numberBytes == 0)
    {
      *pBuffer = (char)('0' + tmp);
      numberBytes++;
    }
    else
    {
      pBuffer++;
    }

    return _mFormat_Append_DisplayWithAlign_Internal(numberBytes, numberBytes, text, pBuffer, fs, true);
  }
  }
}

size_t _mFormat_AppendStringWithLength(const char *value, const size_t length, const mFormatState &fs, char *text)
{
  return _mFormat_Append_DisplayWithAlign_Internal(length, text, value, fs);
}

size_t _mFormat_AppendMString(const mString &value, const mFormatState &fs, char *text)
{
  if (value.bytes <= 1)
    return 0;

  return _mFormat_Append_DisplayWithAlign_Internal(value.bytes - 1, value.count - 1, text, value.c_str(), fs, false);
}

size_t _mFormat_AppendInplaceString(const char *string, const size_t count, const size_t length, const mFormatState &fs, char *text)
{
  if (length <= 1 || string == nullptr)
    return 0;

  return _mFormat_Append_DisplayWithAlign_Internal(length - 1, count - 1, text, string, fs, false);
}

size_t _mFormat_HandleNonzeroFloat(const bool isNegative, const uint64_t significand, const int64_t exponent, const mFormatState &fs, char *text)
{
  size_t signChars = 0;
  size_t numberBytes = 0;
  char signChar = '-';

  switch (fs.signOption)
  {
  case mFSO_Both:
  {
    signChars = 1;

    if (!isNegative)
      signChar = '+';

    break;
  }

  case mFSO_NegativeOrFill:
  {
    signChars = 1;

    if (!isNegative)
    {
      if (fs.fillCharacterIsZero)
        signChar = ' ';
      else
        signChar = fs.fillCharacter;
    }

    break;
  }

  case mFSO_NegativeOnly:
  {
    if (isNegative)
      signChars = 1;

    break;
  }
  }

  char buffer[21];
  char *pBuffer = &buffer[mARRAYSIZE(buffer) - 1];

  // Serialize significand.
  {
    uint64_t tmp = significand;

    while (tmp >= 100)
    {
      memcpy(pBuffer - 1, _mFormat_DecimalLUT + (tmp % 100) * 2, 2);
      pBuffer -= 2;
      tmp /= 100;
      numberBytes += 2;
    }

    if (tmp >= 10)
    {
      *pBuffer = (char)('0' + (tmp % 10));
      pBuffer--;
      tmp /= 10;
      numberBytes++;
    }

    if (tmp != 0 || numberBytes == 0)
    {
      *pBuffer = (char)('0' + tmp);
      numberBytes++;
    }
    else
    {
      pBuffer++;
    }
  }

  bool scientificNotation = fs.scientificNotation;

  if (!scientificNotation && fs.adaptiveFloatScientificNotation)
  {
    const int64_t approxExponent = (int64_t)numberBytes + exponent; // may not be accurate, because rounding might add another digit.

    scientificNotation = approxExponent >= fs.adaptiveScientificNotationPositiveExponentThreshold || approxExponent <= fs.adaptiveScientificNotationNegativeExponentThreshold;
  }

  if (!scientificNotation)
  {
    return _mFormat_Append_DecimalFloat(isNegative, signChar, signChars, numberBytes, pBuffer, exponent, fs, text);
  }
  else
  {
    int64_t maxDecimalDigits = fs.maxChars - signChars - 1 - 1 - 2;
    const size_t absTmpExponent = mAbs(exponent + (int64_t)numberBytes - 1); // This may be inaccurate in case rounding introduces another digit.

    if (absTmpExponent < 9) // In case rounding introduces another digit this is < 9, not <= 9
      maxDecimalDigits -= 1;
    else if (absTmpExponent < 99) // same here.
      maxDecimalDigits -= 2;
    else
      maxDecimalDigits -= 3; // the maximum double exponent is 308/-308.

    const size_t fractionalDigits = mMin(fs.fractionalDigits, (size_t)mMax(maxDecimalDigits, 0LL));

    char *pRoundChar = pBuffer + fractionalDigits;
    const bool round = numberBytes - 1 > fractionalDigits && pBuffer[fractionalDigits + 1] >= '5';

    if (round)
    {
      while (true)
      {
        if (pRoundChar < pBuffer)
        {
          *pRoundChar = '1';
          pBuffer--;
          numberBytes++;
          break;
        }
        else
        {
          const char c = ((*pRoundChar - '0') + 1) % 10;
          *pRoundChar = c + '0';

          if (c == 0)
            pRoundChar--;
          else
            break;
        }
      }
    }

    int64_t scientificExponent = exponent + numberBytes - 1;

    char exponentBuffer[13];
    char *pExponentBuffer = &exponentBuffer[mARRAYSIZE(exponentBuffer) - 1];
    size_t exponentBytes = 0;

    {
      size_t tmp = mAbs(scientificExponent);

      while (tmp >= 10)
      {
        *pExponentBuffer = (char)('0' + (tmp % 10));
        pExponentBuffer--;
        tmp /= 10;
        exponentBytes++;
      }

      if (tmp != 0 || exponentBytes == 0)
      {
        *pExponentBuffer = (char)('0' + tmp);
        pExponentBuffer--;
        exponentBytes++;
      }
    }

    // Add Sign & Expnent.
    if (scientificExponent < 0)
      *pExponentBuffer = '-';
    else
      *pExponentBuffer = '+';

    pExponentBuffer--;
    *pExponentBuffer = fs.exponentChar;
    exponentBytes += 2;

    return _mFormat_Append_DecimalFloatScientific(isNegative, signChar, signChars, numberBytes, pBuffer, 1, pExponentBuffer, exponentBytes, fractionalDigits, fs, text);
  }
}

size_t _mFormat_Append(const float_t value, const mFormatState &fs, char *text)
{
  typedef decltype(value) Float;
  typedef jkj::dragonbox::default_float_traits<Float> FloatTraits;

  auto const br = jkj::dragonbox::float_bits<Float, FloatTraits>(value);
  auto const exponent_bits = br.extract_exponent_bits();
  auto const s = br.remove_exponent_bits(exponent_bits);

  if (br.is_finite(exponent_bits))
  {
    if (br.is_nonzero())
    {
      auto result = jkj::dragonbox::to_decimal<Float, FloatTraits>(s, exponent_bits,
        jkj::dragonbox::policy::sign::ignore,
        jkj::dragonbox::policy::trailing_zero::remove,
        jkj::dragonbox::policy::decimal_to_binary_rounding::nearest_to_even,
        jkj::dragonbox::policy::binary_to_decimal_rounding::to_even,
        jkj::dragonbox::policy::cache::full);

      return _mFormat_HandleNonzeroFloat(s.is_negative(), result.significand, result.exponent, fs, text);
    }
    else
    {
      size_t signChars = 0;
      char signChar = '+';

      switch (fs.signOption)
      {
      case mFSO_Both:
        signChars = 1;
        break;

      case mFSO_NegativeOrFill:
        signChars = 1;

        if (fs.fillCharacterIsZero)
          signChar = ' ';
        else
          signChar = fs.fillCharacter;

        break;
      }

      if (!fs.scientificNotation)
      {
        return _mFormat_Append_DecimalFloat(false, signChar, signChars, 1, "0", 0, fs, text);
      }
      else
      {
        const size_t fractionalDigits = mMin(fs.fractionalDigits, (size_t)mMax(0LL, (int64_t)fs.maxChars - 1 - 1 - 3));

        char exponentBuffer[3];
        exponentBuffer[0] = fs.exponentChar;
        exponentBuffer[1] = '+';
        exponentBuffer[2] = '0';

        return _mFormat_Append_DecimalFloatScientific(false, signChar, signChars, 1, "0", 1, exponentBuffer, 3, fractionalDigits, fs, text);
      }
    }
  }
  else
  {
    if (s.has_all_zero_significand_bits())
    {
      if (s.is_negative())
        return _mFormat_Append_DisplayWithAlign_Internal(fs.negativeInfinityBytes, fs.negativeInfinityCount, text, fs.negativeInfinityChars, fs, false);
      else
        return _mFormat_Append_DisplayWithAlign_Internal(fs.infinityBytes, fs.infinityCount, text, fs.infinityChars, fs, false);
    }
    else
    {
      return _mFormat_Append_DisplayWithAlign_Internal(fs.nanBytes, fs.nanCount, text, fs.nanChars, fs, false);
    }
  }
}

size_t _mFormat_Append(const double_t value, const mFormatState &fs, char *text)
{
  typedef decltype(value) Float;
  typedef jkj::dragonbox::default_float_traits<Float> FloatTraits;

  auto const br = jkj::dragonbox::float_bits<Float, FloatTraits>(value);
  auto const exponent_bits = br.extract_exponent_bits();
  auto const s = br.remove_exponent_bits(exponent_bits);

  if (br.is_finite(exponent_bits))
  {
    if (br.is_nonzero())
    {
      auto result = jkj::dragonbox::to_decimal<Float, FloatTraits>(s, exponent_bits,
        jkj::dragonbox::policy::sign::ignore,
        jkj::dragonbox::policy::trailing_zero::ignore,
        jkj::dragonbox::policy::decimal_to_binary_rounding::nearest_to_even,
        jkj::dragonbox::policy::binary_to_decimal_rounding::to_even,
        jkj::dragonbox::policy::cache::full);

      return _mFormat_HandleNonzeroFloat(s.is_negative(), result.significand, result.exponent, fs, text);
    }
    else
    {
      size_t signChars = 0;
      char signChar = '+';

      switch (fs.signOption)
      {
      case mFSO_Both:
        signChars = 1;
        break;

      case mFSO_NegativeOrFill:
        signChars = 1;

        if (fs.fillCharacterIsZero)
          signChar = ' ';
        else
          signChar = fs.fillCharacter;

        break;
      }

      if (!fs.scientificNotation)
      {
        return _mFormat_Append_DecimalFloat(false, signChar, signChars, 1, "0", 0, fs, text);
      }
      else
      {
        const size_t fractionalDigits = mMin(fs.fractionalDigits, (size_t)mMax(0LL, (int64_t)fs.maxChars - 1 - 1 - 3));

        char exponentBuffer[3];
        exponentBuffer[0] = fs.exponentChar;
        exponentBuffer[1] = '+';
        exponentBuffer[2] = '0';

        return _mFormat_Append_DecimalFloatScientific(false, signChar, signChars, 1, "0", 1, exponentBuffer, 3, fractionalDigits, fs, text);
      }
    }
  }
  else
  {
    if (s.has_all_zero_significand_bits())
    {
      if (s.is_negative())
        return _mFormat_Append_DisplayWithAlign_Internal(fs.negativeInfinityBytes, fs.negativeInfinityCount, text, fs.negativeInfinityChars, fs, false);
      else
        return _mFormat_Append_DisplayWithAlign_Internal(fs.infinityBytes, fs.infinityCount, text, fs.infinityChars, fs, false);
    }
    else
    {
      return _mFormat_Append_DisplayWithAlign_Internal(fs.nanBytes, fs.nanCount, text, fs.nanChars, fs, false);
    }
  }
}

size_t _mFormat_AppendBool(const bool value, const mFormatState &fs, char *text)
{
  if (value)
    return _mFormat_AppendStringWithLength(fs.trueChars, fs.trueBytes, fs, text);
  else
    return _mFormat_AppendStringWithLength(fs.falseChars, fs.falseBytes, fs, text);
}

size_t _mFormat_Append(const wchar_t value, const mFormatState &fs, char *text)
{
  if (value == 0)
    return 0;

  char buffer[mString_MaxUtf16CharInUtf8Chars + 1];
  wchar_t wbuffer[2] = { value, 0 };

  const size_t bytes = WideCharToMultiByte(CP_UTF8, 0, wbuffer, (int32_t)mARRAYSIZE(wbuffer), buffer, (int32_t)mARRAYSIZE(buffer), nullptr, false);

  if (bytes <= 1 || fs.maxChars < bytes - 1)
    return 0;

  memcpy(text, buffer, bytes - 1);
  
  return bytes - 1;
}

size_t _mFormat_AppendWStringWithLength(const wchar_t *string, const size_t charCount, const mFormatState &fs, char *text)
{
  if (string == nullptr)
    return 0;

  const size_t wcharLength = charCount + 1;
  char *buffer = nullptr;
  const size_t bufferCapacity = (wcharLength - 1) * mString_MaxUtf16CharInUtf8Chars + 1;
  mAllocator * const pAllocator = &mDefaultTempAllocator;

  if (mFAILED(mAllocator_AllocateZero(pAllocator, &buffer, bufferCapacity)))
    return 0;

  mDEFER_CALL_2(mAllocator_FreePtr, pAllocator, &buffer);

  const size_t bytes = WideCharToMultiByte(CP_UTF8, 0, string, (int32_t)wcharLength, buffer, (int32_t)bufferCapacity, nullptr, false);

  if (bytes <= 1)
    return 0;

  return _mFormat_AppendStringWithLength(buffer, bytes - 1, fs, text);
}

//////////////////////////////////////////////////////////////////////////

void _mFormat_Append_DecimalDigitsWithGroupDigitsNoSign_Internal(const size_t numberBytes, char *text, const char *buffer, const mFormatState &fs)
{
  size_t numbersRemaining = numberBytes;

  *text = *buffer;
  text++;
  buffer++;
  numbersRemaining--;

  while (numbersRemaining)
  {
    bool groupingCharNecessary = false;

    switch (fs.digitGroupingOption)
    {
    default:
    case mFDGO_Thousand:
      groupingCharNecessary = (numbersRemaining % 3 == 0);
      break;

    case mFDGO_TenThousand:
      groupingCharNecessary = (numbersRemaining % 4 == 0);
      break;

    case mFDGO_Indian:
      groupingCharNecessary = ((numbersRemaining > 3 && numbersRemaining % 2 == 1) || numbersRemaining == 3);
      break;
    }

    if (groupingCharNecessary)
    {
      memcpy(text, fs.digitGroupingChars, fs.digitGroupingCharLength);
      text += fs.digitGroupingCharLength;
    }

    *text = *buffer;
    text++;
    buffer++;
    numbersRemaining--;
  }
}

void _mFormat_Append_DisplayWithAlignNoGroupingWithSign_Internal(const size_t totalBytes, const size_t signChars, const char signChar, const size_t numberBytes, char *text, const char *buffer, const mFormatState &fs)
{
  size_t alignedChars = totalBytes;

  if ((!fs.alignSign || fs.numberAlign == mFA_Left) && signChars)
  {
    *text = signChar;
    text++;
    alignedChars--;
  }

  switch (fs.numberAlign)
  {
  default:
  case mFA_Left:
  {
    memcpy(text, buffer, numberBytes);
    text += numberBytes;

    size_t spacing = 0;
    char fillCharacter = fs.fillCharacter;

    if (fs.fillCharacterIsZero)
    {
      mFAIL_DEBUG("This isn't how fill characters are intended to be used...");
      fillCharacter = ' ';
    }

    while (totalBytes + spacing <= fs.minChars)
    {
      *text = fillCharacter;
      text++;
      spacing++;
    }

    break;
  }

  case mFA_Right:
  {
    size_t spacing = 0;

    while (totalBytes + spacing < fs.minChars)
    {
      *text = fs.fillCharacter;
      text++;
      spacing++;
    }

    if (fs.alignSign && signChars)
    {
      *text = signChar;
      text++;
    }

    memcpy(text, buffer, numberBytes);

    break;
  }

  case mFA_Center:
  {
    size_t spacing = 0;
    char fillCharacter = fs.fillCharacter;

    if (fs.fillCharacterIsZero)
    {
      mFAIL_DEBUG("This isn't how fill characters are intended to be used...");
      fillCharacter = ' ';
    }

    while (spacing < (fs.minChars - totalBytes) / 2)
    {
      *text = fillCharacter;
      text++;
      spacing++;
    }

    if (fs.alignSign && signChars)
    {
      *text = signChar;
      text++;
    }

    memcpy(text, buffer, numberBytes);
    text += numberBytes;

    while (totalBytes + spacing < fs.minChars)
    {
      *text = fillCharacter;
      text++;
      spacing++;
    }

    break;
  }
  }
}

size_t _mFormat_Append_DisplayWithAlign_Internal(const size_t totalBytes, const size_t totalChars, char *text, const char *buffer, const mFormatState &fs, const bool isNumber)
{
  char *originalTextPosition = text;

  if (totalChars <= fs.maxChars)
  {
    const mFormatAlign align = isNumber ? fs.numberAlign : fs.stringAlign;

    switch (align)
    {
    default:
    case mFA_Left:
    {
      memcpy(text, buffer, totalBytes);
      text += totalBytes;

      size_t spacing = 0;
      char fillCharacter = fs.fillCharacter;

      if (fs.fillCharacterIsZero)
      {
        mFAIL_DEBUG("This isn't how fill characters are intended to be used...");
        fillCharacter = ' ';
      }

      while (totalChars + spacing < fs.minChars)
      {
        *text = fillCharacter;
        text++;
        spacing++;
      }

      return text - originalTextPosition;
    }

    case mFA_Right:
    {
      size_t spacing = 0;

      while (totalChars + spacing < fs.minChars)
      {
        *text = fs.fillCharacter;
        text++;
        spacing++;
      }

      memcpy(text, buffer, totalBytes);

      return text - originalTextPosition + totalBytes;
    }

    case mFA_Center:
    {
      size_t spacing = 0;
      char fillCharacter = fs.fillCharacter;

      if (fs.fillCharacterIsZero)
      {
        mFAIL_DEBUG("This isn't how fill characters are intended to be used...");
        fillCharacter = ' ';
      }

      if (fs.minChars > totalChars)
      {
        while (spacing < (fs.minChars - totalChars) / 2)
        {
          *text = fillCharacter;
          text++;
          spacing++;
        }
      }

      memcpy(text, buffer, totalBytes);
      text += totalBytes;

      while (totalChars + spacing < fs.minChars)
      {
        *text = fillCharacter;
        text++;
        spacing++;
      }

      return text - originalTextPosition;
    }
    }
  }
  else
  {
    if (isNumber)
    {
      switch (fs.numberOverflow)
      {
      case mFOB_AlignLeft:
        memcpy(text, buffer, fs.maxChars);
        break;

      default:
      case mFOB_AlignRight:
        buffer += totalBytes - fs.maxChars;
        memcpy(text, buffer, fs.maxChars);
        break;
      }

      return fs.maxChars;
    }
    else
    {
      bool useEllipsis = fs.stringOverflowEllipsis && fs.maxChars >= fs.stringOverflowEllipsisCount;

      size_t bytesRemaining = totalBytes;
      size_t charsRemaining = fs.maxChars;

      if (useEllipsis)
        charsRemaining -= fs.stringOverflowEllipsisCount;

      while (charsRemaining > 0 && bytesRemaining > 0)
      {
        size_t charSize;

        if (!mString_IsValidChar(buffer, bytesRemaining, nullptr, &charSize))
          return totalBytes - bytesRemaining;

        *text = *buffer;

        if (charSize >= 2)
          text[1] = buffer[1];

        if (charSize >= 3)
          text[2] = buffer[2];

        if (charSize == 4)
          text[3] = buffer[3];

        mASSERT_DEBUG(charSize <= 4, "Invalid char size.");

        text += charSize;
        buffer += charSize;
        bytesRemaining -= charSize;
        charsRemaining--;
      }

      if (useEllipsis)
      {
        memcpy(text, fs.stringOverflowEllipsisChars, fs.stringOverflowEllipsisLength);

        return text - originalTextPosition + fs.stringOverflowEllipsisLength;
      }
      else
      {
        return text - originalTextPosition;
      }
    }
  }
}

size_t _mFormat_Append_DisplayWithAlign_Internal(const size_t totalBytes, char *text, const char *buffer, const mFormatState &fs)
{
  char *originalTextPosition = text;

  if (totalBytes <= fs.maxChars)
  {
    switch (fs.stringAlign)
    {
    default:
    case mFA_Left:
    {
      memcpy(text, buffer, totalBytes);
      text += totalBytes;

      if (fs.minChars * 4 > totalBytes)
      {
        size_t spacing = 0;
        const size_t totalChars = _mFormat_GetStringCount(buffer, totalBytes);

        if (totalChars)
        {
          char fillCharacter = fs.fillCharacter;

          if (fs.fillCharacterIsZero)
          {
            mFAIL_DEBUG("This isn't how fill characters are intended to be used...");
            fillCharacter = ' ';
          }

          while (totalChars + spacing < fs.minChars)
          {
            *text = fillCharacter;
            text++;
            spacing++;
          }
        }
      }

      return text - originalTextPosition;
    }

    case mFA_Right:
    {
      const size_t totalChars = _mFormat_GetStringCount(buffer, totalBytes);

      if (totalChars)
      {
        size_t spacing = 0;

        while (totalChars + spacing < fs.minChars)
        {
          *text = fs.fillCharacter;
          text++;
          spacing++;
        }
      }

      memcpy(text, buffer, totalBytes);

      return text - originalTextPosition + totalBytes;
    }

    case mFA_Center:
    {
      size_t spacing = 0;
      char fillCharacter = fs.fillCharacter;

      if (fs.fillCharacterIsZero)
      {
        mFAIL_DEBUG("This isn't how fill characters are intended to be used...");
        fillCharacter = ' ';
      }

      const size_t totalChars = _mFormat_GetStringCount(buffer, totalBytes);

      if (totalChars < fs.minChars)
      {
        while (spacing < (fs.minChars - totalChars) / 2)
        {
          *text = fillCharacter;
          text++;
          spacing++;
        }
      }

      memcpy(text, buffer, totalBytes);
      text += totalBytes;

      if (totalChars < fs.minChars)
      {
        while (totalChars + spacing < fs.minChars)
        {
          *text = fillCharacter;
          text++;
          spacing++;
        }
      }

      return text - originalTextPosition;
    }
    }
  }
  else
  {
    const size_t totalChars = _mFormat_GetStringCount(buffer, totalBytes);

    return _mFormat_Append_DisplayWithAlign_Internal(totalBytes, totalChars, text, buffer, fs, false);
  }
}

size_t _mFormat_Append_DecimalInsufficientSize(const bool negative, const size_t signChars, const char signChar, const mFormatState &fs, char *text)
{
  if (fs.maxChars < 2 || (signChars && fs.maxChars < 3))
  {
    return 0;
  }
  else
  {
    *text = negative ? '<' : '>';
    text++;

    size_t charsRemaining = fs.maxChars - 1;

    if (signChars)
    {
      *text = signChar;
      text++;
      charsRemaining -= 1;
    }
    else if (negative)
    {
      *text = '0';
      return 2;
    }

    text += charsRemaining - 1;

    *text = '9';
    text--;
    charsRemaining--;
    size_t digits = 1;

    while (charsRemaining)
    {
      if (fs.groupDigits)
      {
        bool groupingCharNecessary;

        switch (fs.digitGroupingOption)
        {
        default:
        case mFDGO_Thousand:
          groupingCharNecessary = (digits % 3 == 0);
          break;

        case mFDGO_TenThousand:
          groupingCharNecessary = (digits % 4 == 0);
          break;

        case mFDGO_Indian:
          groupingCharNecessary = ((digits > 3 && digits % 2 == 1) || digits == 3);
          break;
        }

        if (groupingCharNecessary)
        {
          if (charsRemaining <= fs.digitGroupingCharLength)
          {
            char fillCharacter = fs.fillCharacter;

            if (fs.fillCharacterIsZero)
              fillCharacter = ' ';

            for (size_t i = 0; i < charsRemaining; i++)
            {
              *text = fillCharacter;
              text--;
            }

            break;
          }

          mASSERT_DEBUG(fs.digitGroupingCharLength >= 1, "If you don't want a digit grouping char, set fs.digitGroupingOption to false.");
          text -= (fs.digitGroupingCharLength - 1);
          memcpy(text, fs.digitGroupingChars, fs.digitGroupingCharLength);
          text--;
          charsRemaining -= fs.digitGroupingCharLength;
        }
      }

      *text = '9';
      text--;
      charsRemaining--;
      digits++;
    }

    return fs.maxChars;
  }
}

// Side effect: The buffer should provide room for one more digit in front of the actual first digit (in case rounding introduces another digit).
size_t _mFormat_Append_DecimalFloat(const bool negative, const char signChar, const size_t signChars, const size_t digits, char *inputBuffer, const int64_t exponent, const mFormatState &fs, char *text)
{
  size_t inputDigits = digits;
  int64_t inputExponent = exponent;
  char *buffer = inputBuffer;

  // Handle rounding.
  {
    const int64_t tmpDecimalSeparatorPosition = (int64_t)inputDigits + inputExponent;
    const size_t tmpSignificantDigits = (size_t)mMax(0LL, tmpDecimalSeparatorPosition);
    const size_t tmpNecessaryChars = signChars + mMax(1ULL, tmpSignificantDigits + fs.groupDigits * (_mFormat_GetDigitGroupingCharCount(tmpSignificantDigits, fs) * fs.digitGroupingCharLength));
    const size_t tmpExistentFractionalDigits = (size_t)mMax(0LL, (int64_t)inputDigits - tmpDecimalSeparatorPosition);
    size_t tmpMaxFractionalChars = mMin(fs.maxChars - tmpNecessaryChars, fs.fractionalDigits + 1);
    
    if (fs.fillCharacterIsZero && fs.numberAlign == mFA_Left && fs.minChars > tmpNecessaryChars)
      tmpMaxFractionalChars = mMax(fs.minChars - tmpNecessaryChars, tmpMaxFractionalChars);
    
    const size_t tmpMaxFractionalDigitChars = !!tmpMaxFractionalChars * (tmpMaxFractionalChars - 1LL);

    if (tmpExistentFractionalDigits > tmpMaxFractionalDigitChars)
    {
      char *roundChar = buffer + tmpMaxFractionalDigitChars + tmpDecimalSeparatorPosition;

      if (*roundChar >= '5')
      {
        roundChar--;

        while (true)
        {
          if (roundChar < buffer)
          {
            *roundChar = '1';
            buffer--;
            inputDigits++;
            break;
          }
          else
          {
            const char c = ((*roundChar - '0') + 1) % 10;
            *roundChar = c + '0';

            if (c == 0)
              roundChar--;
            else
              break;
          }
        }
      }
    }
  }

  size_t usefulDigits = inputDigits;
  int64_t decimalSeparatorPosition = inputDigits + inputExponent;
  const size_t significantDigits = (size_t)mMax(0LL, decimalSeparatorPosition);

  const size_t existentSignificantDigits = (size_t)mMax(0LL, mMin((int64_t)inputDigits, decimalSeparatorPosition));
  const size_t significantChars = mMax(1ULL, significantDigits + fs.groupDigits * (_mFormat_GetDigitGroupingCharCount(significantDigits, fs) * fs.digitGroupingCharLength));
  const size_t necessaryChars = significantChars + signChars;

  size_t maxFractionalChars = (size_t)mMin(mMax(0LL, (int64_t)fs.maxChars - (int64_t)necessaryChars), (int64_t)fs.fractionalDigits + 1);

  if (fs.fillCharacterIsZero && fs.numberAlign == mFA_Left && fs.minChars > necessaryChars)
    maxFractionalChars = mMax(fs.minChars - necessaryChars, maxFractionalChars);
  
  const size_t maxFractionalDigitChars = !!maxFractionalChars * (maxFractionalChars - 1LL);

  if (fs.adaptiveFractionalDigits)
  {
    size_t lastUsefulDigit = (size_t)-1;

    for (size_t i = 0; i < maxFractionalDigitChars; i++)
    {
      const int64_t index = decimalSeparatorPosition + i;

      if (index < 0)
        continue;
      else if (index >= (int64_t)usefulDigits || index >= (int64_t)(maxFractionalDigitChars + significantDigits))
        break;
      else if (buffer[index] != '0')
        lastUsefulDigit = i;
    }

    usefulDigits = lastUsefulDigit + 1 + decimalSeparatorPosition;
  }

  const size_t usefulFractionalDigits = mMax(0LL, (int64_t)usefulDigits - decimalSeparatorPosition);
  const size_t totalFractionalDigits = fs.adaptiveFractionalDigits ? mMin(usefulFractionalDigits, maxFractionalDigitChars) : maxFractionalDigitChars;
  const size_t totalChars = necessaryChars + !!totalFractionalDigits + totalFractionalDigits; // The !!totalPostDecimalSeparatorDigits is the decimal separator.

  if (totalChars <= fs.maxChars)
  {
    const bool fitsExact = totalChars >= fs.minChars;
    size_t spacing = 0;

    if (signChars && (fitsExact || !fs.alignSign || fs.numberAlign == mFA_Left))
    {
      *text = signChar;
      text++;
    }

    if (!fitsExact)
    {
      if (fs.numberAlign == mFA_Right)
      {
        if (fs.fillCharacterIsZero && signChars && fs.alignSign)
        {
          *text = signChar;
          text++;
        }

        if (fs.fillCharacterIsZero && fs.groupDigits)
        {
          size_t digitIndex = mMax(significantDigits, 1ULL);
          size_t charsRemaining = fs.minChars - totalChars;
          text += charsRemaining;
          char *pTmpBuffer = text - 1;

          while (charsRemaining)
          {
            bool groupingCharNecessary;

            switch (fs.digitGroupingOption)
            {
            default:
            case mFDGO_Thousand:
              groupingCharNecessary = (digitIndex % 3 == 0);
              break;

            case mFDGO_TenThousand:
              groupingCharNecessary = (digitIndex % 4 == 0);
              break;

            case mFDGO_Indian:
              groupingCharNecessary = ((digitIndex > 3 && digitIndex % 2 == 1) || digitIndex == 3);
              break;
            }

            if (groupingCharNecessary)
            {
              if (charsRemaining <= fs.digitGroupingCharLength)
              {
                char fillCharacter = ' ';

                for (size_t i = 0; i < charsRemaining; i++)
                {
                  *pTmpBuffer = fillCharacter;
                  pTmpBuffer--;
                }

                break;
              }

              mASSERT_DEBUG(fs.digitGroupingCharLength >= 1, "If you don't want a digit grouping char, set fs.digitGroupingOption to false.");
              pTmpBuffer -= (fs.digitGroupingCharLength - 1);
              memcpy(pTmpBuffer, fs.digitGroupingChars, fs.digitGroupingCharLength);
              pTmpBuffer--;
              charsRemaining -= fs.digitGroupingCharLength;
            }

            *pTmpBuffer = '0';
            pTmpBuffer--;
            charsRemaining--;
            digitIndex++;
          }
        }
        else
        {
          while (totalChars + spacing < fs.minChars)
          {
            *text = fs.fillCharacter;
            text++;
            spacing++;
          }

          if (!fs.fillCharacterIsZero && signChars && fs.alignSign)
          {
            *text = signChar;
            text++;
          }
        }
      }
      else if (fs.numberAlign == mFA_Center)
      {
        char fillCharacter = fs.fillCharacter;

        if (fs.fillCharacterIsZero)
        {
          mFAIL_DEBUG("This isn't how fill characters are intended to be used...");
          fillCharacter = ' ';
        }

        while (spacing < (fs.minChars - totalChars) / 2)
        {
          *text = fillCharacter;
          text++;
          spacing++;
        }

        if (signChars && fs.alignSign)
        {
          *text = signChar;
          text++;
        }
      }
    }

    // Significant Digits.
    if (significantDigits)
    {
      text += significantChars;
      char *pTmpBuffer = text - 1;
      size_t charsRemaining = significantChars;
      size_t digitIndex = 0;

      while (charsRemaining)
      {
        bool groupingCharNecessary = false;

        if (fs.groupDigits && digitIndex > 0)
        {
          switch (fs.digitGroupingOption)
          {
          default:
          case mFDGO_Thousand:
            groupingCharNecessary = (digitIndex % 3 == 0);
            break;

          case mFDGO_TenThousand:
            groupingCharNecessary = (digitIndex % 4 == 0);
            break;

          case mFDGO_Indian:
            groupingCharNecessary = ((digitIndex > 3 && digitIndex % 2 == 1) || digitIndex == 3);
            break;
          }

          if (groupingCharNecessary)
          {
            if (charsRemaining <= fs.digitGroupingCharLength)
            {
              char fillCharacter = ' ';

              for (size_t i = 0; i < charsRemaining; i++)
              {
                *pTmpBuffer = fillCharacter;
                pTmpBuffer--;
              }

              break;
            }

            mASSERT_DEBUG(fs.digitGroupingCharLength >= 1, "If you don't want a digit grouping char, set fs.digitGroupingOption to false.");
            pTmpBuffer -= (fs.digitGroupingCharLength - 1);
            memcpy(pTmpBuffer, fs.digitGroupingChars, fs.digitGroupingCharLength);
            pTmpBuffer--;
            charsRemaining -= fs.digitGroupingCharLength;
          }
        }

        if (significantDigits - digitIndex > existentSignificantDigits)
          *pTmpBuffer = '0';
        else
          *pTmpBuffer = buffer[significantDigits - digitIndex - 1];
        
        pTmpBuffer--;
        charsRemaining--;
        digitIndex++;
      }
    }
    else
    {
      *text = '0';
      text++;
    }

    // Fractional digits.
    if (totalFractionalDigits > 0)
    {
      memcpy(text, fs.decimalSeparatorChars, fs.decimalSeparatorLength);
      text += fs.decimalSeparatorLength;

      size_t currentDecimalDigits = 0;

      if (decimalSeparatorPosition < 0)
      {
        const size_t digitsToAdd = mMin((size_t)-decimalSeparatorPosition, totalFractionalDigits);
        memset(text, '0', digitsToAdd);
        text += digitsToAdd;
        currentDecimalDigits += digitsToAdd;
      }

      if (usefulFractionalDigits > 0 && totalFractionalDigits > currentDecimalDigits)
      {
        const size_t digitsToCopy = mMin(mMin(usefulDigits, usefulFractionalDigits), totalFractionalDigits - currentDecimalDigits);
        memcpy(text, buffer + mMax(0LL, decimalSeparatorPosition), digitsToCopy);
        text += digitsToCopy;
        currentDecimalDigits += digitsToCopy;
      }

      if (totalFractionalDigits > currentDecimalDigits)
      {
        const size_t digitsToAdd = totalFractionalDigits - currentDecimalDigits;
        memset(text, '0', digitsToAdd);
        text += digitsToAdd;
        //currentDecimalDigits += digitsToAdd; // irrelevant, since nobody's gonna add any more decimal digits.
      }
    }

    if (!fitsExact)
    {
      if (fs.fillCharacterIsZero && fs.numberAlign == mFA_Left)
      {
        if (totalFractionalDigits == 0)
        {
          if (fs.minChars - totalChars > fs.decimalSeparatorLength)
          {
            memcpy(text, fs.decimalSeparatorChars, fs.decimalSeparatorLength);
            text += fs.decimalSeparatorLength;
            spacing += fs.decimalSeparatorLength;

            while (spacing + totalChars < fs.minChars)
            {
              *text = '0';
              text++;
              spacing++;
            }
          }
        }
        else
        {
          while (spacing + totalChars < fs.minChars)
          {
            *text = '0';
            text++;
            spacing++;
          }
        }
      }
      
      if (fs.numberAlign == mFA_Center || fs.numberAlign == mFA_Left)
      {
        char fillCharacter = fs.fillCharacter;

        if (fs.fillCharacterIsZero)
          fillCharacter = ' '; // we can end up here if there isn't enough place for a decimal separator as well.

        while (spacing + totalChars < fs.minChars)
        {
          *text = fillCharacter;
          text++;
          spacing++;
        }
      }
    }

    if (fitsExact)
      return totalChars;
    else
      return mMax(fs.minChars, totalChars);
  }
  else
  {
    return _mFormat_Append_DecimalInsufficientSize(negative, signChars, signChar, fs, text);
  }
}

size_t _mFormat_Append_DecimalFloatScientific(const bool negative, const char signChar, const size_t signChars, const size_t digits, const char *buffer, const size_t decimalSeparatorPosition, const char *exponentBuffer, const size_t exponentLength, const size_t fractionalDigits, const mFormatState &fs, char *text)
{
  mUnused(negative);

  mASSERT_DEBUG(decimalSeparatorPosition <= digits, "Unexpected decimal separator position.");

  size_t usefulDigits = digits;
  size_t usefulFractionalDigits = fractionalDigits;

  if (fs.adaptiveFractionalDigits)
  {
    size_t lastUsefulDigit = 0;

    for (size_t i = 0; i <= fractionalDigits; i++)
    {
      if (decimalSeparatorPosition + i > digits)
        break;
      else if (buffer[decimalSeparatorPosition + i] != '0')
        lastUsefulDigit = i;
    }

    usefulDigits = (lastUsefulDigit + decimalSeparatorPosition);
    usefulFractionalDigits = usefulDigits - decimalSeparatorPosition;
  }

  const size_t totalChars = signChars + decimalSeparatorPosition + (usefulFractionalDigits > 0 ? (fs.decimalSeparatorLength + usefulFractionalDigits) : 0) + exponentLength;

  mASSERT_DEBUG(decimalSeparatorPosition <= usefulDigits, "Unexpected decimal separator position.");

  const bool fitsExact = totalChars >= fs.minChars;
  size_t spacing = 0;

  if (signChars && (fitsExact || !fs.alignSign || fs.numberAlign == mFA_Left))
  {
    *text = signChar;
    text++;
  }

  if (!fitsExact)
  {
    if (fs.numberAlign == mFA_Right)
    {
      if (fs.fillCharacterIsZero && signChars && fs.alignSign)
      {
        *text = signChar;
        text++;
      }

      if (fs.fillCharacterIsZero && fs.groupDigits)
      {
        size_t digitIndex = decimalSeparatorPosition;
        size_t charsRemaining = fs.minChars - totalChars;
        text += charsRemaining;
        char *pTmpBuffer = text - 1;

        while (charsRemaining)
        {
          bool groupingCharNecessary;

          switch (fs.digitGroupingOption)
          {
          default:
          case mFDGO_Thousand:
            groupingCharNecessary = (digitIndex % 3 == 0);
            break;

          case mFDGO_TenThousand:
            groupingCharNecessary = (digitIndex % 4 == 0);
            break;

          case mFDGO_Indian:
            groupingCharNecessary = ((digitIndex > 3 && digitIndex % 2 == 1) || digitIndex == 3);
            break;
          }

          if (groupingCharNecessary)
          {
            if (charsRemaining <= fs.digitGroupingCharLength)
            {
              char fillCharacter = ' ';

              for (size_t i = 0; i < charsRemaining; i++)
              {
                *pTmpBuffer = fillCharacter;
                pTmpBuffer--;
              }

              break;
            }

            mASSERT_DEBUG(fs.digitGroupingCharLength >= 1, "If you don't want a digit grouping char, set fs.digitGroupingOption to false.");
            pTmpBuffer -= (fs.digitGroupingCharLength - 1);
            memcpy(pTmpBuffer, fs.digitGroupingChars, fs.digitGroupingCharLength);
            pTmpBuffer--;
            charsRemaining -= fs.digitGroupingCharLength;
          }

          *pTmpBuffer = '0';
          pTmpBuffer--;
          charsRemaining--;
          digitIndex++;
        }
      }
      else
      {
        while (totalChars + spacing < fs.minChars)
        {
          *text = fs.fillCharacter;
          text++;
          spacing++;
        }

        if (!fs.fillCharacterIsZero && signChars && fs.alignSign)
        {
          *text = signChar;
          text++;
        }
      }
    }
    else if (fs.numberAlign == mFA_Center)
    {
      char fillCharacter = fs.fillCharacter;

      if (fs.fillCharacterIsZero)
      {
        mFAIL_DEBUG("This isn't how fill characters are intended to be used...");
        fillCharacter = ' ';
      }

      while (spacing < (fs.minChars - totalChars) / 2)
      {
        *text = fillCharacter;
        text++;
        spacing++;
      }

      if (signChars && fs.alignSign)
      {
        *text = signChar;
        text++;
      }
    }
  }

  // Significant Digits.
  memcpy(text, buffer, decimalSeparatorPosition);
  text += decimalSeparatorPosition;

  if (usefulFractionalDigits > 0)
  {
    memcpy(text, fs.decimalSeparatorChars, fs.decimalSeparatorLength);
    text += fs.decimalSeparatorLength;

    // Decimal digits.
    for (size_t i = 0; i < usefulFractionalDigits; i++)
    {
      if (i + decimalSeparatorPosition >= usefulDigits)
        *text = '0';
      else
        *text = buffer[i + decimalSeparatorPosition];

      text++;
    }
  }

  if (!fitsExact && fs.fillCharacterIsZero && fs.numberAlign == mFA_Left)
  {
    if (usefulFractionalDigits == 0)
    {
      if (fs.minChars - totalChars > fs.decimalSeparatorLength)
      {
        memcpy(text, fs.decimalSeparatorChars, fs.decimalSeparatorLength);
        text += fs.decimalSeparatorLength;
        spacing += fs.decimalSeparatorLength;

        while (spacing + totalChars < fs.minChars)
        {
          *text = '0';
          text++;
          spacing++;
        }
      }
    }
    else
    {
      while (spacing + totalChars < fs.minChars)
      {
        *text = '0';
        text++;
        spacing++;
      }
    }
  }

  memcpy(text, exponentBuffer, exponentLength);
  text += exponentLength;

  if (!fitsExact)
  {
    if (fs.numberAlign == mFA_Center || fs.numberAlign == mFA_Left)
    {
      char fillCharacter = fs.fillCharacter;

      if (fs.fillCharacterIsZero)
        fillCharacter = ' '; // we can end up here if there isn't enough place for a decimal separator as well.

      while (spacing + totalChars < fs.minChars)
      {
        *text = fillCharacter;
        text++;
        spacing++;
      }
    }
  }

  if (fitsExact)
    return totalChars;
  else
    return mMax(fs.minChars, totalChars);
}

size_t _mFormat_Append_Decimal(const bool negative, const char signChar, const size_t signChars, const size_t _digits, const char *buffer, const mFormatState &fs, char *text)
{
  mASSERT_DEBUG(_digits > 0, "The number of digits should NEVER be zero here.");

  size_t numberBytes = _digits;

  if (fs.groupDigits)
  {
    size_t totalBytes = signChars + numberBytes + _mFormat_GetDigitGroupingCharCount(numberBytes, fs) * fs.digitGroupingCharLength;

    if (fs.minChars <= totalBytes && fs.maxChars >= totalBytes)
    {
      if (signChars)
      {
        *text = signChar;
        text++;
      }

      _mFormat_Append_DecimalDigitsWithGroupDigitsNoSign_Internal(numberBytes, text, buffer, fs);

      return totalBytes;
    }
    else if (fs.maxChars < totalBytes)
    {
      return _mFormat_Append_DecimalInsufficientSize(negative, signChars, signChar, fs, text);
    }
    else if (fs.minChars > totalBytes)
    {
      size_t alignedChars = totalBytes;

      if ((!fs.alignSign || fs.numberAlign == mFA_Left) && signChars)
      {
        *text = signChar;
        text++;
        alignedChars--;
      }
      
      switch (fs.numberAlign)
      {
      default:
      case mFA_Left:
      {
        char fillCharacter = fs.fillCharacter;

        if (fs.fillCharacterIsZero)
        {
          mFAIL_DEBUG("This isn't how fill characters are intended to be used...");
          fillCharacter = ' ';
        }

        _mFormat_Append_DecimalDigitsWithGroupDigitsNoSign_Internal(numberBytes, text, buffer, fs);
        text += totalBytes - signChars;

        size_t spacing = 0;
      
        while (totalBytes + spacing < fs.minChars)
        {
          *text = fillCharacter;
          text++;
          spacing++;
        }
      
        break;
      }
      
      case mFA_Right:
      {      
        size_t spacing = 0;
      
        if (!fs.fillCharacterIsZero)
        {
          while (totalBytes + spacing < fs.minChars)
          {
            *text = fs.fillCharacter;
            text++;
            spacing++;
          }

          if (fs.alignSign && signChars)
          {
            *text = signChar;
            text++;
          }
        }
        else
        {
          size_t fillCharsRemaining = fs.minChars - numberBytes - signChars;
          size_t fillCharDigit = numberBytes;

          if (fs.alignSign && signChars)
          {
            *text = signChar;
            text++;
          }

          char *tempText = text + fillCharsRemaining - 1;
          text += fillCharsRemaining;

          while (fillCharsRemaining)
          {
            bool groupingCharNecessary;

            switch (fs.digitGroupingOption)
            {
            default:
            case mFDGO_Thousand:
              groupingCharNecessary = (fillCharDigit % 3 == 0);
              break;

            case mFDGO_TenThousand:
              groupingCharNecessary = (fillCharDigit % 4 == 0);
              break;

            case mFDGO_Indian:
              groupingCharNecessary = ((fillCharDigit > 3 && fillCharDigit % 2 == 1) || fillCharDigit == 3);
              break;
            }

            if (groupingCharNecessary)
            {
              if (fillCharsRemaining <= fs.digitGroupingCharLength)
              {
                char fillCharacter = fs.fillCharacter;

                if (fs.fillCharacterIsZero)
                  fillCharacter = ' ';

                for (size_t i = 0; i < fillCharsRemaining; i++)
                {
                  *tempText = fillCharacter;
                  tempText--;
                }

                break;
              }

              mASSERT_DEBUG(fs.digitGroupingCharLength >= 1, "If you don't want a digit grouping char, set fs.digitGroupingOption to false.");
              tempText -= (fs.digitGroupingCharLength - 1);
              memcpy(tempText, fs.digitGroupingChars, fs.digitGroupingCharLength);
              tempText--;
              fillCharsRemaining -= fs.digitGroupingCharLength;
            }

            *tempText = '0';
            tempText--;
            fillCharsRemaining--;
            fillCharDigit++;
          }
        }

        _mFormat_Append_DecimalDigitsWithGroupDigitsNoSign_Internal(numberBytes, text, buffer, fs);
      
        break;
      }
      
      case mFA_Center:
      {
        size_t spacing = 0;
        char fillCharacter = fs.fillCharacter;
      
        if (fs.fillCharacterIsZero)
        {
          mFAIL_DEBUG("This isn't how fill characters are intended to be used...");
          fillCharacter = ' ';
        }
      
        while (spacing < (fs.minChars - totalBytes) / 2)
        {
          *text = fillCharacter;
          text++;
          spacing++;
        }
      
        if (fs.alignSign && signChars)
        {
          *text = signChar;
          text++;
        }

        _mFormat_Append_DecimalDigitsWithGroupDigitsNoSign_Internal(numberBytes, text, buffer, fs);
        text += totalBytes - signChars;
      
        while (totalBytes + spacing < fs.minChars)
        {
          *text = fillCharacter;
          text++;
          spacing++;
        }
      
        break;
      }
      }
      
      return fs.minChars;
    }
    else
    {
      mFAIL_DEBUG("This state is not intended to ever be hit (unless you've perhaps configured the mFormatState incorrectly).");
      return 0;
    }
  }
  else
  {
    size_t totalBytes = signChars + numberBytes;

    if (fs.minChars <= totalBytes && fs.maxChars >= totalBytes)
    {
      if (signChars)
      {
        *text = signChar;
        text++;
      }

      memcpy(text, buffer, numberBytes);

      return totalBytes;
    }
    else if (fs.maxChars < totalBytes)
    {
      return _mFormat_Append_DecimalInsufficientSize(negative, signChars, signChar, fs, text);
    }
    else if (fs.minChars > totalBytes)
    {
      _mFormat_Append_DisplayWithAlignNoGroupingWithSign_Internal(totalBytes, signChars, signChar, numberBytes, text, buffer, fs);

      return fs.minChars;
    }
    else
    {
      mFAIL_DEBUG("This state is not intended to ever be hit (unless you've perhaps configured the mFormatState incorrectly).");
      return 0;
    }
  }
}
