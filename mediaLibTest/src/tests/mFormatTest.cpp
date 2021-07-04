#include "mTestLib.h"

#include "mFormat.h"

#define mTEST_FORMAT(expected, result) \
  do { memset(mFormat_GetState().textStart, 0xFF, mFormat_GetState().textCapacity); \
    const char *r = (result); \
    if (strcmp(expected, r) != 0) { \
      printf("\n\nUnexpected result:\nExpected:\n\t'%s'\nReceived:\n\t'%s'\n\n", expected, r); \
      mTEST_FAIL(); \
    } \
  } while (0); 

mTEST(mFormat, TestStrings)
{
  mTEST_ALLOCATOR_SETUP();

  mFormatState_ResetCulture();

  mFormatState &fs = mFormat_GetState();

  mDEFER(mAllocator_FreePtr(fs.pAllocator, &fs.textStart); fs.textCapacity = 0; fs.pAllocator = nullptr;);
  mAllocator_FreePtr(fs.pAllocator, &fs.textStart);
  fs.pAllocator = pAllocator;
  fs.textCapacity = 0;

  mTEST_FORMAT("Test", mFormat("Test"));

  const char test2[] = "test2";
  char test3[] = "test3";
  char *test4 = "test4";
  const char *test5 = "test5";

  mTEST_FORMAT("test2test2", mFormat(test2, test2));
  mTEST_FORMAT("test3test3", mFormat(test3, test3));
  mTEST_FORMAT("test4test4", mFormat(test4, test4));
  mTEST_FORMAT("test5test5", mFormat(test5, test5));

  mTEST_FORMAT("This is a test <- 999 or <-9,999 or -12,345", mFormat("This is a test ", mFInt<mFGroupDigits, mFSignBoth, mFMaxDigits<6>>(-123456), " or ", mFInt<mFGroupDigits, mFSignBoth, mFMaxDigits<7>>(-123456), " or ", mFInt<mFGroupDigits, mFSignBoth, mFMaxDigits<7>>(-12345)));

  // Test behaviour with invalid mString.
  {
    // Let's craft an invalid mString!
    mString a = "abcdefgh";
    a.text[3] = '\xFF';
    a.text[4] = '\xFF';

    mTEST_FORMAT("abc", mFormat(mFString(a, mFMaxChars<5>(), mFNoEllipsis())));
    mTEST_FORMAT("ab...", mFormat(mFString(a, mFMaxChars<5>(), mFEllipsis())));

    const mString b = "zabcdefg";
    mTEST_FORMAT("zabcdefg", mFormat(mFString(b, mFEllipsis())));
    mTEST_FORMAT("za...", mFormat(mFString(b, mFMaxChars<5>(), mFEllipsis())));
  }

  mTEST_FORMAT("abcde", mFormat(mFString("abcde", mFMaxChars<5>(), mFEllipsis())));
  mTEST_FORMAT("abcde", mFormat(mFString("abcde", mFMaxChars<5>(), mFNoEllipsis())));
  mTEST_FORMAT("abcde", mFormat(mFString("abcde", mFMaxChars<6>(), mFEllipsis())));
  mTEST_FORMAT("abcde", mFormat(mFString("abcde", mFMaxChars<6>(), mFNoEllipsis())));
  mTEST_FORMAT("ab", mFormat(mFString("abcde", mFMaxChars<2>(), mFNoEllipsis())));
  mTEST_FORMAT("ab", mFormat(mFString("abcde", mFMaxChars<2>(), mFEllipsis())));

  mTEST_FORMAT("ABCD", mFormat(mFString("ABCD", mFMinChars<3>(), mFAlignStringLeft())));
  mTEST_FORMAT("ABCD", mFormat(mFString("ABCD", mFMinChars<3>(), mFAlignStringRight())));
  mTEST_FORMAT("ABCD", mFormat(mFString("ABCD", mFMinChars<3>(), mFAlignStringCenter())));

  mTEST_FORMAT("ABCD", mFormat(mFString("ABCD", mFAlignStringLeft())));
  mTEST_FORMAT("ABCD", mFormat(mFString("ABCD", mFAlignStringRight())));
  mTEST_FORMAT("ABCD", mFormat(mFString("ABCD", mFAlignStringCenter())));

  mTEST_FORMAT("ABCD  ", mFormat(mFString("ABCD", mFMinChars<6>(), mFAlignStringLeft())));
  mTEST_FORMAT("  ABCD", mFormat(mFString("ABCD", mFMinChars<6>(), mFAlignStringRight())));
  mTEST_FORMAT(" ABCD ", mFormat(mFString("ABCD", mFMinChars<6>(), mFAlignStringCenter())));

  mTEST_FORMAT("ABCD    ", mFormat(mFString("ABCD", mFMinChars<8>(), mFAlignStringLeft())));
  mTEST_FORMAT("    ABCD", mFormat(mFString("ABCD", mFMinChars<8>(), mFAlignStringRight())));
  mTEST_FORMAT("  ABCD  ", mFormat(mFString("ABCD", mFMinChars<8>(), mFAlignStringCenter())));

  mString s = "ABCD";

  mTEST_FORMAT("ABCD", mFormat(mFString(s, mFAlignStringLeft())));
  mTEST_FORMAT("ABCD", mFormat(mFString(s, mFAlignStringRight())));
  mTEST_FORMAT("ABCD", mFormat(mFString(s, mFAlignStringCenter())));

  mTEST_FORMAT("ABCD", mFormat(mFString(s, mFMinChars<3>(), mFAlignStringLeft())));
  mTEST_FORMAT("ABCD", mFormat(mFString(s, mFMinChars<3>(), mFAlignStringRight())));
  mTEST_FORMAT("ABCD", mFormat(mFString(s, mFMinChars<3>(), mFAlignStringCenter())));

  mTEST_FORMAT("ABCD  ", mFormat(mFString(s, mFMinChars<6>(), mFAlignStringLeft())));
  mTEST_FORMAT("  ABCD", mFormat(mFString(s, mFMinChars<6>(), mFAlignStringRight())));
  mTEST_FORMAT(" ABCD ", mFormat(mFString(s, mFMinChars<6>(), mFAlignStringCenter())));

  mTEST_FORMAT("ABCD    ", mFormat(mFString(s, mFMinChars<8>(), mFAlignStringLeft())));
  mTEST_FORMAT("    ABCD", mFormat(mFString(s, mFMinChars<8>(), mFAlignStringRight())));
  mTEST_FORMAT("  ABCD  ", mFormat(mFString(s, mFMinChars<8>(), mFAlignStringCenter())));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mFormat, TestIntegers)
{
  mTEST_ALLOCATOR_SETUP();

  mFormatState_ResetCulture();

  mFormatState &fs = mFormat_GetState();

  mDEFER(mAllocator_FreePtr(fs.pAllocator, &fs.textStart); fs.textCapacity = 0; fs.pAllocator = nullptr;);
  mAllocator_FreePtr(fs.pAllocator, &fs.textStart);
  fs.pAllocator = pAllocator;
  fs.textCapacity = 0;

  mTEST_FORMAT("12345", mFormat(12345));
  mTEST_FORMAT("12340", mFormat(12340));
  mTEST_FORMAT("-12345", mFormat(-12345));
  mTEST_FORMAT("-12340", mFormat(-12340));
  mTEST_FORMAT("12345", mFormat((size_t)12345));
  mTEST_FORMAT("12340", mFormat((size_t)12340));
  mTEST_FORMAT("12345", mFormat(mFInt<mFMaxDigits<6>>(12345)));
  mTEST_FORMAT("12345", mFormat(mFInt<mFMaxDigits<5>>(12345)));
  mTEST_FORMAT("67890", mFormat(mFInt<mFMaxDigits<5>>(67890)));
  mTEST_FORMAT(">999", mFormat(mFInt<mFMaxDigits<4>>(12345)));
  mTEST_FORMAT(">+99", mFormat(mFInt<mFMaxDigits<4>, mFSignBoth>(12345)));
  mTEST_FORMAT("<-99", mFormat(mFInt<mFMaxDigits<4>, mFSignBoth>(-12345)));
  mTEST_FORMAT("<0", mFormat(mFInt<mFMaxDigits<4>, mFSignNever>(-12345)));
  mTEST_FORMAT("-12345", mFormat(mFInt<mFSignBoth>(-12345)));
  mTEST_FORMAT("+12345", mFormat(mFInt<mFSignBoth>(12345)));
  mTEST_FORMAT("+12345", mFormat(mFUInt<mFSignBoth>(12345)));
  mTEST_FORMAT(" 12345", mFormat(mFUInt<mFSignNegativeOrFill>(12345)));
  mTEST_FORMAT(" 12345", mFormat(mFInt<mFSignNegativeOrFill>(12345)));
  mTEST_FORMAT("-12345", mFormat(mFInt<mFSignNegativeOrFill>(-12345)));
  mTEST_FORMAT("+12345", mFormat(mFUInt<mFSignBoth, mFMinDigits<6>, mFFillWhitespace, mFSignNotAligned>(12345)));
  mTEST_FORMAT("+ 12345", mFormat(mFUInt<mFSignBoth, mFMinDigits<7>, mFFillWhitespace, mFSignNotAligned>(12345)));
  mTEST_FORMAT(" +12345", mFormat(mFUInt<mFSignBoth, mFMinDigits<7>, mFFillWhitespace, mFSignAligned, mFAlignNumRight>(12345)));
  mTEST_FORMAT("  12345", mFormat(mFUInt<mFMinDigits<7>, mFFillWhitespace, mFSignNotAligned>(12345)));
  mTEST_FORMAT("0012345", mFormat(mFUInt<mFMinDigits<7>, mFFillZeroes, mFSignNotAligned>(12345)));
  mTEST_FORMAT("0F", mFormat(mFUInt<mFMinDigits<2>, mFFillZeroes, mFHex>(0xF)));
  mTEST_FORMAT("0x0F", mFormat("0x", mFUInt<mFMinDigits<2>, mFFillZeroes, mFHex>(0xF)));
  mTEST_FORMAT(" F", mFormat(mFUInt<mFMinDigits<2>, mFFillWhitespace, mFHex>(0xF)));
  mTEST_FORMAT(" F ", mFormat(mFUInt<mFMinDigits<3>, mFFillWhitespace, mFAlignNumCenter, mFHex>(0xF)));
  mTEST_FORMAT("F  ", mFormat(mFUInt<mFMinDigits<3>, mFFillWhitespace, mFAlignNumLeft, mFHex>(0xF)));
  mTEST_FORMAT("  F", mFormat(mFUInt<mFMinDigits<3>, mFFillWhitespace, mFAlignNumRight, mFHex>(0xF)));
  mTEST_FORMAT("  FA50  ", mFormat(mFUInt<mFMinDigits<8>, mFFillWhitespace, mFAlignNumCenter, mFHex>(0xFA50)));
  mTEST_FORMAT("FA50    ", mFormat(mFUInt<mFMinDigits<8>, mFFillWhitespace, mFAlignNumLeft, mFHex>(0xFA50)));
  mTEST_FORMAT("    FA50", mFormat(mFUInt<mFMinDigits<8>, mFFillWhitespace, mFAlignNumRight, mFHex>(0xFA50)));
  mTEST_FORMAT("  fa50  ", mFormat(mFUInt<mFMinDigits<8>, mFFillWhitespace, mFAlignNumCenter, mFHex, mFHexLowercase>(0xFA50)));
  mTEST_FORMAT("fa50    ", mFormat(mFUInt<mFMinDigits<8>, mFFillWhitespace, mFAlignNumLeft, mFHex, mFHexLowercase>(0xFA50)));
  mTEST_FORMAT("    fa50", mFormat(mFUInt<mFMinDigits<8>, mFFillWhitespace, mFAlignNumRight, mFHex, mFHexLowercase>(0xFA50)));
  mTEST_FORMAT("F", mFormat(mFUInt<mFHex>(0xF)));
  mTEST_FORMAT("FC01", mFormat(mFInt<mFHex>(0xFC01)));
  mTEST_FORMAT("10101100", mFormat(mFInt<mFBinary>(0b10101100)));
  mTEST_FORMAT("12345", mFormat(mFUInt<mFHex>(0x12345)));
  mTEST_FORMAT("345", mFormat(mFUInt<mFHex, mFMaxDigits<3>>(0x12345)));
  mTEST_FORMAT("110", mFormat(mFUInt<mFBinary, mFMaxDigits<3>>(0b1110)));
  mTEST_FORMAT("1110", mFormat(mFUInt<mFBinary>(0b1110)));
  mTEST_FORMAT("111000", mFormat(mFUInt<mFBinary>(0b111000)));
  mTEST_FORMAT("10101100", mFormat(mFUInt<mFBinary>(0b10101100)));

  mTEST_FORMAT("255", mFormat((uint8_t)255));
  mTEST_FORMAT("-128", mFormat((int8_t)-128));
  mTEST_FORMAT("127", mFormat((int8_t)127));
  mTEST_FORMAT("65535", mFormat((uint16_t)65535));
  mTEST_FORMAT("-12345", mFormat((int16_t)-12345));
  mTEST_FORMAT("23456", mFormat((int16_t)23456));
  mTEST_FORMAT("3000000000", mFormat((uint32_t)3000000000));
  mTEST_FORMAT("-1000000000", mFormat((int32_t)-1000000000));
  mTEST_FORMAT("1000000000", mFormat((int32_t)1000000000));
  mTEST_FORMAT("3000000000000", mFormat(3000000000000ULL));
  mTEST_FORMAT("18446744073709551615", mFormat(UINT64_MAX));
  mTEST_FORMAT("-10000000000000", mFormat(-10000000000000LL));
  mTEST_FORMAT("10000000000000", mFormat(10000000000000LL));
  mTEST_FORMAT("9223372036854775807", mFormat(INT64_MAX));
  mTEST_FORMAT("-9223372036854775808", mFormat(INT64_MIN));

  mTEST_FORMAT("1234", mFormat(mFUInt<mFAlignNumLeft>(1234)));
  mTEST_FORMAT("1234", mFormat(mFUInt<mFAlignNumRight>(1234)));
  mTEST_FORMAT("1234", mFormat(mFUInt<mFAlignNumCenter>(1234)));

  mTEST_FORMAT(">99", mFormat(mFUInt<mFAlignNumLeft, mFMaxDigits<3>>(1234)));
  mTEST_FORMAT(">99", mFormat(mFUInt<mFAlignNumRight, mFMaxDigits<3>>(1234)));
  mTEST_FORMAT(">99", mFormat(mFUInt<mFAlignNumCenter, mFMaxDigits<3>>(1234)));

  mTEST_FORMAT("1234", mFormat(mFUInt<mFAlignNumLeft, mFMinDigits<3>>(1234)));
  mTEST_FORMAT("1234", mFormat(mFUInt<mFAlignNumRight, mFMinDigits<3>>(1234)));
  mTEST_FORMAT("1234", mFormat(mFUInt<mFAlignNumCenter, mFMinDigits<3>>(1234)));

  mTEST_FORMAT("1234  ", mFormat(mFUInt<mFAlignNumLeft, mFMinDigits<6>>(1234)));
  mTEST_FORMAT("  1234", mFormat(mFUInt<mFAlignNumRight, mFMinDigits<6>>(1234)));
  mTEST_FORMAT(" 1234 ", mFormat(mFUInt<mFAlignNumCenter, mFMinDigits<6>>(1234)));

  mTEST_FORMAT("+1234    ", mFormat(mFUInt<mFSignAligned, mFSignBoth, mFAlignNumLeft, mFMinDigits<9>>(1234)));
  mTEST_FORMAT("    +1234", mFormat(mFUInt<mFSignAligned, mFSignBoth, mFAlignNumRight, mFMinDigits<9>>(1234)));
  mTEST_FORMAT("  +1234  ", mFormat(mFUInt<mFSignAligned, mFSignBoth, mFAlignNumCenter, mFMinDigits<9>>(1234)));

  mTEST_FORMAT("+1234    ", mFormat(mFUInt<mFSignNotAligned, mFSignBoth, mFAlignNumLeft, mFMinDigits<9>>(1234)));
  mTEST_FORMAT("+    1234", mFormat(mFUInt<mFSignNotAligned, mFSignBoth, mFAlignNumRight, mFMinDigits<9>>(1234)));
  mTEST_FORMAT("+  1234  ", mFormat(mFUInt<mFSignNotAligned, mFSignBoth, mFAlignNumCenter, mFMinDigits<9>>(1234)));

  mTEST_FORMAT("123,456,789", mFormat(mFUInt<mFGroupDigits>(123456789)));
  mTEST_FORMAT("1,234,567", mFormat(mFUInt<mFGroupDigits>(1234567)));
  mTEST_FORMAT("12,345", mFormat(mFUInt<mFGroupDigits>(12345)));
  mTEST_FORMAT("1,234", mFormat(mFUInt<mFGroupDigits>(1234)));
  mTEST_FORMAT("123", mFormat(mFUInt<mFGroupDigits>(123)));
  mTEST_FORMAT("123,456,789", mFormat(mFInt<mFGroupDigits>(123456789)));
  mTEST_FORMAT("1,234,567", mFormat(mFInt<mFGroupDigits>(1234567)));
  mTEST_FORMAT("12,345", mFormat(mFInt<mFGroupDigits>(12345)));
  mTEST_FORMAT("1,234", mFormat(mFInt<mFGroupDigits>(1234)));
  mTEST_FORMAT("123", mFormat(mFInt<mFGroupDigits>(123)));
  mTEST_FORMAT("-123,456,789", mFormat(mFInt<mFGroupDigits>(-123456789)));
  mTEST_FORMAT("-1,234,567", mFormat(mFInt<mFGroupDigits>(-1234567)));
  mTEST_FORMAT("-12,345", mFormat(mFInt<mFGroupDigits>(-12345)));
  mTEST_FORMAT("-1,234", mFormat(mFInt<mFGroupDigits>(-1234)));
  mTEST_FORMAT("-123", mFormat(mFInt<mFGroupDigits>(-123)));

  mTEST_FORMAT("-1,234,567,890", mFormat(mFInt<mFGroupDigits, mFMaxDigits<15>, mFSignBoth>(-1234567890)));
  mTEST_FORMAT("-1,234,567,890", mFormat(mFInt<mFGroupDigits, mFMaxDigits<14>, mFSignBoth>(-1234567890)));
  mTEST_FORMAT("<-999,999,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<13>, mFSignBoth>(-1234567890)));
  mTEST_FORMAT("<-99,999,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<12>, mFSignBoth>(-1234567890)));
  mTEST_FORMAT("<-9,999,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<11>, mFSignBoth>(-1234567890)));
  mTEST_FORMAT("<- 999,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<10>, mFSignBoth>(-1234567890)));
  mTEST_FORMAT("<-999,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<9>, mFSignBoth>(-1234567890)));
  mTEST_FORMAT("<-99,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<8>, mFSignBoth>(-1234567890)));
  mTEST_FORMAT("<-999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<5>, mFSignBoth>(-1234567890)));
  mTEST_FORMAT("+1,234,567,890", mFormat(mFInt<mFGroupDigits, mFMaxDigits<15>, mFSignBoth>(1234567890)));
  mTEST_FORMAT("+1,234,567,890", mFormat(mFInt<mFGroupDigits, mFMaxDigits<14>, mFSignBoth>(1234567890)));
  mTEST_FORMAT(">+999,999,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<13>, mFSignBoth>(1234567890)));
  mTEST_FORMAT(">+99,999,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<12>, mFSignBoth>(1234567890)));
  mTEST_FORMAT(">+9,999,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<11>, mFSignBoth>(1234567890)));
  mTEST_FORMAT(">+ 999,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<10>, mFSignBoth>(1234567890)));
  mTEST_FORMAT(">+999,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<9>, mFSignBoth>(1234567890)));
  mTEST_FORMAT(">+99,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<8>, mFSignBoth>(1234567890)));
  mTEST_FORMAT("1,234,567,890", mFormat(mFInt<mFGroupDigits, mFMaxDigits<15>, mFSignNegativeOnly>(1234567890)));
  mTEST_FORMAT("1,234,567,890", mFormat(mFInt<mFGroupDigits, mFMaxDigits<14>, mFSignNegativeOnly>(1234567890)));
  mTEST_FORMAT("1,234,567,890", mFormat(mFInt<mFGroupDigits, mFMaxDigits<13>, mFSignNegativeOnly>(1234567890)));
  mTEST_FORMAT(">999,999,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<12>, mFSignNegativeOnly>(1234567890)));
  mTEST_FORMAT(">99,999,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<11>, mFSignNegativeOnly>(1234567890)));
  mTEST_FORMAT(">9,999,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<10>, mFSignNegativeOnly>(1234567890)));
  mTEST_FORMAT("> 999,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<9>, mFSignNegativeOnly>(1234567890)));
  mTEST_FORMAT(">999,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<8>, mFSignNegativeOnly>(1234567890)));
  mTEST_FORMAT(">99,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<7>, mFSignNegativeOnly>(1234567890)));
  mTEST_FORMAT(">9,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<6>, mFSignNegativeOnly>(1234567890)));
  mTEST_FORMAT("> 999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<5>, mFSignNegativeOnly>(1234567890)));

  mTEST_FORMAT("        9", mFormat(mFInt<mFGroupDigits, mFMinDigits<9>, mFSignNegativeOnly>(9)));
  mTEST_FORMAT("-       9", mFormat(mFInt<mFGroupDigits, mFMinDigits<9>, mFSignNegativeOnly, mFSignNotAligned>(-9)));
  mTEST_FORMAT("       -9", mFormat(mFInt<mFGroupDigits, mFMinDigits<9>, mFSignNegativeOnly, mFSignAligned>(-9)));
  mTEST_FORMAT("0,000,009", mFormat(mFInt<mFGroupDigits, mFMinDigits<9>, mFFillZeroes, mFSignNegativeOnly>(9)));
  mTEST_FORMAT("- 000,009", mFormat(mFInt<mFGroupDigits, mFMinDigits<9>, mFFillZeroes, mFSignNegativeOnly>(-9)));
  mTEST_FORMAT(" 000,009", mFormat(mFInt<mFGroupDigits, mFMinDigits<8>, mFFillZeroes, mFSignNegativeOnly>(9)));
  mTEST_FORMAT("-000,009", mFormat(mFInt<mFGroupDigits, mFMinDigits<8>, mFFillZeroes, mFSignNegativeOnly>(-9)));
  mTEST_FORMAT("+000,009", mFormat(mFInt<mFGroupDigits, mFMinDigits<8>, mFFillZeroes, mFSignBoth>(+9)));
  mTEST_FORMAT(" 00,009", mFormat(mFInt<mFGroupDigits, mFMinDigits<7>, mFFillZeroes, mFSignNegativeOrFill>(9)));
  mTEST_FORMAT("000,009", mFormat(mFInt<mFGroupDigits, mFMinDigits<7>, mFFillZeroes, mFSignNegativeOnly>(9)));
  mTEST_FORMAT("-00,009", mFormat(mFInt<mFGroupDigits, mFMinDigits<7>, mFFillZeroes, mFSignNegativeOnly>(-9)));
  mTEST_FORMAT("+00,009", mFormat(mFInt<mFGroupDigits, mFMinDigits<7>, mFFillZeroes, mFSignBoth>(+9)));
  mTEST_FORMAT("00,009", mFormat(mFInt<mFGroupDigits, mFMinDigits<6>, mFFillZeroes, mFSignNegativeOnly>(9)));
  mTEST_FORMAT("-0,009", mFormat(mFInt<mFGroupDigits, mFMinDigits<6>, mFFillZeroes, mFSignNegativeOnly>(-9)));
  mTEST_FORMAT("+0,009", mFormat(mFInt<mFGroupDigits, mFMinDigits<6>, mFFillZeroes, mFSignBoth>(+9)));
  mTEST_FORMAT("0,009", mFormat(mFInt<mFGroupDigits, mFMinDigits<5>, mFFillZeroes, mFSignNegativeOnly>(9)));
  mTEST_FORMAT("- 009", mFormat(mFInt<mFGroupDigits, mFMinDigits<5>, mFFillZeroes, mFSignNegativeOnly>(-9)));
  mTEST_FORMAT("+ 009", mFormat(mFInt<mFGroupDigits, mFMinDigits<5>, mFFillZeroes, mFSignBoth>(+9)));
  mTEST_FORMAT("+1,234,567", mFormat(mFInt<mFGroupDigits, mFMinDigits<10>, mFAlignNumLeft, mFSignBoth, mFSignNotAligned>(1234567)));
  mTEST_FORMAT("+1,234,567", mFormat(mFInt<mFGroupDigits, mFAlignNumLeft, mFSignBoth, mFSignNotAligned>(1234567)));
  mTEST_FORMAT("+1,234,567", mFormat(mFInt<mFGroupDigits, mFAlignNumRight, mFSignBoth, mFSignNotAligned>(1234567)));
  mTEST_FORMAT("+1,234,567", mFormat(mFInt<mFGroupDigits, mFAlignNumCenter, mFSignBoth, mFSignNotAligned>(1234567)));
  mTEST_FORMAT("+1,234,567  ", mFormat(mFInt<mFGroupDigits, mFMinDigits<12>, mFAlignNumLeft, mFSignBoth, mFSignNotAligned>(1234567)));
  mTEST_FORMAT("+1,234,567  ", mFormat(mFInt<mFGroupDigits, mFMinDigits<12>, mFAlignNumLeft, mFSignBoth, mFSignAligned>(1234567)));
  mTEST_FORMAT("1,234,567   ", mFormat(mFInt<mFGroupDigits, mFMinDigits<12>, mFAlignNumLeft, mFSignNegativeOnly, mFSignAligned>(1234567)));
  mTEST_FORMAT(" 1,234,567  ", mFormat(mFInt<mFGroupDigits, mFMinDigits<12>, mFAlignNumLeft, mFSignNegativeOrFill, mFSignAligned>(1234567)));
  mTEST_FORMAT("-1,234,567  ", mFormat(mFInt<mFGroupDigits, mFMinDigits<12>, mFAlignNumLeft, mFSignNegativeOrFill, mFSignAligned>(-1234567)));
  mTEST_FORMAT(" +1,234,567 ", mFormat(mFInt<mFGroupDigits, mFMinDigits<12>, mFAlignNumCenter, mFSignBoth, mFSignAligned>(1234567)));
  mTEST_FORMAT("+ 1,234,567 ", mFormat(mFInt<mFGroupDigits, mFMinDigits<12>, mFAlignNumCenter, mFSignBoth, mFSignNotAligned>(1234567)));

  fs.digitGroupingOption = mFDGO_TenThousand;

  mTEST_FORMAT("1,2345,6789", mFormat(mFUInt<mFGroupDigits>(123456789)));
  mTEST_FORMAT("1234,5678", mFormat(mFUInt<mFGroupDigits>(12345678)));
  mTEST_FORMAT("123,4567", mFormat(mFUInt<mFGroupDigits>(1234567)));
  mTEST_FORMAT("1,2345", mFormat(mFUInt<mFGroupDigits>(12345)));
  mTEST_FORMAT("1234", mFormat(mFUInt<mFGroupDigits>(1234)));
  mTEST_FORMAT("123", mFormat(mFUInt<mFGroupDigits>(123)));
  mTEST_FORMAT("1,2345,6789", mFormat(mFInt<mFGroupDigits>(123456789)));
  mTEST_FORMAT("1234,5678", mFormat(mFInt<mFGroupDigits>(12345678)));
  mTEST_FORMAT("123,4567", mFormat(mFInt<mFGroupDigits>(1234567)));
  mTEST_FORMAT("1,2345", mFormat(mFInt<mFGroupDigits>(12345)));
  mTEST_FORMAT("1234", mFormat(mFInt<mFGroupDigits>(1234)));
  mTEST_FORMAT("123", mFormat(mFInt<mFGroupDigits>(123)));
  mTEST_FORMAT("-1,2345,6789", mFormat(mFInt<mFGroupDigits>(-123456789)));
  mTEST_FORMAT("-1234,5678", mFormat(mFInt<mFGroupDigits>(-12345678)));
  mTEST_FORMAT("-123,4567", mFormat(mFInt<mFGroupDigits>(-1234567)));
  mTEST_FORMAT("-1,2345", mFormat(mFInt<mFGroupDigits>(-12345)));
  mTEST_FORMAT("-1234", mFormat(mFInt<mFGroupDigits>(-1234)));
  mTEST_FORMAT("-123", mFormat(mFInt<mFGroupDigits>(-123)));

  mTEST_FORMAT("-12,3456,7890", mFormat(mFInt<mFGroupDigits, mFMaxDigits<14>, mFSignBoth>(-1234567890)));
  mTEST_FORMAT("-12,3456,7890", mFormat(mFInt<mFGroupDigits, mFMaxDigits<13>, mFSignBoth>(-1234567890)));
  mTEST_FORMAT("<- 9999,9999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<12>, mFSignBoth>(-1234567890)));
  mTEST_FORMAT("<-9999,9999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<11>, mFSignBoth>(-1234567890)));
  mTEST_FORMAT("<-999,9999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<10>, mFSignBoth>(-1234567890)));
  mTEST_FORMAT("<-99,9999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<9>, mFSignBoth>(-1234567890)));
  mTEST_FORMAT("<-9,9999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<8>, mFSignBoth>(-1234567890)));
  mTEST_FORMAT("<- 9999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<7>, mFSignBoth>(-1234567890)));
  mTEST_FORMAT("<-9999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<6>, mFSignBoth>(-1234567890)));
  mTEST_FORMAT("+12,3456,7890", mFormat(mFInt<mFGroupDigits, mFMaxDigits<14>, mFSignBoth>(1234567890)));
  mTEST_FORMAT("+12,3456,7890", mFormat(mFInt<mFGroupDigits, mFMaxDigits<13>, mFSignBoth>(1234567890)));
  mTEST_FORMAT(">+ 9999,9999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<12>, mFSignBoth>(1234567890)));
  mTEST_FORMAT(">+9999,9999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<11>, mFSignBoth>(1234567890)));
  mTEST_FORMAT(">+999,9999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<10>, mFSignBoth>(1234567890)));
  mTEST_FORMAT(">+99,9999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<9>, mFSignBoth>(1234567890)));
  mTEST_FORMAT(">+9,9999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<8>, mFSignBoth>(1234567890)));
  mTEST_FORMAT(">+ 9999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<7>, mFSignBoth>(1234567890)));
  mTEST_FORMAT(">+9999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<6>, mFSignBoth>(1234567890)));
  mTEST_FORMAT("12,3456,7890", mFormat(mFInt<mFGroupDigits, mFMaxDigits<13>, mFSignNegativeOnly>(1234567890)));
  mTEST_FORMAT("12,3456,7890", mFormat(mFInt<mFGroupDigits, mFMaxDigits<12>, mFSignNegativeOnly>(1234567890)));
  mTEST_FORMAT("> 9999,9999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<11>, mFSignNegativeOnly>(1234567890)));
  mTEST_FORMAT(">9999,9999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<10>, mFSignNegativeOnly>(1234567890)));
  mTEST_FORMAT(">999,9999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<9>, mFSignNegativeOnly>(1234567890)));
  mTEST_FORMAT(">99,9999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<8>, mFSignNegativeOnly>(1234567890)));
  mTEST_FORMAT(">9,9999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<7>, mFSignNegativeOnly>(1234567890)));
  mTEST_FORMAT("> 9999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<6>, mFSignNegativeOnly>(1234567890)));
  mTEST_FORMAT(">9999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<5>, mFSignNegativeOnly>(1234567890)));
  mTEST_FORMAT(">999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<4>, mFSignNegativeOnly>(1234567890)));

  mTEST_FORMAT("        9", mFormat(mFInt<mFGroupDigits, mFMinDigits<9>, mFSignNegativeOnly>(9)));
  mTEST_FORMAT("-       9", mFormat(mFInt<mFGroupDigits, mFMinDigits<9>, mFSignNegativeOnly, mFSignNotAligned>(-9)));
  mTEST_FORMAT("       -9", mFormat(mFInt<mFGroupDigits, mFMinDigits<9>, mFSignNegativeOnly, mFSignAligned>(-9)));
  mTEST_FORMAT("0000,0009", mFormat(mFInt<mFGroupDigits, mFMinDigits<9>, mFFillZeroes, mFSignNegativeOnly>(9)));
  mTEST_FORMAT("-000,0009", mFormat(mFInt<mFGroupDigits, mFMinDigits<9>, mFFillZeroes, mFSignNegativeOnly>(-9)));
  mTEST_FORMAT("000,0009", mFormat(mFInt<mFGroupDigits, mFMinDigits<8>, mFFillZeroes, mFSignNegativeOnly>(9)));
  mTEST_FORMAT("-00,0009", mFormat(mFInt<mFGroupDigits, mFMinDigits<8>, mFFillZeroes, mFSignNegativeOnly>(-9)));
  mTEST_FORMAT("+00,0009", mFormat(mFInt<mFGroupDigits, mFMinDigits<8>, mFFillZeroes, mFSignBoth>(+9)));
  mTEST_FORMAT("00,0009", mFormat(mFInt<mFGroupDigits, mFMinDigits<7>, mFFillZeroes, mFSignNegativeOnly>(9)));
  mTEST_FORMAT(" 0,0009", mFormat(mFInt<mFGroupDigits, mFMinDigits<7>, mFFillZeroes, mFSignNegativeOrFill>(9)));
  mTEST_FORMAT("-0,0009", mFormat(mFInt<mFGroupDigits, mFMinDigits<7>, mFFillZeroes, mFSignNegativeOnly>(-9)));
  mTEST_FORMAT("+0,0009", mFormat(mFInt<mFGroupDigits, mFMinDigits<7>, mFFillZeroes, mFSignBoth>(+9)));
  mTEST_FORMAT("0,0009", mFormat(mFInt<mFGroupDigits, mFMinDigits<6>, mFFillZeroes, mFSignNegativeOnly>(9)));
  mTEST_FORMAT("- 0009", mFormat(mFInt<mFGroupDigits, mFMinDigits<6>, mFFillZeroes, mFSignNegativeOnly>(-9)));
  mTEST_FORMAT("+ 0009", mFormat(mFInt<mFGroupDigits, mFMinDigits<6>, mFFillZeroes, mFSignBoth>(+9)));
  mTEST_FORMAT(" 0009", mFormat(mFInt<mFGroupDigits, mFMinDigits<5>, mFFillZeroes, mFSignNegativeOnly>(9)));
  mTEST_FORMAT("-0009", mFormat(mFInt<mFGroupDigits, mFMinDigits<5>, mFFillZeroes, mFSignNegativeOnly>(-9)));
  mTEST_FORMAT("+0009", mFormat(mFInt<mFGroupDigits, mFMinDigits<5>, mFFillZeroes, mFSignBoth>(+9)));
  mTEST_FORMAT("+123,4567", mFormat(mFInt<mFGroupDigits, mFMinDigits<9>, mFAlignNumLeft, mFSignBoth, mFSignNotAligned>(1234567)));
  mTEST_FORMAT("+123,4567", mFormat(mFInt<mFGroupDigits, mFAlignNumLeft, mFSignBoth, mFSignNotAligned>(1234567)));
  mTEST_FORMAT("+123,4567", mFormat(mFInt<mFGroupDigits, mFAlignNumRight, mFSignBoth, mFSignNotAligned>(1234567)));
  mTEST_FORMAT("+123,4567", mFormat(mFInt<mFGroupDigits, mFAlignNumCenter, mFSignBoth, mFSignNotAligned>(1234567)));
  mTEST_FORMAT("+123,4567   ", mFormat(mFInt<mFGroupDigits, mFMinDigits<12>, mFAlignNumLeft, mFSignBoth, mFSignNotAligned>(1234567)));
  mTEST_FORMAT("+123,4567   ", mFormat(mFInt<mFGroupDigits, mFMinDigits<12>, mFAlignNumLeft, mFSignBoth, mFSignAligned>(1234567)));
  mTEST_FORMAT("123,4567    ", mFormat(mFInt<mFGroupDigits, mFMinDigits<12>, mFAlignNumLeft, mFSignNegativeOnly, mFSignAligned>(1234567)));
  mTEST_FORMAT(" 123,4567   ", mFormat(mFInt<mFGroupDigits, mFMinDigits<12>, mFAlignNumLeft, mFSignNegativeOrFill, mFSignAligned>(1234567)));
  mTEST_FORMAT("-123,4567   ", mFormat(mFInt<mFGroupDigits, mFMinDigits<12>, mFAlignNumLeft, mFSignNegativeOrFill, mFSignAligned>(-1234567)));
  mTEST_FORMAT(" +123,4567 ", mFormat(mFInt<mFGroupDigits, mFMinDigits<11>, mFAlignNumCenter, mFSignBoth, mFSignAligned>(1234567)));
  mTEST_FORMAT("+ 123,4567 ", mFormat(mFInt<mFGroupDigits, mFMinDigits<11>, mFAlignNumCenter, mFSignBoth, mFSignNotAligned>(1234567)));

  fs.digitGroupingOption = mFDGO_Indian;

  mTEST_FORMAT("12,34,56,789", mFormat(mFUInt<mFGroupDigits>(123456789)));
  mTEST_FORMAT("1,23,45,678", mFormat(mFUInt<mFGroupDigits>(12345678)));
  mTEST_FORMAT("12,34,567", mFormat(mFUInt<mFGroupDigits>(1234567)));
  mTEST_FORMAT("12,345", mFormat(mFUInt<mFGroupDigits>(12345)));
  mTEST_FORMAT("1,234", mFormat(mFUInt<mFGroupDigits>(1234)));
  mTEST_FORMAT("123", mFormat(mFUInt<mFGroupDigits>(123)));
  mTEST_FORMAT("12,34,56,789", mFormat(mFInt<mFGroupDigits>(123456789)));
  mTEST_FORMAT("1,23,45,678", mFormat(mFInt<mFGroupDigits>(12345678)));
  mTEST_FORMAT("12,34,567", mFormat(mFInt<mFGroupDigits>(1234567)));
  mTEST_FORMAT("12,345", mFormat(mFInt<mFGroupDigits>(12345)));
  mTEST_FORMAT("1,234", mFormat(mFInt<mFGroupDigits>(1234)));
  mTEST_FORMAT("123", mFormat(mFInt<mFGroupDigits>(123)));
  mTEST_FORMAT("-12,34,56,789", mFormat(mFInt<mFGroupDigits>(-123456789)));
  mTEST_FORMAT("-1,23,45,678", mFormat(mFInt<mFGroupDigits>(-12345678)));
  mTEST_FORMAT("-12,34,567", mFormat(mFInt<mFGroupDigits>(-1234567)));
  mTEST_FORMAT("-12,345", mFormat(mFInt<mFGroupDigits>(-12345)));
  mTEST_FORMAT("-1,234", mFormat(mFInt<mFGroupDigits>(-1234)));
  mTEST_FORMAT("-123", mFormat(mFInt<mFGroupDigits>(-123)));

  mTEST_FORMAT("-1,23,45,67,890", mFormat(mFInt<mFGroupDigits, mFMaxDigits<16>, mFSignBoth>(-1234567890)));
  mTEST_FORMAT("-1,23,45,67,890", mFormat(mFInt<mFGroupDigits, mFMaxDigits<15>, mFSignBoth>(-1234567890)));
  mTEST_FORMAT("<-99,99,99,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<14>, mFSignBoth>(-1234567890)));
  mTEST_FORMAT("<-9,99,99,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<13>, mFSignBoth>(-1234567890)));
  mTEST_FORMAT("<- 99,99,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<12>, mFSignBoth>(-1234567890)));
  mTEST_FORMAT("<-99,99,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<11>, mFSignBoth>(-1234567890)));
  mTEST_FORMAT("<-9,99,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<10>, mFSignBoth>(-1234567890)));
  mTEST_FORMAT("<- 99,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<9>, mFSignBoth>(-1234567890)));
  mTEST_FORMAT("<-99,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<8>, mFSignBoth>(-1234567890)));
  mTEST_FORMAT("<-9,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<7>, mFSignBoth>(-1234567890)));
  mTEST_FORMAT("<- 999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<6>, mFSignBoth>(-1234567890)));
  mTEST_FORMAT("<-999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<5>, mFSignBoth>(-1234567890)));
  mTEST_FORMAT("<-99", mFormat(mFInt<mFGroupDigits, mFMaxDigits<4>, mFSignBoth>(-1234567890)));
  mTEST_FORMAT("+1,23,45,67,890", mFormat(mFInt<mFGroupDigits, mFMaxDigits<16>, mFSignBoth>(1234567890)));
  mTEST_FORMAT("+1,23,45,67,890", mFormat(mFInt<mFGroupDigits, mFMaxDigits<15>, mFSignBoth>(1234567890)));
  mTEST_FORMAT(">+99,99,99,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<14>, mFSignBoth>(1234567890)));
  mTEST_FORMAT(">+9,99,99,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<13>, mFSignBoth>(1234567890)));
  mTEST_FORMAT(">+ 99,99,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<12>, mFSignBoth>(1234567890)));
  mTEST_FORMAT(">+99,99,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<11>, mFSignBoth>(1234567890)));
  mTEST_FORMAT(">+9,99,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<10>, mFSignBoth>(1234567890)));
  mTEST_FORMAT(">+ 99,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<9>, mFSignBoth>(1234567890)));
  mTEST_FORMAT(">+99,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<8>, mFSignBoth>(1234567890)));
  mTEST_FORMAT(">+9,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<7>, mFSignBoth>(1234567890)));
  mTEST_FORMAT(">+ 999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<6>, mFSignBoth>(1234567890)));
  mTEST_FORMAT(">+999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<5>, mFSignBoth>(1234567890)));
  mTEST_FORMAT(">+99", mFormat(mFInt<mFGroupDigits, mFMaxDigits<4>, mFSignBoth>(1234567890)));
  mTEST_FORMAT("1,23,45,67,890", mFormat(mFInt<mFGroupDigits, mFMaxDigits<15>, mFSignNegativeOnly>(1234567890)));
  mTEST_FORMAT("1,23,45,67,890", mFormat(mFInt<mFGroupDigits, mFMaxDigits<14>, mFSignNegativeOnly>(1234567890)));
  mTEST_FORMAT(">99,99,99,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<13>, mFSignNegativeOnly>(1234567890)));
  mTEST_FORMAT(">9,99,99,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<12>, mFSignNegativeOnly>(1234567890)));
  mTEST_FORMAT("> 99,99,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<11>, mFSignNegativeOnly>(1234567890)));
  mTEST_FORMAT(">99,99,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<10>, mFSignNegativeOnly>(1234567890)));
  mTEST_FORMAT(">9,99,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<9>, mFSignNegativeOnly>(1234567890)));
  mTEST_FORMAT("> 99,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<8>, mFSignNegativeOnly>(1234567890)));
  mTEST_FORMAT(">99,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<7>, mFSignNegativeOnly>(1234567890)));
  mTEST_FORMAT(">9,999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<6>, mFSignNegativeOnly>(1234567890)));
  mTEST_FORMAT("> 999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<5>, mFSignNegativeOnly>(1234567890)));
  mTEST_FORMAT(">999", mFormat(mFInt<mFGroupDigits, mFMaxDigits<4>, mFSignNegativeOnly>(1234567890)));
  mTEST_FORMAT(">99", mFormat(mFInt<mFGroupDigits, mFMaxDigits<3>, mFSignNegativeOnly>(1234567890)));

  mTEST_FORMAT("        9", mFormat(mFInt<mFGroupDigits, mFMinDigits<9>, mFSignNegativeOnly>(9)));
  mTEST_FORMAT("-       9", mFormat(mFInt<mFGroupDigits, mFMinDigits<9>, mFSignNegativeOnly, mFSignNotAligned>(-9)));
  mTEST_FORMAT("       -9", mFormat(mFInt<mFGroupDigits, mFMinDigits<9>, mFSignNegativeOnly, mFSignAligned>(-9)));
  mTEST_FORMAT("00,00,009", mFormat(mFInt<mFGroupDigits, mFMinDigits<9>, mFFillZeroes, mFSignNegativeOnly>(9)));
  mTEST_FORMAT("-0,00,009", mFormat(mFInt<mFGroupDigits, mFMinDigits<9>, mFFillZeroes, mFSignNegativeOnly>(-9)));
  mTEST_FORMAT("0,00,009", mFormat(mFInt<mFGroupDigits, mFMinDigits<8>, mFFillZeroes, mFSignNegativeOnly>(9)));
  mTEST_FORMAT("- 00,009", mFormat(mFInt<mFGroupDigits, mFMinDigits<8>, mFFillZeroes, mFSignNegativeOnly>(-9)));
  mTEST_FORMAT("+ 00,009", mFormat(mFInt<mFGroupDigits, mFMinDigits<8>, mFFillZeroes, mFSignBoth>(+9)));
  mTEST_FORMAT(" 00,009", mFormat(mFInt<mFGroupDigits, mFMinDigits<7>, mFFillZeroes, mFSignNegativeOnly>(9)));
  mTEST_FORMAT("-00,009", mFormat(mFInt<mFGroupDigits, mFMinDigits<7>, mFFillZeroes, mFSignNegativeOnly>(-9)));
  mTEST_FORMAT("+00,009", mFormat(mFInt<mFGroupDigits, mFMinDigits<7>, mFFillZeroes, mFSignBoth>(+9)));
  mTEST_FORMAT("00,009", mFormat(mFInt<mFGroupDigits, mFMinDigits<6>, mFFillZeroes, mFSignNegativeOnly>(9)));
  mTEST_FORMAT("-0,009", mFormat(mFInt<mFGroupDigits, mFMinDigits<6>, mFFillZeroes, mFSignNegativeOnly>(-9)));
  mTEST_FORMAT("+0,009", mFormat(mFInt<mFGroupDigits, mFMinDigits<6>, mFFillZeroes, mFSignBoth>(+9)));
  mTEST_FORMAT("0,009", mFormat(mFInt<mFGroupDigits, mFMinDigits<5>, mFFillZeroes, mFSignNegativeOnly>(9)));
  mTEST_FORMAT("- 009", mFormat(mFInt<mFGroupDigits, mFMinDigits<5>, mFFillZeroes, mFSignNegativeOnly>(-9)));
  mTEST_FORMAT("+ 009", mFormat(mFInt<mFGroupDigits, mFMinDigits<5>, mFFillZeroes, mFSignBoth>(+9)));
  mTEST_FORMAT("+12,34,567", mFormat(mFInt<mFGroupDigits, mFMinDigits<10>, mFAlignNumLeft, mFSignBoth, mFSignNotAligned>(1234567)));
  mTEST_FORMAT("+12,34,567", mFormat(mFInt<mFGroupDigits, mFAlignNumLeft, mFSignBoth, mFSignNotAligned>(1234567)));
  mTEST_FORMAT("+12,34,567", mFormat(mFInt<mFGroupDigits, mFAlignNumRight, mFSignBoth, mFSignNotAligned>(1234567)));
  mTEST_FORMAT("+12,34,567", mFormat(mFInt<mFGroupDigits, mFAlignNumCenter, mFSignBoth, mFSignNotAligned>(1234567)));
  mTEST_FORMAT("+12,34,567  ", mFormat(mFInt<mFGroupDigits, mFMinDigits<12>, mFAlignNumLeft, mFSignBoth, mFSignNotAligned>(1234567)));
  mTEST_FORMAT("+12,34,567  ", mFormat(mFInt<mFGroupDigits, mFMinDigits<12>, mFAlignNumLeft, mFSignBoth, mFSignAligned>(1234567)));
  mTEST_FORMAT("12,34,567   ", mFormat(mFInt<mFGroupDigits, mFMinDigits<12>, mFAlignNumLeft, mFSignNegativeOnly, mFSignAligned>(1234567)));
  mTEST_FORMAT(" 12,34,567  ", mFormat(mFInt<mFGroupDigits, mFMinDigits<12>, mFAlignNumLeft, mFSignNegativeOrFill, mFSignAligned>(1234567)));
  mTEST_FORMAT("-12,34,567  ", mFormat(mFInt<mFGroupDigits, mFMinDigits<12>, mFAlignNumLeft, mFSignNegativeOrFill, mFSignAligned>(-1234567)));
  mTEST_FORMAT(" +12,34,567 ", mFormat(mFInt<mFGroupDigits, mFMinDigits<12>, mFAlignNumCenter, mFSignBoth, mFSignAligned>(1234567)));
  mTEST_FORMAT("+ 12,34,567 ", mFormat(mFInt<mFGroupDigits, mFMinDigits<12>, mFAlignNumCenter, mFSignBoth, mFSignNotAligned>(1234567)));

  mFormatState_ResetCulture();

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mFormat, TestBools)
{
  mTEST_ALLOCATOR_SETUP();

  mFormatState_ResetCulture();

  mFormatState &fs = mFormat_GetState();

  mDEFER(mAllocator_FreePtr(fs.pAllocator, &fs.textStart); fs.textCapacity = 0; fs.pAllocator = nullptr;);
  mAllocator_FreePtr(fs.pAllocator, &fs.textStart);
  fs.pAllocator = pAllocator;
  fs.textCapacity = 0;

  mTEST_FORMAT("true", mFormat(true));
  mTEST_FORMAT("false", mFormat(false));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mFormat, TestFloats)
{
  mTEST_ALLOCATOR_SETUP();

  mFormatState_ResetCulture();

  mFormatState &fs = mFormat_GetState();

  mDEFER(mAllocator_FreePtr(fs.pAllocator, &fs.textStart); fs.textCapacity = 0; fs.pAllocator = nullptr;);
  mAllocator_FreePtr(fs.pAllocator, &fs.textStart);
  fs.pAllocator = pAllocator;
  fs.textCapacity = 0;

  mTEST_FORMAT("1.23450e+4", mFormat(mFFloat<mFExponent, mFFractionalDigitsFixed>(12345.0f)));
  mTEST_FORMAT("1.2345e+4", mFormat(mFFloat<mFExponent>(12345.0f)));
  mTEST_FORMAT("1.23456e+4", mFormat(mFFloat<mFExponent>(12345.6f)));
  mTEST_FORMAT("1.23457e+4", mFormat(mFFloat<mFExponent>(12345.67f)));
  mTEST_FORMAT("9.04259e-22", mFormat(mFFloat<mFExponent>(9.04259e-22f)));
  mTEST_FORMAT("9.04259e+22", mFormat(mFFloat<mFExponent>(9.04259e+22f)));
  mTEST_FORMAT("9.99999e-1", mFormat(mFFloat<mFExponent>(0.999999f)));
  mTEST_FORMAT("1.0000e+0", mFormat(mFFloat<mFExponent, mFFractionalDigits<4>, mFFractionalDigitsFixed>(0.999999f)));
  mTEST_FORMAT("1e+0", mFormat(mFFloat<mFExponent, mFFractionalDigits<4>>(0.999999f)));
  mTEST_FORMAT("0.00000e+0", mFormat(mFFloat<mFExponent, mFFractionalDigitsFixed>(0)));
  mTEST_FORMAT("0e+0", mFormat(mFFloat<mFExponent>(0)));
  mTEST_FORMAT("0.0e+0", mFormat(mFFloat<mFExponent, mFFractionalDigitsFixed, mFMaxDigits<6>>(0)));
  mTEST_FORMAT("0e+0", mFormat(mFFloat<mFExponent, mFFractionalDigitsFixed, mFMaxDigits<5>>(0)));
  mTEST_FORMAT("0e+0", mFormat(mFFloat<mFExponent, mFFractionalDigitsFixed, mFMaxDigits<4>>(0)));
  mTEST_FORMAT("-1.9e+0", mFormat(mFFloat<mFExponent, mFFractionalDigitsFixed, mFMaxDigits<7>>(-1.9f)));
  mTEST_FORMAT("-2e+0", mFormat(mFFloat<mFExponent, mFFractionalDigitsFixed, mFMaxDigits<6>>(-1.9f)));
  mTEST_FORMAT("-2e+0", mFormat(mFFloat<mFExponent, mFFractionalDigitsFixed, mFMaxDigits<5>>(-1.9f)));
  mTEST_FORMAT("-1.9e+0", mFormat(mFFloat<mFExponent, mFMaxDigits<7>>(-1.9f)));
  mTEST_FORMAT("-2e+0", mFormat(mFFloat<mFExponent, mFMaxDigits<6>>(-1.9f)));
  mTEST_FORMAT("-2e+0", mFormat(mFFloat<mFExponent, mFMaxDigits<5>>(-1.9f)));
  mTEST_FORMAT("-1e+0", mFormat(mFFloat<mFExponent, mFMaxDigits<5>>(-1.4f)));
  mTEST_FORMAT("-00001.23450e+4", mFormat(mFFloat<mFExponent, mFMinDigits<15>, mFAlignNumRight, mFFillZeroes, mFSignAligned, mFFractionalDigitsFixed>(-12345.0f)));
  mTEST_FORMAT("-00001.23450e+4", mFormat(mFFloat<mFExponent, mFMinDigits<15>, mFAlignNumRight, mFFillZeroes, mFSignNotAligned, mFFractionalDigitsFixed>(-12345.0f)));
  mTEST_FORMAT("-000001.2345e+4", mFormat(mFFloat<mFExponent, mFMinDigits<15>, mFAlignNumRight, mFFillZeroes, mFSignAligned>(-12345.0f)));
  mTEST_FORMAT("-000001.2345e+4", mFormat(mFFloat<mFExponent, mFMinDigits<15>, mFAlignNumRight, mFFillZeroes, mFSignNotAligned>(-12345.0f)));
  mTEST_FORMAT("-00001.23456e+4", mFormat(mFFloat<mFExponent, mFMinDigits<15>, mFAlignNumRight, mFFillZeroes, mFSignAligned>(-12345.6f)));
  mTEST_FORMAT("-00001.23456e+4", mFormat(mFFloat<mFExponent, mFMinDigits<15>, mFAlignNumRight, mFFillZeroes, mFSignNotAligned>(-12345.6f)));
  mTEST_FORMAT("-00001.23456e+4", mFormat(mFFloat<mFExponent, mFMinDigits<15>, mFAlignNumRight, mFFillZeroes, mFSignAligned, mFFractionalDigitsFixed>(-12345.6f)));
  mTEST_FORMAT("-00001.23456e+4", mFormat(mFFloat<mFExponent, mFMinDigits<15>, mFAlignNumRight, mFFillZeroes, mFSignNotAligned, mFFractionalDigitsFixed>(-12345.6f)));

  mTEST_FORMAT("-1.23450e+4", mFormat(mFFloat<mFExponent, mFMinDigits<11>, mFAlignNumRight, mFFillZeroes, mFGroupDigits, mFSignAligned, mFFractionalDigitsFixed>(-12345.0f)));
  mTEST_FORMAT("-01.2345e+4", mFormat(mFFloat<mFExponent, mFMinDigits<11>, mFAlignNumRight, mFFillZeroes, mFGroupDigits, mFSignAligned>(-12345.0f)));
  mTEST_FORMAT("-01.23450e+4", mFormat(mFFloat<mFExponent, mFMinDigits<12>, mFAlignNumRight, mFFillZeroes, mFGroupDigits, mFSignAligned, mFFractionalDigitsFixed>(-12345.0f)));
  mTEST_FORMAT("-001.2345e+4", mFormat(mFFloat<mFExponent, mFMinDigits<12>, mFAlignNumRight, mFFillZeroes, mFGroupDigits, mFSignAligned>(-12345.0f)));
  mTEST_FORMAT("-0,001.23450e+4", mFormat(mFFloat<mFExponent, mFMinDigits<15>, mFAlignNumRight, mFFillZeroes, mFGroupDigits, mFSignAligned, mFFractionalDigitsFixed>(-12345.0f)));
  mTEST_FORMAT("-00,001.2345e+4", mFormat(mFFloat<mFExponent, mFMinDigits<15>, mFAlignNumRight, mFFillZeroes, mFGroupDigits, mFSignAligned>(-12345.0f)));
  mTEST_FORMAT("-00,001.23450e+4", mFormat(mFFloat<mFExponent, mFMinDigits<16>, mFAlignNumRight, mFFillZeroes, mFGroupDigits, mFSignAligned, mFFractionalDigitsFixed>(-12345.0f)));
  mTEST_FORMAT("-000,001.2345e+4", mFormat(mFFloat<mFExponent, mFMinDigits<16>, mFAlignNumRight, mFFillZeroes, mFGroupDigits, mFSignAligned>(-12345.0f)));
  mTEST_FORMAT("-000,001.23450e+4", mFormat(mFFloat<mFExponent, mFMinDigits<17>, mFAlignNumRight, mFFillZeroes, mFGroupDigits, mFSignAligned, mFFractionalDigitsFixed>(-12345.0f)));
  mTEST_FORMAT("- 000,001.2345e+4", mFormat(mFFloat<mFExponent, mFMinDigits<17>, mFAlignNumRight, mFFillZeroes, mFGroupDigits, mFSignAligned>(-12345.0f)));
  mTEST_FORMAT("- 000,001.23450e+4", mFormat(mFFloat<mFExponent, mFMinDigits<18>, mFAlignNumRight, mFFillZeroes, mFGroupDigits, mFSignAligned, mFFractionalDigitsFixed>(-12345.0f)));
  mTEST_FORMAT("-0,000,001.23450e+4", mFormat(mFFloat<mFExponent, mFMinDigits<19>, mFAlignNumRight, mFFillZeroes, mFGroupDigits, mFSignAligned, mFFractionalDigitsFixed>(-12345.0f)));
  mTEST_FORMAT("-00,000,001.23450e+4", mFormat(mFFloat<mFExponent, mFMinDigits<20>, mFAlignNumRight, mFFillZeroes, mFGroupDigits, mFSignAligned, mFFractionalDigitsFixed>(-12345.0f)));

  fs.digitGroupingOption = mFDGO_TenThousand;

  mTEST_FORMAT("-1.23450e+4", mFormat(mFFloat<mFExponent, mFMinDigits<11>, mFAlignNumRight, mFFillZeroes, mFGroupDigits, mFSignAligned, mFFractionalDigitsFixed>(-12345.0f)));
  mTEST_FORMAT("-01.23450e+4", mFormat(mFFloat<mFExponent, mFMinDigits<12>, mFAlignNumRight, mFFillZeroes, mFGroupDigits, mFSignAligned, mFFractionalDigitsFixed>(-12345.0f)));
  mTEST_FORMAT("-0,0001.23450e+4", mFormat(mFFloat<mFExponent, mFMinDigits<16>, mFAlignNumRight, mFFillZeroes, mFGroupDigits, mFSignAligned, mFFractionalDigitsFixed>(-12345.0f)));
  mTEST_FORMAT("-00,0001.23450e+4", mFormat(mFFloat<mFExponent, mFMinDigits<17>, mFAlignNumRight, mFFillZeroes, mFGroupDigits, mFSignAligned, mFFractionalDigitsFixed>(-12345.0f)));
  mTEST_FORMAT("-000,0001.23450e+4", mFormat(mFFloat<mFExponent, mFMinDigits<18>, mFAlignNumRight, mFFillZeroes, mFGroupDigits, mFSignAligned, mFFractionalDigitsFixed>(-12345.0f)));
  mTEST_FORMAT("-0000,0001.23450e+4", mFormat(mFFloat<mFExponent, mFMinDigits<19>, mFAlignNumRight, mFFillZeroes, mFGroupDigits, mFSignAligned, mFFractionalDigitsFixed>(-12345.0f)));
  mTEST_FORMAT("- 0000,0001.23450e+4", mFormat(mFFloat<mFExponent, mFMinDigits<20>, mFAlignNumRight, mFFillZeroes, mFGroupDigits, mFSignAligned, mFFractionalDigitsFixed>(-12345.0f)));

  fs.digitGroupingOption = mFDGO_Indian;

  mTEST_FORMAT("-1.23450e+4", mFormat(mFFloat<mFExponent, mFMinDigits<11>, mFAlignNumRight, mFFillZeroes, mFGroupDigits, mFSignAligned, mFFractionalDigitsFixed>(-12345.0f)));
  mTEST_FORMAT("-01.23450e+4", mFormat(mFFloat<mFExponent, mFMinDigits<12>, mFAlignNumRight, mFFillZeroes, mFGroupDigits, mFSignAligned, mFFractionalDigitsFixed>(-12345.0f)));
  mTEST_FORMAT("- 001.23450e+4", mFormat(mFFloat<mFExponent, mFMinDigits<14>, mFAlignNumRight, mFFillZeroes, mFGroupDigits, mFSignAligned, mFFractionalDigitsFixed>(-12345.0f)));
  mTEST_FORMAT("-0,001.23450e+4", mFormat(mFFloat<mFExponent, mFMinDigits<15>, mFAlignNumRight, mFFillZeroes, mFGroupDigits, mFSignAligned, mFFractionalDigitsFixed>(-12345.0f)));
  mTEST_FORMAT("-00,001.23450e+4", mFormat(mFFloat<mFExponent, mFMinDigits<16>, mFAlignNumRight, mFFillZeroes, mFGroupDigits, mFSignAligned, mFFractionalDigitsFixed>(-12345.0f)));
  mTEST_FORMAT("- 00,001.23450e+4", mFormat(mFFloat<mFExponent, mFMinDigits<17>, mFAlignNumRight, mFFillZeroes, mFGroupDigits, mFSignAligned, mFFractionalDigitsFixed>(-12345.0f)));
  mTEST_FORMAT("-0,00,001.23450e+4", mFormat(mFFloat<mFExponent, mFMinDigits<18>, mFAlignNumRight, mFFillZeroes, mFGroupDigits, mFSignAligned, mFFractionalDigitsFixed>(-12345.0f)));
  mTEST_FORMAT("-00,00,001.23450e+4", mFormat(mFFloat<mFExponent, mFMinDigits<19>, mFAlignNumRight, mFFillZeroes, mFGroupDigits, mFSignAligned, mFFractionalDigitsFixed>(-12345.0f)));
  mTEST_FORMAT("- 00,00,001.23450e+4", mFormat(mFFloat<mFExponent, mFMinDigits<20>, mFAlignNumRight, mFFillZeroes, mFGroupDigits, mFSignAligned, mFFractionalDigitsFixed>(-12345.0f)));
  mTEST_FORMAT("-0,00,00,001.23450e+4", mFormat(mFFloat<mFExponent, mFMinDigits<21>, mFAlignNumRight, mFFillZeroes, mFGroupDigits, mFSignAligned, mFFractionalDigitsFixed>(-12345.0f)));

  fs.digitGroupingOption = mFDGO_Thousand;

  mTEST_FORMAT("     -1.2345e+4", mFormat(mFFloat<mFExponent, mFMinDigits<15>, mFAlignNumRight, mFSignAligned>(-12345.0f)));
  mTEST_FORMAT("    -1.23450e+4", mFormat(mFFloat<mFExponent, mFMinDigits<15>, mFAlignNumRight, mFSignAligned, mFFractionalDigitsFixed>(-12345.0f)));
  mTEST_FORMAT("-     1.2345e+4", mFormat(mFFloat<mFExponent, mFMinDigits<15>, mFAlignNumRight, mFSignNotAligned>(-12345.0f)));
  mTEST_FORMAT("-    1.23450e+4", mFormat(mFFloat<mFExponent, mFMinDigits<15>, mFAlignNumRight, mFSignNotAligned, mFFractionalDigitsFixed>(-12345.0f)));
  mTEST_FORMAT("-1.234500000e+4", mFormat(mFFloat<mFExponent, mFMinDigits<15>, mFAlignNumLeft, mFFillZeroes, mFSignAligned>(-12345.0f)));
  mTEST_FORMAT("-1.234500000e+4", mFormat(mFFloat<mFExponent, mFMinDigits<15>, mFAlignNumLeft, mFFillZeroes, mFSignNotAligned, mFFractionalDigitsFixed>(-12345.0f)));
  mTEST_FORMAT("-1.2345e+4     ", mFormat(mFFloat<mFExponent, mFMinDigits<15>, mFAlignNumLeft, mFSignAligned>(-12345.0f)));
  mTEST_FORMAT("-1.23450e+4    ", mFormat(mFFloat<mFExponent, mFMinDigits<15>, mFAlignNumLeft, mFSignAligned, mFFractionalDigitsFixed>(-12345.0f)));
  mTEST_FORMAT("-1.2345e+4     ", mFormat(mFFloat<mFExponent, mFMinDigits<15>, mFAlignNumLeft, mFSignNotAligned>(-12345.0f)));
  mTEST_FORMAT("-1.23450e+4    ", mFormat(mFFloat<mFExponent, mFMinDigits<15>, mFAlignNumLeft, mFSignNotAligned, mFFractionalDigitsFixed>(-12345.0f)));
  mTEST_FORMAT("-1.000000000e+0", mFormat(mFFloat<mFExponent, mFMinDigits<15>, mFFractionalDigits<0>, mFAlignNumLeft, mFFillZeroes>(-1.0f)));
  mTEST_FORMAT("-1.000000000e+0", mFormat(mFFloat<mFExponent, mFMinDigits<15>, mFFractionalDigits<0>, mFAlignNumLeft, mFFillZeroes, mFFractionalDigitsFixed>(-1.0f)));
  mTEST_FORMAT("-1e+0 ", mFormat(mFFloat<mFExponent, mFMinDigits<6>, mFFractionalDigits<0>, mFAlignNumLeft, mFFillZeroes>(-1.0f)));
  mTEST_FORMAT("-1e+0 ", mFormat(mFFloat<mFExponent, mFMinDigits<6>, mFFractionalDigits<0>, mFAlignNumLeft, mFFillZeroes, mFFractionalDigitsFixed>(-1.0f)));
  mTEST_FORMAT("-1e+0          ", mFormat(mFFloat<mFExponent, mFMinDigits<15>, mFFractionalDigits<0>, mFAlignNumLeft>(-1.0f)));
  mTEST_FORMAT("-1e+0          ", mFormat(mFFloat<mFExponent, mFMinDigits<15>, mFFractionalDigits<0>, mFAlignNumLeft, mFFractionalDigitsFixed>(-1.0f)));
  mTEST_FORMAT("1.0000000000e+0", mFormat(mFFloat<mFExponent, mFMinDigits<15>, mFFractionalDigits<0>, mFAlignNumLeft, mFFillZeroes>(1.0f)));
  mTEST_FORMAT("1.0000000000e+0", mFormat(mFFloat<mFExponent, mFMinDigits<15>, mFFractionalDigits<0>, mFAlignNumLeft, mFFillZeroes, mFFractionalDigitsFixed>(1.0f)));
  mTEST_FORMAT("1e+0 ", mFormat(mFFloat<mFExponent, mFMinDigits<5>, mFFractionalDigits<0>, mFAlignNumLeft, mFFillZeroes>(1.0f)));
  mTEST_FORMAT("1e+0 ", mFormat(mFFloat<mFExponent, mFMinDigits<5>, mFFractionalDigits<0>, mFAlignNumLeft, mFFillZeroes, mFFractionalDigitsFixed>(1.0f)));
  mTEST_FORMAT("1e+0           ", mFormat(mFFloat<mFExponent, mFMinDigits<15>, mFFractionalDigits<0>, mFAlignNumLeft>(1.0f)));
  mTEST_FORMAT("1e+0           ", mFormat(mFFloat<mFExponent, mFMinDigits<15>, mFFractionalDigits<0>, mFAlignNumLeft, mFFractionalDigitsFixed>(1.0f)));
  mTEST_FORMAT("   -1.2345e+4   ", mFormat(mFFloat<mFExponent, mFMinDigits<16>, mFAlignNumCenter, mFSignAligned>(-12345.0f)));
  mTEST_FORMAT("  -1.23450e+4  ", mFormat(mFFloat<mFExponent, mFMinDigits<15>, mFAlignNumCenter, mFSignAligned, mFFractionalDigitsFixed>(-12345.0f)));
  mTEST_FORMAT("-   1.2345e+4   ", mFormat(mFFloat<mFExponent, mFMinDigits<16>, mFAlignNumCenter, mFSignNotAligned>(-12345.0f)));
  mTEST_FORMAT("-  1.23450e+4  ", mFormat(mFFloat<mFExponent, mFMinDigits<15>, mFAlignNumCenter, mFSignNotAligned, mFFractionalDigitsFixed>(-12345.0f)));

  mTEST_FORMAT("-1.23450e+4", mFormat(mFFloat<mFExponent, mFMinDigits<3>, mFAlignNumCenter, mFSignNotAligned, mFFractionalDigitsFixed>(-12345.0f)));
  mTEST_FORMAT("-1.2345e+4", mFormat(mFFloat<mFExponent, mFMinDigits<3>, mFAlignNumCenter, mFSignNotAligned>(-12345.0f)));
  mTEST_FORMAT("-1.23450e+4", mFormat(mFFloat<mFExponent, mFMinDigits<3>, mFAlignNumLeft, mFSignNotAligned, mFFractionalDigitsFixed>(-12345.0f)));
  mTEST_FORMAT("-1.2345e+4", mFormat(mFFloat<mFExponent, mFMinDigits<3>, mFAlignNumLeft, mFSignNotAligned>(-12345.0f)));
  mTEST_FORMAT("-1.23450e+4", mFormat(mFFloat<mFExponent, mFMinDigits<3>, mFAlignNumRight, mFSignNotAligned, mFFractionalDigitsFixed>(-12345.0f)));
  mTEST_FORMAT("-1.2345e+4", mFormat(mFFloat<mFExponent, mFMinDigits<3>, mFAlignNumRight, mFSignNotAligned>(-12345.0f)));

  mTEST_FORMAT("12345", mFormat(12345.0f));
  mTEST_FORMAT("12345.00000", mFormat(mFFloat<mFFractionalDigitsFixed>(12345.0f)));
  mTEST_FORMAT("0.12345", mFormat(0.12345f));
  mTEST_FORMAT("0.12345", mFormat(mFFloat<mFFractionalDigitsFixed>(0.12345f)));
  mTEST_FORMAT("12345.123", mFormat(12345.12345f));
  mTEST_FORMAT("12345.12300", mFormat(mFFloat<mFFractionalDigitsFixed>(12345.12345f)));

  mTEST_FORMAT("NaN", mFormat((float_t)NAN));
  mTEST_FORMAT("Infinity", mFormat((float_t)INFINITY));
  mTEST_FORMAT("-Infinity", mFormat(-(float_t)INFINITY));
  mTEST_FORMAT("0", mFormat(0.0f));
  mTEST_FORMAT("+0", mFormat(mFFloat<mFSignBoth>(0.0f)));
  mTEST_FORMAT(" 0", mFormat(mFFloat<mFSignNegativeOrFill>(0.0f)));
  mTEST_FORMAT(" 0", mFormat(mFFloat<mFSignNegativeOrFill, mFFillZeroes>(0.0f)));
  mTEST_FORMAT("0e+0", mFormat(mFFloat<mFExponent>(0.0f)));

  mTEST_FORMAT("NaN", mFormat((double_t)NAN));
  mTEST_FORMAT("Infinity", mFormat((double_t)INFINITY));
  mTEST_FORMAT("-Infinity", mFormat(-(double_t)INFINITY));
  mTEST_FORMAT("0", mFormat(0.0));
  mTEST_FORMAT("+0", mFormat(mFDouble<mFSignBoth>(0.0f)));
  mTEST_FORMAT(" 0", mFormat(mFDouble<mFSignNegativeOrFill>(0.0f)));
  mTEST_FORMAT(" 0", mFormat(mFDouble<mFSignNegativeOrFill, mFFillZeroes>(0.0f)));
  mTEST_FORMAT("0e+0", mFormat(mFDouble<mFExponent>(0.0)));

  mTEST_FORMAT("12345000000000000000.00000", mFormat(mFFloat<mFFractionalDigitsFixed>(12345000000000000000.0f)));
  mTEST_FORMAT("12345000000000000000", mFormat(12345000000000000000.0f));
  mTEST_FORMAT("10000000000000000000", mFormat(10000000000000000000.0f));
  mTEST_FORMAT("0.00001", mFormat(0.0000091234f));
  mTEST_FORMAT("0", mFormat(0.0000019876f));
  mTEST_FORMAT("0.00000", mFormat(mFFloat<mFFractionalDigitsFixed>(0.0000019876f)));
  mTEST_FORMAT("0.00001", mFormat(0.0000091234f));
  mTEST_FORMAT("0", mFormat(0.0000019876f));
  mTEST_FORMAT("0.00000", mFormat(mFFloat<mFFractionalDigitsFixed>(0.0000019876f)));
  mTEST_FORMAT("-0.00001", mFormat(-0.0000091234f));
  mTEST_FORMAT("-0", mFormat(-0.0000019876f));
  mTEST_FORMAT("-0.00000", mFormat(mFFloat<mFFractionalDigitsFixed>(-0.0000019876f)));
  mTEST_FORMAT("-0.00001", mFormat(-0.0000091234f));
  mTEST_FORMAT("-0", mFormat(-0.0000019876f));
  mTEST_FORMAT("-0.00000", mFormat(mFFloat<mFFractionalDigitsFixed>(-0.0000019876f)));
  mTEST_FORMAT("10", mFormat(10.0f));
  mTEST_FORMAT("10.00000", mFormat(mFFloat<mFFractionalDigitsFixed>(9.999995f)));
  mTEST_FORMAT("10", mFormat(9.999995f));
  mTEST_FORMAT("9.99999", mFormat(mFFloat<mFFractionalDigitsFixed>(9.999994f)));
  mTEST_FORMAT("9.99999", mFormat(9.999994f));
  mTEST_FORMAT("-10.00000", mFormat(mFFloat<mFFractionalDigitsFixed>(-9.999995f)));
  mTEST_FORMAT("-10", mFormat(-9.999995f));
  mTEST_FORMAT("-9.99999", mFormat(mFFloat<mFFractionalDigitsFixed>(-9.999994f)));
  mTEST_FORMAT("-9.99999", mFormat(-9.999994f));
  mTEST_FORMAT("12.4", mFormat(12.399999f));
  mTEST_FORMAT("-12.4", mFormat(-12.399999f));
  mTEST_FORMAT("12.39999", mFormat(12.399991f));
  mTEST_FORMAT("-12.39999", mFormat(-12.399991f));
  mTEST_FORMAT("0", mFormat((float_t)FLT_EPSILON));
  mTEST_FORMAT("0.00000", mFormat(mFFloat<mFFractionalDigitsFixed>(FLT_EPSILON)));
  mTEST_FORMAT("-0", mFormat(-(float_t)FLT_EPSILON));
  mTEST_FORMAT("-0.00000", mFormat(mFFloat<mFFractionalDigitsFixed>(-FLT_EPSILON)));

  mTEST_FORMAT("0", mFormat(DBL_EPSILON));
  mTEST_FORMAT("0.00000", mFormat(mFDouble<mFFractionalDigitsFixed>(DBL_EPSILON)));
  mTEST_FORMAT("-0", mFormat(-DBL_EPSILON));
  mTEST_FORMAT("-0.00000", mFormat(mFDouble<mFFractionalDigitsFixed>(-DBL_EPSILON)));

  mTEST_FORMAT("12345", mFormat(mFFloat<mFExponentAdaptive>(12345)));
  mTEST_FORMAT("1.2345e+4", mFormat(mFFloat<mFExponentAdaptive, mFExponent>(12345)));
  mTEST_FORMAT("12345.00000", mFormat(mFFloat<mFExponentAdaptive, mFFractionalDigitsFixed>(12345)));
  mTEST_FORMAT("0.00123", mFormat(mFFloat<mFExponentAdaptive>(0.0012345)));
  mTEST_FORMAT("1.2345e-3", mFormat(mFFloat<mFExponentAdaptive, mFExponent>(0.0012345)));
  mTEST_FORMAT("0.00123", mFormat(mFFloat<mFExponentAdaptive, mFFractionalDigitsFixed>(0.0012345)));
  mTEST_FORMAT("123450000000000000000", mFormat(1.2345e20));
  mTEST_FORMAT("1.2345e+20", mFormat(mFFloat<mFExponentAdaptive>(1.2345e20)));
  mTEST_FORMAT("1.2345e+20", mFormat(mFFloat<mFExponentAdaptive, mFExponent>(1.2345e20)));
  mTEST_FORMAT("1.23450e+20", mFormat(mFFloat<mFExponentAdaptive, mFFractionalDigitsFixed>(1.2345e20)));
  mTEST_FORMAT("0", mFormat(1.2345e-20));
  mTEST_FORMAT("1.2345e-20", mFormat(mFFloat<mFExponentAdaptive>(1.2345e-20)));
  mTEST_FORMAT("1.2345e-20", mFormat(mFFloat<mFExponentAdaptive, mFExponent>(1.2345e-20)));
  mTEST_FORMAT("1.23450e-20", mFormat(mFFloat<mFExponentAdaptive, mFFractionalDigitsFixed>(1.2345e-20)));

  mTEST_FORMAT("-12345", mFormat(mFFloat<mFExponentAdaptive>(-12345)));
  mTEST_FORMAT("-1.2345e+4", mFormat(mFFloat<mFExponentAdaptive, mFExponent>(-12345)));
  mTEST_FORMAT("-12345.00000", mFormat(mFFloat<mFExponentAdaptive, mFFractionalDigitsFixed>(-12345)));
  mTEST_FORMAT("-0.00123", mFormat(mFFloat<mFExponentAdaptive>(-0.0012345)));
  mTEST_FORMAT("-1.2345e-3", mFormat(mFFloat<mFExponentAdaptive, mFExponent>(-0.0012345)));
  mTEST_FORMAT("-0.00123", mFormat(mFFloat<mFExponentAdaptive, mFFractionalDigitsFixed>(-0.0012345)));
  mTEST_FORMAT("-123450000000000000000", mFormat(-1.2345e20));
  mTEST_FORMAT("-1.2345e+20", mFormat(mFFloat<mFExponentAdaptive>(-1.2345e20)));
  mTEST_FORMAT("-1.2345e+20", mFormat(mFFloat<mFExponentAdaptive, mFExponent>(-1.2345e20)));
  mTEST_FORMAT("-1.23450e+20", mFormat(mFFloat<mFExponentAdaptive, mFFractionalDigitsFixed>(-1.2345e20)));
  mTEST_FORMAT("-0", mFormat(-1.2345e-20));
  mTEST_FORMAT("-1.2345e-20", mFormat(mFFloat<mFExponentAdaptive>(-1.2345e-20)));
  mTEST_FORMAT("-1.2345e-20", mFormat(mFFloat<mFExponentAdaptive, mFExponent>(-1.2345e-20)));
  mTEST_FORMAT("-1.23450e-20", mFormat(mFFloat<mFExponentAdaptive, mFFractionalDigitsFixed>(-1.2345e-20)));

  mTEST_FORMAT("-1.23450e-200", mFormat(mFDouble<mFExponentAdaptive, mFFractionalDigitsFixed>(-1.2345e-200)));

  mTEST_FORMAT("9087650000000000000000", mFormat(9087650000000000000000.0f));
  mTEST_FORMAT("9,087,650,000,000,000,000,000", mFormat(mFFloat<mFGroupDigits>(9087650000000000000000.0f)));
  mTEST_FORMAT("-9,087,650,000,000,000,000,000", mFormat(mFFloat<mFGroupDigits>(-9087650000000000000000.0f)));
  mTEST_FORMAT("-000,000,009,087,650,000,000,000,000,000", mFormat(mFFloat<mFGroupDigits, mFMinDigits<40>, mFFillZeroes>(-9087650000000000000000.0f)));
  mTEST_FORMAT(" 000,000,009,087,650,000,000,000,000,000", mFormat(mFFloat<mFGroupDigits, mFMinDigits<40>, mFFillZeroes>(9087650000000000000000.0f)));
  mTEST_FORMAT("00,009,087,650,000,000,000,000,000.00000", mFormat(mFFloat<mFGroupDigits, mFMinDigits<40>, mFFillZeroes, mFFractionalDigitsFixed>(9087650000000000000000.0f)));
  mTEST_FORMAT("-0,009,087,650,000,000,000,000,000.00000", mFormat(mFFloat<mFGroupDigits, mFMinDigits<40>, mFFillZeroes, mFFractionalDigitsFixed>(-9087650000000000000000.0f)));
  mTEST_FORMAT("-9,087,650,000,000,000,000,000.00000", mFormat(mFFloat<mFGroupDigits, mFMaxDigits<36>, mFFractionalDigitsFixed>(-9087650000000000000000.0f)));
  mTEST_FORMAT("-9,087,650,000,000,000,000,000.0", mFormat(mFFloat<mFGroupDigits, mFMaxDigits<32>, mFFractionalDigitsFixed>(-9087650000000000000000.0f)));
  mTEST_FORMAT("9,087,650,000,000,000,000,000.00", mFormat(mFFloat<mFGroupDigits, mFMaxDigits<32>, mFFractionalDigitsFixed>(9087650000000000000000.0f)));
  mTEST_FORMAT("12345.7", mFormat(mFFloat<mFMaxDigits<7>>(12345.67f)));
  mTEST_FORMAT("12,346", mFormat(mFFloat<mFGroupDigits, mFMaxDigits<7>>(12345.67f)));
  mTEST_FORMAT("12,345.7", mFormat(mFFloat<mFGroupDigits, mFMaxDigits<8>>(12345.67f)));
  mTEST_FORMAT("-12,345.7", mFormat(mFFloat<mFGroupDigits, mFMaxDigits<9>>(-12345.67f)));
  mTEST_FORMAT("-12,346", mFormat(mFFloat<mFGroupDigits, mFMaxDigits<8>>(-12345.67f)));
  mTEST_FORMAT("-12,346", mFormat(mFFloat<mFGroupDigits, mFMaxDigits<7>>(-12345.67f)));

  mTEST_FORMAT("0.00012", mFormat(mFFloat<mFGroupDigits, mFMaxDigits<8>>(0.000123)));
  mTEST_FORMAT("0.00012", mFormat(mFFloat<mFGroupDigits, mFMaxDigits<7>>(0.000123)));
  mTEST_FORMAT("0.0001", mFormat(mFFloat<mFGroupDigits, mFMaxDigits<6>>(0.000123)));
  mTEST_FORMAT("0", mFormat(mFFloat<mFGroupDigits, mFMaxDigits<5>>(0.000123)));
  mTEST_FORMAT("0", mFormat(mFFloat<mFGroupDigits, mFMaxDigits<4>>(0.000123)));
  mTEST_FORMAT("0", mFormat(mFFloat<mFGroupDigits, mFMaxDigits<3>>(0.000123)));
  mTEST_FORMAT("0", mFormat(mFFloat<mFGroupDigits, mFMaxDigits<2>>(0.000123)));
  mTEST_FORMAT("0", mFormat(mFFloat<mFGroupDigits, mFMaxDigits<1>>(0.000123)));

  mTEST_FORMAT("0.01230", mFormat(mFFloat<mFFractionalDigitsFixed, mFMaxDigits<7>>(0.0123)));
  mTEST_FORMAT("0.12300", mFormat(mFFloat<mFFractionalDigitsFixed, mFMaxDigits<7>>(0.123)));
  mTEST_FORMAT("1.23000", mFormat(mFFloat<mFFractionalDigitsFixed, mFMaxDigits<7>>(1.23)));
  mTEST_FORMAT("12.3000", mFormat(mFFloat<mFFractionalDigitsFixed, mFMaxDigits<7>>(12.3)));
  mTEST_FORMAT("123.000", mFormat(mFFloat<mFFractionalDigitsFixed, mFMaxDigits<7>>(123)));
  mTEST_FORMAT("1230.00", mFormat(mFFloat<mFFractionalDigitsFixed, mFMaxDigits<7>>(1230)));
  mTEST_FORMAT("12300.0", mFormat(mFFloat<mFFractionalDigitsFixed, mFMaxDigits<7>>(12300)));
  mTEST_FORMAT("123000", mFormat(mFFloat<mFFractionalDigitsFixed, mFMaxDigits<7>>(123000)));
  mTEST_FORMAT("1230000", mFormat(mFFloat<mFFractionalDigitsFixed, mFMaxDigits<7>>(1230000)));
  mTEST_FORMAT(" 123000", mFormat(mFFloat<mFFractionalDigitsFixed, mFMaxDigits<7>, mFMinDigits<7>>(123000)));
  mTEST_FORMAT("123000", mFormat(mFFloat<mFFractionalDigitsFixed, mFMaxDigits<6>>(123000)));
  mTEST_FORMAT(">9999", mFormat(mFFloat<mFFractionalDigitsFixed, mFMaxDigits<5>>(123000)));
  mTEST_FORMAT(">999", mFormat(mFFloat<mFFractionalDigitsFixed, mFMaxDigits<4>>(123000)));
  mTEST_FORMAT(">99", mFormat(mFFloat<mFFractionalDigitsFixed, mFMaxDigits<3>>(123000)));
  mTEST_FORMAT(">9", mFormat(mFFloat<mFFractionalDigitsFixed, mFMaxDigits<2>>(123000)));
  mTEST_FORMAT("", mFormat(mFFloat<mFFractionalDigitsFixed, mFMaxDigits<1>>(123000)));
  mTEST_FORMAT("-123000", mFormat(mFFloat<mFFractionalDigitsFixed, mFMaxDigits<7>>(-123000)));
  mTEST_FORMAT("<-9999", mFormat(mFFloat<mFFractionalDigitsFixed, mFMaxDigits<6>>(-123000)));
  mTEST_FORMAT("<-999", mFormat(mFFloat<mFFractionalDigitsFixed, mFMaxDigits<5>>(-123000)));
  mTEST_FORMAT("<-99", mFormat(mFFloat<mFFractionalDigitsFixed, mFMaxDigits<4>>(-123000)));
  mTEST_FORMAT("<-9", mFormat(mFFloat<mFFractionalDigitsFixed, mFMaxDigits<3>>(-123000)));
  mTEST_FORMAT("", mFormat(mFFloat<mFFractionalDigitsFixed, mFMaxDigits<2>>(-123000)));
  mTEST_FORMAT("", mFormat(mFFloat<mFFractionalDigitsFixed, mFMaxDigits<1>>(-123000)));
  mTEST_FORMAT("+123000", mFormat(mFFloat<mFSignBoth, mFMaxDigits<7>>(123000)));
  mTEST_FORMAT(">+9999", mFormat(mFFloat<mFSignBoth, mFMaxDigits<6>>(123000)));
  mTEST_FORMAT(">+999", mFormat(mFFloat<mFSignBoth, mFMaxDigits<5>>(123000)));
  mTEST_FORMAT(">+99", mFormat(mFFloat<mFSignBoth, mFMaxDigits<4>>(123000)));
  mTEST_FORMAT(">+9", mFormat(mFFloat<mFSignBoth, mFMaxDigits<3>>(123000)));
  mTEST_FORMAT("", mFormat(mFFloat<mFSignBoth, mFMaxDigits<2>>(123000)));
  mTEST_FORMAT("", mFormat(mFFloat<mFSignBoth, mFMaxDigits<1>>(123000)));

  mTEST_FORMAT("  0", mFormat(mFFloat<mFMaxDigits<5>, mFMinDigits<3>>(0.000123)));
  mTEST_FORMAT("000", mFormat(mFFloat<mFMaxDigits<5>, mFMinDigits<3>, mFFillZeroes>(0.000123)));
  mTEST_FORMAT("0.000", mFormat(mFFloat<mFMaxDigits<5>, mFMinDigits<3>, mFFillZeroes, mFFractionalDigitsFixed>(0.000123)));

  mTEST_FORMAT("0.0001230000", mFormat(mFFloat<mFFractionalDigitsFixed, mFFractionalDigits<10>>(0.000123)));
  mTEST_FORMAT("     0.0001230000", mFormat(mFFloat<mFMinDigits<17>, mFFractionalDigitsFixed, mFFractionalDigits<10>>(0.000123)));
  mTEST_FORMAT("000000.0001230000", mFormat(mFFloat<mFMinDigits<17>, mFFillZeroes, mFFractionalDigitsFixed, mFFractionalDigits<10>>(0.000123)));
  mTEST_FORMAT("00,000.0001230000", mFormat(mFFloat<mFMinDigits<17>, mFFillZeroes, mFFractionalDigitsFixed, mFGroupDigits, mFFractionalDigits<10>>(0.000123)));
  mTEST_FORMAT("-0.0001230000", mFormat(mFFloat<mFFractionalDigitsFixed, mFFractionalDigits<10>>(-0.000123)));
  mTEST_FORMAT("    -0.0001230000", mFormat(mFFloat<mFMinDigits<17>, mFFractionalDigitsFixed, mFSignAligned, mFFractionalDigits<10>>(-0.000123)));
  mTEST_FORMAT("-    0.0001230000", mFormat(mFFloat<mFMinDigits<17>, mFFractionalDigitsFixed, mFSignNotAligned, mFFractionalDigits<10>>(-0.000123)));
  mTEST_FORMAT("-00000.0001230000", mFormat(mFFloat<mFMinDigits<17>, mFFillZeroes, mFFractionalDigitsFixed, mFFractionalDigits<10>>(-0.000123)));
  mTEST_FORMAT("-0,000.0001230000", mFormat(mFFloat<mFMinDigits<17>, mFFillZeroes, mFFractionalDigitsFixed, mFGroupDigits, mFFractionalDigits<10>>(-0.000123)));
  mTEST_FORMAT("00,000,012,345.67", mFormat(mFDouble<mFMinDigits<17>, mFFillZeroes, mFGroupDigits, mFFractionalDigits<10>>(12345.67)));
  mTEST_FORMAT("12,345.6700000000", mFormat(mFDouble<mFMinDigits<17>, mFFillZeroes, mFFractionalDigitsFixed, mFGroupDigits, mFFractionalDigits<10>>(12345.67)));
  mTEST_FORMAT("0,012,345.6700000000", mFormat(mFDouble<mFMinDigits<20>, mFFillZeroes, mFFractionalDigitsFixed, mFGroupDigits, mFFractionalDigits<10>>(12345.67)));
  mTEST_FORMAT("            12345.67", mFormat(mFDouble<mFMinDigits<20>>(12345.67)));
  mTEST_FORMAT("00000000000012345.67", mFormat(mFDouble<mFMinDigits<20>, mFFillZeroes>(12345.67)));
  mTEST_FORMAT("           12,345.67", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits>(12345.67)));
  mTEST_FORMAT("0,000,000,012,345.67", mFormat(mFDouble<mFMinDigits<20>, mFFillZeroes, mFGroupDigits>(12345.67)));
  mTEST_FORMAT("- 000,000,012,345.67", mFormat(mFDouble<mFMinDigits<20>, mFSignAligned, mFFillZeroes, mFGroupDigits>(-12345.67)));
  mTEST_FORMAT("-000,000,012,345.67", mFormat(mFDouble<mFMinDigits<19>, mFSignAligned, mFFillZeroes, mFGroupDigits>(-12345.67)));
  mTEST_FORMAT("- 000,000,012,345.67", mFormat(mFDouble<mFMinDigits<20>, mFSignNotAligned, mFFillZeroes, mFGroupDigits>(-12345.67)));
  mTEST_FORMAT("-000,000,012,345.67", mFormat(mFDouble<mFMinDigits<19>, mFSignNotAligned, mFFillZeroes, mFGroupDigits>(-12345.67)));
  mTEST_FORMAT("12,345.6700000000000", mFormat(mFDouble<mFMinDigits<20>, mFFillZeroes, mFGroupDigits, mFAlignNumLeft>(12345.67)));
  mTEST_FORMAT(" 12,345.670000000000", mFormat(mFDouble<mFMinDigits<20>, mFFillZeroes, mFGroupDigits, mFAlignNumLeft, mFSignNegativeOrFill>(12345.67)));
  mTEST_FORMAT("12,345.67           ", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumLeft>(12345.67)));
  mTEST_FORMAT("12345.67            ", mFormat(mFDouble<mFMinDigits<20>, mFAlignNumLeft>(12345.67)));
  mTEST_FORMAT("-12,345.67          ", mFormat(mFDouble<mFMinDigits<20>, mFSignAligned, mFGroupDigits, mFAlignNumLeft>(-12345.67)));
  mTEST_FORMAT("-12,345.67          ", mFormat(mFDouble<mFMinDigits<20>, mFSignAligned, mFGroupDigits, mFAlignNumLeft>(-12345.67)));
  mTEST_FORMAT("-12345.67           ", mFormat(mFDouble<mFMinDigits<20>, mFSignNotAligned, mFAlignNumLeft>(-12345.67)));
  mTEST_FORMAT("-12345.67           ", mFormat(mFDouble<mFMinDigits<20>, mFSignNotAligned, mFAlignNumLeft>(-12345.67)));
  mTEST_FORMAT("     12,345.67     ", mFormat(mFDouble<mFMinDigits<19>, mFGroupDigits, mFAlignNumCenter>(12345.67)));
  mTEST_FORMAT("      12345.67      ", mFormat(mFDouble<mFMinDigits<20>, mFAlignNumCenter>(12345.67)));
  mTEST_FORMAT("     -12,345.67     ", mFormat(mFDouble<mFMinDigits<20>, mFSignAligned, mFGroupDigits, mFAlignNumCenter>(-12345.67)));
  mTEST_FORMAT("     +12,345.67     ", mFormat(mFDouble<mFMinDigits<20>, mFSignAligned, mFGroupDigits, mFAlignNumCenter, mFSignBoth>(12345.67)));
  mTEST_FORMAT("      12,345.67     ", mFormat(mFDouble<mFMinDigits<20>, mFSignAligned, mFGroupDigits, mFAlignNumCenter, mFSignNegativeOrFill>(12345.67)));
  mTEST_FORMAT("     12,345.67     ", mFormat(mFDouble<mFMinDigits<19>, mFSignAligned, mFGroupDigits, mFAlignNumCenter, mFSignNever>(-12345.67)));
  mTEST_FORMAT("      -12345.67      ", mFormat(mFDouble<mFMinDigits<21>, mFSignAligned, mFAlignNumCenter>(-12345.67)));
  mTEST_FORMAT("-     12,345.67     ", mFormat(mFDouble<mFMinDigits<20>, mFSignNotAligned, mFGroupDigits, mFAlignNumCenter>(-12345.67)));
  mTEST_FORMAT("-      12345.67      ", mFormat(mFDouble<mFMinDigits<21>, mFSignNotAligned, mFAlignNumCenter>(-12345.67)));
  mTEST_FORMAT("-12345.6700000000000", mFormat(mFDouble<mFMinDigits<20>, mFAlignNumLeft, mFFillZeroes>(-12345.67)));
  mTEST_FORMAT("12345.67000000000000", mFormat(mFDouble<mFMinDigits<20>, mFAlignNumLeft, mFFillZeroes>(12345.67)));
  mTEST_FORMAT("12,345.6700000000000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumLeft, mFFillZeroes>(12345.67)));
  mTEST_FORMAT("-12,345.670000000000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumLeft, mFFillZeroes>(-12345.67)));
  mTEST_FORMAT("-12,345.000000000000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumLeft, mFFillZeroes>(-12345.00)));
  mTEST_FORMAT("-12,345 ", mFormat(mFDouble<mFMinDigits<8>, mFGroupDigits, mFAlignNumLeft, mFFillZeroes>(-12345.00)));
  mTEST_FORMAT("-12,345 ", mFormat(mFDouble<mFMinDigits<8>, mFMaxDigits<8>, mFGroupDigits, mFAlignNumLeft, mFFillZeroes, mFFractionalDigitsFixed>(-12345.00)));
  mTEST_FORMAT("-12,345", mFormat(mFDouble<mFMinDigits<7>, mFGroupDigits, mFAlignNumLeft, mFFillZeroes>(-12345.00)));
  mTEST_FORMAT("0.001239500000000000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumLeft, mFFillZeroes>(0.0012395)));
  mTEST_FORMAT("0.00124             ", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumLeft>(0.0012395)));
  mTEST_FORMAT("0.123450000000000000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumLeft, mFFillZeroes>(0.12345)));
  mTEST_FORMAT("0.12345             ", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumLeft, mFFractionalDigitsFixed>(0.12345)));
  mTEST_FORMAT("0.00012             ", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumLeft, mFFractionalDigitsFixed>(0.00012345)));
  mTEST_FORMAT("-0.12345            ", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumLeft, mFFractionalDigitsFixed>(-0.12345)));
  mTEST_FORMAT("-0.00012            ", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumLeft, mFFractionalDigitsFixed>(-0.00012345)));
  mTEST_FORMAT("-0000000000000.00012", mFormat(mFDouble<mFMinDigits<20>, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(-0.00012345)));

  mTEST_FORMAT("-0,000,000,000.00012", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(-0.00012345)));
  mTEST_FORMAT("-0,000,000,000.00012", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(-0.00012345)));
  mTEST_FORMAT("00,000,000,000.00012", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(0.00012345)));
  mTEST_FORMAT("00,000,000,000.00012", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(0.00012345)));
  mTEST_FORMAT("-0,000,000,000.12345", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(-0.12345)));
  mTEST_FORMAT("-0,000,000,000.12345", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(-0.12345)));
  mTEST_FORMAT("00,000,000,000.12345", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(0.12345)));
  mTEST_FORMAT("00,000,000,000.12345", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(0.12345)));
  mTEST_FORMAT("-0,000,000,001.23450", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(-1.2345)));
  mTEST_FORMAT("-00,000,000,001.2345", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(-1.2345)));
  mTEST_FORMAT("00,000,000,001.23450", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(1.2345)));
  mTEST_FORMAT("000,000,000,001.2345", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(1.2345)));
  mTEST_FORMAT("-0,000,001,234.50000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(-1234.5)));
  mTEST_FORMAT("-0,000,000,001,234.5", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(-1234.5)));
  mTEST_FORMAT("00,000,001,234.50000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(1234.5)));
  mTEST_FORMAT("00,000,000,001,234.5", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(1234.5)));
  mTEST_FORMAT("-0,000,012,345.00000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(-12345)));
  mTEST_FORMAT("-000,000,000,012,345", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(-12345)));
  mTEST_FORMAT("00,000,012,345.00000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(12345)));
  mTEST_FORMAT(" 000,000,000,012,345", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(12345)));
  mTEST_FORMAT("-0,123,450,000.00000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(-123450000)));
  mTEST_FORMAT("-000,000,123,450,000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(-123450000)));
  mTEST_FORMAT("00,123,450,000.00000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(123450000)));
  mTEST_FORMAT(" 000,000,123,450,000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(123450000)));
  mTEST_FORMAT("-1,234,500,000.00000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(-1234500000)));
  mTEST_FORMAT("-000,001,234,500,000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(-1234500000)));
  mTEST_FORMAT("01,234,500,000.00000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(1234500000)));
  mTEST_FORMAT(" 000,001,234,500,000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(1234500000)));
  mTEST_FORMAT("-01,234,500,000.00000", mFormat(mFDouble<mFMinDigits<21>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(-1234500000)));
  mTEST_FORMAT("- 000,001,234,500,000", mFormat(mFDouble<mFMinDigits<21>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(-1234500000)));
  mTEST_FORMAT("001,234,500,000.00000", mFormat(mFDouble<mFMinDigits<21>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(1234500000)));
  mTEST_FORMAT("0,000,001,234,500,000", mFormat(mFDouble<mFMinDigits<21>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(1234500000)));
  mTEST_FORMAT("-1,234,500,000,000.00000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(-1234500000000)));
  mTEST_FORMAT("-001,234,500,000,000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(-1234500000000)));
  mTEST_FORMAT("1,234,500,000,000.00000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(1234500000000)));
  mTEST_FORMAT(" 001,234,500,000,000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(1234500000000)));
  mTEST_FORMAT("-123,450,000,000,000.00000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(-123450000000000)));
  mTEST_FORMAT("-123,450,000,000,000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(-123450000000000)));
  mTEST_FORMAT("123,450,000,000,000.00000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(123450000000000)));
  mTEST_FORMAT(" 123,450,000,000,000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(123450000000000)));
  mTEST_FORMAT("0,123,450,000,000,000", mFormat(mFDouble<mFMinDigits<21>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(123450000000000)));
  mTEST_FORMAT("1,234,500,000,000,000.00000", mFormat(mFDouble<mFMinDigits<21>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(1234500000000000)));
  mTEST_FORMAT("1,234,500,000,000,000", mFormat(mFDouble<mFMinDigits<21>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(1234500000000000)));

  fs.digitGroupingOption = mFDGO_TenThousand;

  mTEST_FORMAT("0000,0000,0000.00012", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(0.00012345)));
  mTEST_FORMAT("0000,0000,0000.00012", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(0.00012345)));
  mTEST_FORMAT("0000,0000,0000.01235", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(0.012345)));
  mTEST_FORMAT("0000,0000,0000.01235", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(0.012345)));
  mTEST_FORMAT("0000,0000,0000.12345", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(0.12345)));
  mTEST_FORMAT("0000,0000,0000.12345", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(0.12345)));
  mTEST_FORMAT("0000,0000,0001.23450", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(1.2345)));
  mTEST_FORMAT(" 0000,0000,0001.2345", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(1.2345)));
  mTEST_FORMAT("0000,0000,0123.45000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(123.45)));
  mTEST_FORMAT("00,0000,0000,0123.45", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(123.45)));
  mTEST_FORMAT("0000,0000,1234.50000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(1234.5)));
  mTEST_FORMAT("000,0000,0000,1234.5", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(1234.5)));
  mTEST_FORMAT("0000,0001,2345.00000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(12345)));
  mTEST_FORMAT(" 0000,0000,0001,2345", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(12345)));
  mTEST_FORMAT("0000,0012,3450.00000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(123450)));
  mTEST_FORMAT(" 0000,0000,0012,3450", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(123450)));
  mTEST_FORMAT("1,2345,0000,0000,0000.00000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(12345000000000000)));
  mTEST_FORMAT("1,2345,0000,0000,0000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(12345000000000000)));

  fs.digitGroupingOption = mFDGO_Indian;

  mTEST_FORMAT("0,00,00,00,000.00012", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(0.00012345)));
  mTEST_FORMAT("0,00,00,00,000.00012", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(0.00012345)));
  mTEST_FORMAT("0,00,00,00,000.12345", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(0.12345)));
  mTEST_FORMAT("0,00,00,00,000.12345", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(0.12345)));
  mTEST_FORMAT("0,00,00,00,001.23450", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(1.2345)));
  mTEST_FORMAT("00,00,00,00,001.2345", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(1.2345)));
  mTEST_FORMAT("0,00,00,00,123.45000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(123.45)));
  mTEST_FORMAT("0,00,00,00,00,123.45", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(123.45)));
  mTEST_FORMAT("0,00,00,01,234.50000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(1234.5)));
  mTEST_FORMAT("00,00,00,00,01,234.5", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(1234.5)));
  mTEST_FORMAT("0,00,00,12,345.00000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(12345)));
  mTEST_FORMAT("0,00,00,00,00,12,345", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(12345)));
  mTEST_FORMAT("0,00,01,23,450.00000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(123450)));
  mTEST_FORMAT("0,00,00,00,01,23,450", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(123450)));
  mTEST_FORMAT("0,12,34,50,000.00000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(123450000)));
  mTEST_FORMAT("0,00,00,12,34,50,000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(123450000)));
  mTEST_FORMAT("00,12,34,50,000.00000", mFormat(mFDouble<mFMinDigits<21>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(123450000)));
  mTEST_FORMAT("00,00,00,12,34,50,000", mFormat(mFDouble<mFMinDigits<21>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(123450000)));
  mTEST_FORMAT(" 00,12,34,50,000.00000", mFormat(mFDouble<mFMinDigits<22>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(123450000)));
  mTEST_FORMAT(" 00,00,00,12,34,50,000", mFormat(mFDouble<mFMinDigits<22>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(123450000)));
  mTEST_FORMAT("1,23,45,00,000.00000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(1234500000)));
  mTEST_FORMAT("0,00,01,23,45,00,000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(1234500000)));
  mTEST_FORMAT("12,34,50,00,000.00000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(12345000000)));
  mTEST_FORMAT("0,00,12,34,50,00,000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(12345000000)));
  mTEST_FORMAT("12,34,50,00,00,000.00000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(1234500000000)));
  mTEST_FORMAT("0,12,34,50,00,00,000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(1234500000000)));
  mTEST_FORMAT("1,23,45,00,00,00,000.00000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes, mFFractionalDigitsFixed>(12345000000000)));
  mTEST_FORMAT("1,23,45,00,00,00,000", mFormat(mFDouble<mFMinDigits<20>, mFGroupDigits, mFAlignNumRight, mFFillZeroes>(12345000000000)));

  mFormatState_ResetCulture();

  float_t v = 1.0f;
  v *= 0.00000000000000000001f;

  for (int64_t i = -20; i < 0; i++)
    v *= 10;

  mTEST_FORMAT("1", mFormat(v));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mFormat, TestVectors)
{
  mTEST_ALLOCATOR_SETUP();

  mFormatState_ResetCulture();

  mFormatState &fs = mFormat_GetState();

  mDEFER(mAllocator_FreePtr(fs.pAllocator, &fs.textStart); fs.textCapacity = 0; fs.pAllocator = nullptr;);
  mAllocator_FreePtr(fs.pAllocator, &fs.textStart);
  fs.pAllocator = pAllocator;
  fs.textCapacity = 0;

  mTEST_FORMAT("[0, 1]", mFormat(mVec2i(0, 1)));
  mTEST_FORMAT("[0, 1, 2]", mFormat(mVec3i(0, 1, 2)));
  mTEST_FORMAT("[0, 1, 2, 3]", mFormat(mVec4i(0, 1, 2, 3)));

  mTEST_FORMAT("[0, 1]", mFormat(mVec2f(0, 1)));
  mTEST_FORMAT("[0, 1, 2]", mFormat(mVec3f(0, 1, 2)));
  mTEST_FORMAT("[0, 1, 2, 3]", mFormat(mVec4f(0, 1, 2, 3)));
  mTEST_FORMAT("[0, 1, 2, 3]", mFormat(mVector(0, 1, 2, 3)));
  mTEST_FORMAT("[0.123, -1.123]", mFormat(mVec2f(0.123, -1.123)));
  mTEST_FORMAT("[0.123, -1.123, 2.123]", mFormat(mVec3f(0.123, -1.123, 2.123)));
  mTEST_FORMAT("[0.123, -1.123, 2.123, -3.123]", mFormat(mVec4f(0.123, -1.123, 2.123, -3.123)));
  mTEST_FORMAT("[0.123, -1.123, 2.123, -3.123]", mFormat(mVector(0.123, -1.123, 2.123, -3.123)));

  mTEST_FORMAT("[0, 1]", mFormat(mVec2d(0, 1)));
  mTEST_FORMAT("[0, 1, 2]", mFormat(mVec3d(0, 1, 2)));
  mTEST_FORMAT("[0, 1, 2, 3]", mFormat(mVec4d(0, 1, 2, 3)));

  mTEST_FORMAT("[0, 1]", mFormat(mVec2u(0, 1)));
  mTEST_FORMAT("[0, 1, 2]", mFormat(mVec3u(0, 1, 2)));
  mTEST_FORMAT("[0, 1, 2, 3]", mFormat(mVec4u(0, 1, 2, 3)));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mFormat, TestChars)
{
  mTEST_ALLOCATOR_SETUP();

  mFormatState_ResetCulture();

  mFormatState &fs = mFormat_GetState();

  mDEFER(mAllocator_FreePtr(fs.pAllocator, &fs.textStart); fs.textCapacity = 0; fs.pAllocator = nullptr;);
  mAllocator_FreePtr(fs.pAllocator, &fs.textStart);
  fs.pAllocator = pAllocator;
  fs.textCapacity = 0;

  mTEST_FORMAT("d, e, f, g", mFormat('d', ", ", 'e', ", ", 'f', ", ", 'g'));
  mTEST_FORMAT("Check out these letters: A, B, C, D, E, F, G, a, b, c, d, e, f, g", mFormat("Check out these letters: ", 'A', ", ", 'B', ", ", 'C', ", ", 'D', ", ", 'E', ", ", 'F', ", ", 'G', ", ", 'a', ", ", 'b', ", ", 'c', ", ", 'd', ", ", 'e', ", ", 'f', ", ", 'g'));
  mTEST_FORMAT("This is a null terminator: .", mFormat("This is a null terminator: ", '\0', "."));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mFormat, TestWideChars)
{
  mTEST_ALLOCATOR_SETUP();

  mFormatState_ResetCulture();

  mFormatState &fs = mFormat_GetState();

  mDEFER(mAllocator_FreePtr(fs.pAllocator, &fs.textStart); fs.textCapacity = 0; fs.pAllocator = nullptr;);
  mAllocator_FreePtr(fs.pAllocator, &fs.textStart);
  fs.pAllocator = pAllocator;
  fs.textCapacity = 0;

  mTEST_FORMAT("Check out these letters: A, B, C, D, E, F, G, a, b, c, d, e, f, g", mFormat("Check out these letters: ", L'A', ", ", L'B', ", ", L'C', ", ", L'D', ", ", L'E', ", ", L'F', ", ", L'G', ", ", L'a', ", ", L'b', ", ", L'c', ", ", L'd', ", ", L'e', ", ", L'f', ", ", L'g'));
  mTEST_FORMAT("This is a null terminator: .", mFormat("This is a null terminator: ", L'\0', "."));
  mTEST_FORMAT("This is a three byte UTF8-Character: '\xE2\x80\xA6'.", mFormat("This is a three byte UTF8-Character: '", L'\x2026', "'."));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mFormat, TestWideString)
{
  mTEST_ALLOCATOR_SETUP();

  mFormatState_ResetCulture();

  mFormatState &fs = mFormat_GetState();

  mDEFER(mAllocator_FreePtr(fs.pAllocator, &fs.textStart); fs.textCapacity = 0; fs.pAllocator = nullptr;);
  mAllocator_FreePtr(fs.pAllocator, &fs.textStart);
  fs.pAllocator = pAllocator;
  fs.textCapacity = 0;

  mTEST_FORMAT("test", mFormat(L"test"));

  mTEST_FORMAT("Check out these letters: A, B, C, D, E, F, G, a, b, c, d, e, f, g", mFormat(L"Check out these letters: ", L'A', L", ", L'B', L", ", L'C', L", ", L'D', L", ", L'E', ", ", L'F', L", ", L'G', ", ", 'a', ", ", 'b', ", ", 'c', ", ", 'd', ", ", 'e', ", ", 'f', ", ", 'g'));
  mTEST_FORMAT("This is a null terminator: .", mFormat("This is a null terminator: ", L'\0', "."));
  mTEST_FORMAT("This is a three byte UTF8-Character: '\xE2\x80\xA6'.", mFormat(L"This is a three byte UTF8-Character: '", L'\x2026', "'."));
  mTEST_FORMAT("This is a three byte UTF8-Character: '\xE2\x80\xA6'.", mFormat(L"This is a three byte UTF8-Character: '\x2026'."));

  const wchar_t test2[] = L"test2";
  wchar_t test3[] = L"test3";
  wchar_t *test4 = L"test4";
  const wchar_t *test5 = L"test5";

  mTEST_FORMAT("test2test2", mFormat(test2, test2));
  mTEST_FORMAT("test3test3", mFormat(test3, test3));
  mTEST_FORMAT("test4test4", mFormat(test4, test4));
  mTEST_FORMAT("test5test5", mFormat(test5, test5));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

#ifndef _DEBUG
mTEST(mFormat, TestPerformance)
{
  mTEST_ALLOCATOR_SETUP();

  mFormatState_ResetCulture();

  mFormatState &fs = mFormat_GetState();

  mDEFER(mAllocator_FreePtr(fs.pAllocator, &fs.textStart); fs.textCapacity = 0; fs.pAllocator = nullptr;);
  mAllocator_FreePtr(fs.pAllocator, &fs.textStart);
  fs.pAllocator = pAllocator;
  fs.textCapacity = 0;

  char buffer[1024 * 4];

  {
    size_t countOld = 0;
    const int64_t startOld = mGetCurrentTimeNs();

    for (size_t i = 0; i < 1024 * 1024; i++)
    {
      sprintf(buffer, "A%" PRIu64 " || %" PRIi64, i, (i - 1024LL * 512LL));
      countOld += (size_t)(buffer[0] - 'A' + 1);
    }

    size_t countNew = 0;
    const int64_t endOld = mGetCurrentTimeNs();

    for (size_t i = 0; i < 1024 * 1024; i++)
    {
      mFormatTo(buffer, mARRAYSIZE(buffer), 'A', i, " || ", (i - 1024LL * 512LL));
      countNew += (size_t)(buffer[0] - 'A' + 1);
    }

    const int64_t endNew = mGetCurrentTimeNs();

    mTEST_ASSERT_TRUE((endOld - startOld) / (double_t)(endNew - endOld) > 2.5); // Please don't make this perform terribly. Should be about 6.4x faster.
  }

  {
    size_t countOld = 0;
    const int64_t startOld = mGetCurrentTimeNs();

    for (size_t i = 0; i < 1024 * 1024; i++)
    {
      sprintf(buffer, "A%" PRIu64 " || %" PRIi64 " %6.4f", i, (i - 1024LL * 512LL), i / 1000.0);
      countOld += (size_t)(buffer[0] - 'A' + 1);
    }

    size_t countNew = 0;
    const int64_t endOld = mGetCurrentTimeNs();

    for (size_t i = 0; i < 1024 * 1024; i++)
    {
      mFormatTo(buffer, mARRAYSIZE(buffer), 'A', i, " || ", (i - 1024LL * 512LL), ' ', mFD(Frac(4))(i / 1000.0));
      countNew += (size_t)(buffer[0] - 'A' + 1);
    }

    const int64_t endNew = mGetCurrentTimeNs();

    mTEST_ASSERT_TRUE((endOld - startOld) / (double_t)(endNew - endOld) > 2.0); // Please don't make this perform terribly. Should be about 4.7x faster.
  }

  {
    size_t countOld = 0;
    const int64_t startOld = mGetCurrentTimeNs();

    for (size_t i = 0; i < 1024 * 1024; i++)
    {
      sprintf(buffer, "A%" PRIu64 " || %" PRIi64 " %6.4f", i, (i - 1024LL * 512LL), i / 1000.0f);
      countOld += (size_t)(buffer[0] - 'A' + 1);
    }

    size_t countNew = 0;
    const int64_t endOld = mGetCurrentTimeNs();

    for (size_t i = 0; i < 1024 * 1024; i++)
    {
      mFormatTo(buffer, mARRAYSIZE(buffer), 'A', i, " || ", (i - 1024LL * 512LL), ' ', mFF(Frac(4))(i / 1000.0f));
      countNew += (size_t)(buffer[0] - 'A' + 1);
    }

    const int64_t endNew = mGetCurrentTimeNs();

    mTEST_ASSERT_TRUE((endOld - startOld) / (double_t)(endNew - endOld) > 2.0); // Please don't make this perform terribly. Should be about 4.7x faster.
  }

  {
    size_t countOld = 0;
    const int64_t startOld = mGetCurrentTimeNs();

    for (size_t i = 0; i < 1024 * 1024; i++)
    {
      sprintf(buffer, "A%" PRIu64 " || %" PRIi64 " %e", i, (i - 1024LL * 512LL), i / 1000.0f);
      countOld += (size_t)(buffer[0] - 'A' + 1);
    }

    size_t countNew = 0;
    const int64_t endOld = mGetCurrentTimeNs();

    for (size_t i = 0; i < 1024 * 1024; i++)
    {
      mFormatTo(buffer, mARRAYSIZE(buffer), 'A', i, " || ", (i - 1024LL * 512LL), ' ', mFF(Exp)(i / 1000.0f));
      countNew += (size_t)(buffer[0] - 'A' + 1);
    }

    const int64_t endNew = mGetCurrentTimeNs();

    mTEST_ASSERT_TRUE((endOld - startOld) / (double_t)(endNew - endOld) > 1.5); // Please don't make this perform terribly. Should be about 4.2x faster.
  }

  mTEST_ALLOCATOR_ZERO_CHECK();
}
#endif
