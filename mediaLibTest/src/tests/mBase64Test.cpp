#include "mTestLib.h"
#include "mBase64.h"

mTEST(mBase64, TestIsBase64)
{
  mTEST_ALLOCATOR_SETUP();

  mString nullString;
  bool result;

  mTEST_ASSERT_EQUAL(mR_ArgumentNull, mBase64_IsBase64(nullString, nullptr));
  
  mTEST_ASSERT_SUCCESS(mBase64_IsBase64(nullString, &result));
  mTEST_ASSERT_FALSE(result);

  mTEST_ASSERT_SUCCESS(mBase64_IsBase64("", &result));
  mTEST_ASSERT_TRUE(result);

  mTEST_ASSERT_SUCCESS(mBase64_IsBase64("🌵🦎🎅testמⴲx", &result));
  mTEST_ASSERT_FALSE(result);

  const char validChars[] = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/', '='};

  char text[2] = {};

  for (size_t i = 1; i < 128; i++)
  {
    text[0] = (char)i;

    bool contained = false;

    for (size_t j = 0; j < mARRAYSIZE(validChars); j++)
    {
      if (text[0] == validChars[j])
      {
        contained = true;
        break;
      }
    }

    mTEST_ASSERT_SUCCESS(mBase64_IsBase64(text, &result));
    mTEST_ASSERT_EQUAL(contained, result);
  }

  mTEST_ASSERT_SUCCESS(mBase64_IsBase64("TWFuIGlzIGRpc3Rpbmd1aXNoZWQsIG5vdCBvbmx5IGJ5IGhpcyByZWFzb24sIGJ1dCBieSB0aGlzIHNpbmd1bGFyIHBhc3Npb24gZnJvbSBvdGhlciBhbmltYWxzLCB3aGljaCBpcyBhIGx1c3Qgb2YgdGhlIG1pbmQsIHRoYXQgYnkgYSBwZXJzZXZlcmFuY2Ugb2YgZGVsaWdodCBpbiB0aGUgY29udGludWVkIGFuZCBpbmRlZmF0aWdhYmxlIGdlbmVyYXRpb24gb2Yga25vd2xlZGdlLCBleGNlZWRzIHRoZSBzaG9ydCB2ZWhlbWVuY2Ugb2YgYW55IGNhcm5hbCBwbGVhc3VyZS4=", &result)); // Not Null Terminated!
  mTEST_ASSERT_TRUE(result);

  mTEST_ASSERT_SUCCESS(mBase64_IsBase64("TWFuIGlzIGRpc3Rpbmd1aXNoZWQsIG5vdCBvbmx5IGJ5IGhpcyByZWFzb24sIGJ1dCBieSB0aGlzIHNpbmd1bGFyIHBhc3Npb24gZnJvbSBvdGhlciBhbmltYWxzLCB3aGljaCBpcyBhIGx1c3Qgb2YgdGhlIG1pbmQsIHRoYXQgYnkgYSBwZXJzZXZlcmFuY2Ugb2YgZGVsaWdodCBpbiB0aGUgY29udGludWVkIGFuZCBpbmRlZmF0aWdhYmxlIGdlbmVyYXRpb24gb2Yga25vd2xlZGdlLCBleGNlZWRzIHRoZSBzaG9ydCB2ZWhlbWVuY2Ugb2YgYW55IGNhcm5hbCBwbGVhc3VyZS4==", &result)); // Not Null Terminated!
  mTEST_ASSERT_TRUE(result);

  mTEST_ASSERT_SUCCESS(mBase64_IsBase64("=TWFuIGlzIGRpc3Rpbmd1aXNoZWQsIG5vdCBvbmx5IGJ5IGhpcyByZWFzb24sIGJ1dCBieSB0aGlzIHNpbmd1bGFyIHBhc3Npb24gZnJvbSBvdGhlciBhbmltYWxzLCB3aGljaCBpcyBhIGx1c3Qgb2YgdGhlIG1pbmQsIHRoYXQgYnkgYSBwZXJzZXZlcmFuY2Ugb2YgZGVsaWdodCBpbiB0aGUgY29udGludWVkIGFuZCBpbmRlZmF0aWdhYmxlIGdlbmVyYXRpb24gb2Yga25vd2xlZGdlLCBleGNlZWRzIHRoZSBzaG9ydCB2ZWhlbWVuY2Ugb2YgYW55IGNhcm5hbCBwbGVhc3VyZS4=", &result)); // Not Null Terminated!
  mTEST_ASSERT_FALSE(result);

  mTEST_ASSERT_SUCCESS(mBase64_IsBase64("T=WFuIGlzIGRpc3Rpbmd1aXNoZWQsIG5vdCBvbmx5IGJ5IGhpcyByZWFzb24sIGJ1dCBieSB0aGlzIHNpbmd1bGFyIHBhc3Npb24gZnJvbSBvdGhlciBhbmltYWxzLCB3aGljaCBpcyBhIGx1c3Qgb2YgdGhlIG1pbmQsIHRoYXQgYnkgYSBwZXJzZXZlcmFuY2Ugb2YgZGVsaWdodCBpbiB0aGUgY29udGludWVkIGFuZCBpbmRlZmF0aWdhYmxlIGdlbmVyYXRpb24gb2Yga25vd2xlZGdlLCBleGNlZWRzIHRoZSBzaG9ydCB2ZWhlbWVuY2Ugb2YgYW55IGNhcm5hbCBwbGVhc3VyZS4=", &result)); // Not Null Terminated!
  mTEST_ASSERT_FALSE(result);

  mTEST_ASSERT_SUCCESS(mBase64_IsBase64("TW=FuIGlzIGRpc3Rpbmd1aXNoZWQsIG5vdCBvbmx5IGJ5IGhpcyByZWFzb24sIGJ1dCBieSB0aGlzIHNpbmd1bGFyIHBhc3Npb24gZnJvbSBvdGhlciBhbmltYWxzLCB3aGljaCBpcyBhIGx1c3Qgb2YgdGhlIG1pbmQsIHRoYXQgYnkgYSBwZXJzZXZlcmFuY2Ugb2YgZGVsaWdodCBpbiB0aGUgY29udGludWVkIGFuZCBpbmRlZmF0aWdhYmxlIGdlbmVyYXRpb24gb2Yga25vd2xlZGdlLCBleGNlZWRzIHRoZSBzaG9ydCB2ZWhlbWVuY2Ugb2YgYW55IGNhcm5hbCBwbGVhc3VyZS4=", &result)); // Not Null Terminated!
  mTEST_ASSERT_FALSE(result);

  mTEST_ASSERT_SUCCESS(mBase64_IsBase64(".TWFuIGlzIGRpc3Rpbmd1aXNoZWQsIG5vdCBvbmx5IGJ5IGhpcyByZWFzb24sIGJ1dCBieSB0aGlzIHNpbmd1bGFyIHBhc3Npb24gZnJvbSBvdGhlciBhbmltYWxzLCB3aGljaCBpcyBhIGx1c3Qgb2YgdGhlIG1pbmQsIHRoYXQgYnkgYSBwZXJzZXZlcmFuY2Ugb2YgZGVsaWdodCBpbiB0aGUgY29udGludWVkIGFuZCBpbmRlZmF0aWdhYmxlIGdlbmVyYXRpb24gb2Yga25vd2xlZGdlLCBleGNlZWRzIHRoZSBzaG9ydCB2ZWhlbWVuY2Ugb2YgYW55IGNhcm5hbCBwbGVhc3VyZS4=", &result)); // Not Null Terminated!
  mTEST_ASSERT_FALSE(result);

  mTEST_ASSERT_SUCCESS(mBase64_IsBase64(".TWFuIGlzIGRpc3Rpbmd1aXNoZWQsIG5vdCBvbmx5IGJ5IGhpcyByZWFzb24sIGJ1dCBieSB0aGlzIHNpbmd1bGFyIHBhc3Npb24gZnJvbSBvdGhlciBhbmltYWxzLCB3aGljaCBpcyBhIGx1c3Qgb2YgdGhlIG1pbmQsIHRoYXQgYnkgYSBwZXJzZXZlcmFuY2Ugb2YgZGVsaWdodCBpbiB0aGUgY29udGludWVkIGFuZCBpbmRlZmF0aWdhYmxlIGdlbmVyYXRpb24gb2Yga25vd2xlZGdlLCBleGNlZWRzIHRoZSBzaG9ydCB2ZWhlbWVuY2Ugb2YgYW55IGNhcm5hbCBwbGVhc3VyZS4:=", &result)); // Not Null Terminated!
  mTEST_ASSERT_FALSE(result);

  mTEST_ASSERT_SUCCESS(mBase64_IsBase64("TWFuIGlzIGRpc3Rpbmd1aXNoZWQsIG5vdCBvbmx5IGJ5IGhpcyByZWFzb24sIGJ1dCBieSB0aGlzIHNpbmd1bGFyIHBhc3Npb24gZnJvbSBvdGhlciBhbmltYWxzLCB3aGljaCBpcyBhIGx1c3Qgb2YgdGhlIG1pbmQsIHRoYXQgYnkgYSBwZXJzZXZlc@mFuY2Ugb2YgZGVsaWdodCBpbiB0aGUgY29udGludWVkIGFuZCBpbmRlZmF0aWdhYmxlIGdlbmVyYXRpb24gb2Yga25vd2xlZGdlLCBleGNlZWRzIHRoZSBzaG9ydCB2ZWhlbWVuY2Ugb2YgYW55IGNhcm5hbCBwbGVhc3VyZS4=", &result)); // Not Null Terminated!
  mTEST_ASSERT_FALSE(result);

  mTEST_ASSERT_SUCCESS(mBase64_IsBase64("TWFuIGlzIGRpc3Rpbmd1aXNoZWQsIG5vdCBvbmx5IGJ5IGhpcyByZWFzb24sIGJ1dCBieSB0aGlzIHNpbmd1bGFyIHBhc3Npb24gZnJvbSBvdGhlciBhbmltYWxzLCB3aGljaCBpcyBhIGx1c3Qgb2YgdGhlIG1pbmQsIHRoYXQgYnkgYSBwZXJzZXZlcm[FuY2Ugb2YgZGVsaWdodCBpbiB0aGUgY29udGludWVkIGFuZCBpbmRlZmF0aWdhYmxlIGdlbmVyYXRpb24gb2Yga25vd2xlZGdlLCBleGNlZWRzIHRoZSBzaG9ydCB2ZWhlbWVuY2Ugb2YgYW55IGNhcm5hbCBwbGVhc3VyZS4=", &result)); // Not Null Terminated!
  mTEST_ASSERT_FALSE(result);

  mTEST_ASSERT_SUCCESS(mBase64_IsBase64("TWFuIGlzIGRpc3Rpbmd1aXNoZWQsIG5vdCBvbmx5IGJ5IGhpcyByZWFzb24sIGJ1dCBieSB0aGlzIHNpbmd1bGFyIHBhc3Npb24gZnJvbSBvdGhlciBhbmltYWxzLCB3aGljaCBpcyBhIGx1c3Qgb2YgdGhlIG1pbmQsIHRoYXQgYnkgYSBwZXJzZXZlcmF`uY2Ugb2YgZGVsaWdodCBpbiB0aGUgY29udGludWVkIGFuZCBpbmRlZmF0aWdhYmxlIGdlbmVyYXRpb24gb2Yga25vd2xlZGdlLCBleGNlZWRzIHRoZSBzaG9ydCB2ZWhlbWVuY2Ugb2YgYW55IGNhcm5hbCBwbGVhc3VyZS4=", &result)); // Not Null Terminated!
  mTEST_ASSERT_FALSE(result);

  mTEST_ASSERT_SUCCESS(mBase64_IsBase64("TWFuIGlzIGRpc3Rpbmd1aXNoZWQsIG5vdCBvbmx5IGJ5IGhpcyByZWFzb24sIGJ1dCBieSB0aGlzIHNpbmd1bGFyIHBhc3Npb24gZnJvbSBvdGhlciBhbmltYWxzLCB3aGljaCBpcyBhIGx1c3Qgb2YgdGhlIG1pbmQsIHRoYXQgYnkgYSBwZXJzZXZlcmFu{Y2Ugb2YgZGVsaWdodCBpbiB0aGUgY29udGludWVkIGFuZCBpbmRlZmF0aWdhYmxlIGdlbmVyYXRpb24gb2Yga25vd2xlZGdlLCBleGNlZWRzIHRoZSBzaG9ydCB2ZWhlbWVuY2Ugb2YgYW55IGNhcm5hbCBwbGVhc3VyZS4=", &result)); // Not Null Terminated!
  mTEST_ASSERT_FALSE(result);

  mTEST_ASSERT_SUCCESS(mBase64_IsBase64("TWFuIGlzIGRpc3Rpbmd1aXNoZWQsIG5vdCBvbmx5IGJ5IGhpcyByZWFzb24sIGJ1dCBieSB0aGlzIHNpbmd1bGFyIHBhc3Npb24gZnJvbSBvdGhlciBhbmltYWxzLCB3aGljaCBpcyBhIGx1c3Qgb2YgdGhlIG1pbmQsIHRoYXQgYnkgYSBwZXJzZXZlcmFu🌵🦎🎅testמⴲxY2Ugb2YgZGVsaWdodCBpbiB0aGUgY29udGludWVkIGFuZCBpbmRlZmF0aWdhYmxlIGdlbmVyYXRpb24gb2Yga25vd2xlZGdlLCBleGNlZWRzIHRoZSBzaG9ydCB2ZWhlbWVuY2Ugb2YgYW55IGNhcm5hbCBwbGVhc3VyZS4=", &result)); // Not Null Terminated!
  mTEST_ASSERT_FALSE(result);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mBase64, TestEncodedDecodedLenght)
{
  mTEST_ALLOCATOR_SETUP();

  mString nullString;
  size_t length;

  mTEST_ASSERT_EQUAL(mR_ArgumentNull, mBase64_GetDecodedLength(nullString, nullptr));
  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mBase64_GetDecodedLength(nullString, &length));
  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mBase64_GetDecodedLength("🌵🦎🎅testמⴲx", &length));

  mTEST_ASSERT_EQUAL(mR_ArgumentNull, mBase64_GetEncodedLength(0, nullptr));

  mTEST_ASSERT_SUCCESS(mBase64_GetDecodedLength("", &length));
  mTEST_ASSERT_EQUAL(length, 0);

  mTEST_ASSERT_SUCCESS(mBase64_GetDecodedLength("=", &length));
  mTEST_ASSERT_EQUAL(length, 0);

  mTEST_ASSERT_SUCCESS(mBase64_GetDecodedLength("==", &length));
  mTEST_ASSERT_EQUAL(length, 0);

  mTEST_ASSERT_SUCCESS(mBase64_GetDecodedLength("===", &length));
  mTEST_ASSERT_EQUAL(length, 0);

  mTEST_ASSERT_EQUAL(mR_ResourceInvalid, mBase64_GetDecodedLength("Q", &length));

  mTEST_ASSERT_SUCCESS(mBase64_GetDecodedLength("QQ", &length));
  mTEST_ASSERT_EQUAL(length, 1);

  mTEST_ASSERT_SUCCESS(mBase64_GetDecodedLength("QQ=", &length));
  mTEST_ASSERT_EQUAL(length, 1);

  mTEST_ASSERT_SUCCESS(mBase64_GetDecodedLength("QQ==", &length));
  mTEST_ASSERT_EQUAL(length, 1);

  mTEST_ASSERT_SUCCESS(mBase64_GetDecodedLength("QQ===", &length));
  mTEST_ASSERT_EQUAL(length, 1);

  mTEST_ASSERT_SUCCESS(mBase64_GetDecodedLength("QUE", &length));
  mTEST_ASSERT_EQUAL(length, 2);

  mTEST_ASSERT_SUCCESS(mBase64_GetDecodedLength("QUE=", &length));
  mTEST_ASSERT_EQUAL(length, 2);

  mTEST_ASSERT_SUCCESS(mBase64_GetDecodedLength("QUE==", &length));
  mTEST_ASSERT_EQUAL(length, 2);

  mTEST_ASSERT_SUCCESS(mBase64_GetDecodedLength("QUE===", &length));
  mTEST_ASSERT_EQUAL(length, 2);

  mTEST_ASSERT_SUCCESS(mBase64_GetDecodedLength("TWFu", &length));
  mTEST_ASSERT_EQUAL(length, 3);

  mTEST_ASSERT_SUCCESS(mBase64_GetDecodedLength("TWFu=", &length));
  mTEST_ASSERT_EQUAL(length, 3);

  mTEST_ASSERT_SUCCESS(mBase64_GetDecodedLength("TWFu==", &length));
  mTEST_ASSERT_EQUAL(length, 3);

  mTEST_ASSERT_SUCCESS(mBase64_GetDecodedLength("TWFu===", &length));
  mTEST_ASSERT_EQUAL(length, 3);

  mTEST_ASSERT_SUCCESS(mBase64_GetDecodedLength("IV9GYTAyZg", &length));
  mTEST_ASSERT_EQUAL(length, 7);

  mTEST_ASSERT_SUCCESS(mBase64_GetDecodedLength("IV9GYTAyZg=", &length));
  mTEST_ASSERT_EQUAL(length, 7);

  mTEST_ASSERT_SUCCESS(mBase64_GetDecodedLength("IV9GYTAyZg==", &length));
  mTEST_ASSERT_EQUAL(length, 7);

  mTEST_ASSERT_SUCCESS(mBase64_GetDecodedLength("IV9GYTAyZg===", &length));
  mTEST_ASSERT_EQUAL(length, 7);

  mTEST_ASSERT_EQUAL(mR_ResourceInvalid, mBase64_GetDecodedLength("IV9GYTAyZ", &length));

  mTEST_ASSERT_SUCCESS(mBase64_GetEncodedLength(0, &length));
  mTEST_ASSERT_EQUAL(length, 0);

  mTEST_ASSERT_SUCCESS(mBase64_GetEncodedLength(1, &length));
  mTEST_ASSERT_EQUAL(length, 4);

  mTEST_ASSERT_SUCCESS(mBase64_GetDecodedLength("TWFuIGlzIGRpc3Rpbmd1aXNoZWQsIG5vdCBvbmx5IGJ5IGhpcyByZWFzb24sIGJ1dCBieSB0aGlzIHNpbmd1bGFyIHBhc3Npb24gZnJvbSBvdGhlciBhbmltYWxzLCB3aGljaCBpcyBhIGx1c3Qgb2YgdGhlIG1pbmQsIHRoYXQgYnkgYSBwZXJzZXZlcmFuY2Ugb2YgZGVsaWdodCBpbiB0aGUgY29udGludWVkIGFuZCBpbmRlZmF0aWdhYmxlIGdlbmVyYXRpb24gb2Yga25vd2xlZGdlLCBleGNlZWRzIHRoZSBzaG9ydCB2ZWhlbWVuY2Ugb2YgYW55IGNhcm5hbCBwbGVhc3VyZS4=", &length)); // Not Null Terminated!
  mTEST_ASSERT_EQUAL(length, 269);

  mTEST_ASSERT_SUCCESS(mBase64_GetDecodedLength("TWFuIGlzIGRpc3Rpbmd1aXNoZWQsIG5vdCBvbmx5IGJ5IGhpcyByZWFzb24sIGJ1dCBieSB0aGlzIHNpbmd1bGFyIHBhc3Npb24gZnJvbSBvdGhlciBhbmltYWxzLCB3aGljaCBpcyBhIGx1c3Qgb2YgdGhlIG1pbmQsIHRoYXQgYnkgYSBwZXJzZXZlcmFuY2Ugb2YgZGVsaWdodCBpbiB0aGUgY29udGludWVkIGFuZCBpbmRlZmF0aWdhYmxlIGdlbmVyYXRpb24gb2Yga25vd2xlZGdlLCBleGNlZWRzIHRoZSBzaG9ydCB2ZWhlbWVuY2Ugb2YgYW55IGNhcm5hbCBwbGVhc3VyZS4", &length)); // Not Null Terminated!
  mTEST_ASSERT_EQUAL(length, 269);

  mTEST_ASSERT_SUCCESS(mBase64_GetDecodedLength("TWFuIGlzIGRpc3Rpbmd1aXNoZWQsIG5vdCBvbmx5IGJ5IGhpcyByZWFzb24sIGJ1dCBieSB0aGlzIHNpbmd1bGFyIHBhc3Npb24gZnJvbSBvdGhlciBhbmltYWxzLCB3aGljaCBpcyBhIGx1c3Qgb2YgdGhlIG1pbmQsIHRoYXQgYnkgYSBwZXJzZXZlcmFuY2Ugb2YgZGVsaWdodCBpbiB0aGUgY29udGludWVkIGFuZCBpbmRlZmF0aWdhYmxlIGdlbmVyYXRpb24gb2Yga25vd2xlZGdlLCBleGNlZWRzIHRoZSBzaG9ydCB2ZWhlbWVuY2Ugb2YgYW55IGNhcm5hbCBwbGVhc3VyZS4==", &length)); // Not Null Terminated!
  mTEST_ASSERT_EQUAL(length, 269);

  mTEST_ASSERT_SUCCESS(mBase64_GetDecodedLength("TWFuIGlzIGRpc3Rpbmd1aXNoZWQsIG5vdCBvbmx5IGJ5IGhpcyByZWFzb24sIGJ1dCBieSB0aGlzIHNpbmd1bGFyIHBhc3Npb24gZnJvbSBvdGhlciBhbmltYWxzLCB3aGljaCBpcyBhIGx1c3Qgb2YgdGhlIG1pbmQsIHRoYXQgYnkgYSBwZXJzZXZlcmFuY2Ugb2YgZGVsaWdodCBpbiB0aGUgY29udGludWVkIGFuZCBpbmRlZmF0aWdhYmxlIGdlbmVyYXRpb24gb2Yga25vd2xlZGdlLCBleGNlZWRzIHRoZSBzaG9ydCB2ZWhlbWVuY2Ugb2YgYW55IGNhcm5hbCBwbGVhc3VyZS4===", &length)); // Not Null Terminated!
  mTEST_ASSERT_EQUAL(length, 269);

  mTEST_ASSERT_SUCCESS(mBase64_GetEncodedLength(269, &length));
  mTEST_ASSERT_EQUAL(length, 360);

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mBase64, TestEncodeDecode)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<uint8_t> data;
  size_t length = (size_t)-1;
  mString text, text2;

  mTEST_ASSERT_SUCCESS(mString_Create(&text, "", pAllocator));
  mTEST_ASSERT_SUCCESS(mBase64_Decode(text, &data, pAllocator, &length));
  mTEST_ASSERT_EQUAL(length, 0);

  mTEST_ASSERT_SUCCESS(mString_Create(&text, "QQ", pAllocator));
  mTEST_ASSERT_SUCCESS(mBase64_Decode(text, &data, pAllocator, &length));
  mTEST_ASSERT_EQUAL(length, 1);
  mTEST_ASSERT_EQUAL(data.GetPointer()[0], 'A');
  
  mTEST_ASSERT_SUCCESS(mString_Create(&text, "QQ=", pAllocator));
  mTEST_ASSERT_SUCCESS(mBase64_Decode(text, &data, pAllocator, &length));
  mTEST_ASSERT_EQUAL(length, 1);
  mTEST_ASSERT_EQUAL(data.GetPointer()[0], 'A');
  
  mTEST_ASSERT_SUCCESS(mString_Create(&text, "QQ==", pAllocator));
  mTEST_ASSERT_SUCCESS(mBase64_Decode(text, &data, pAllocator, &length));
  mTEST_ASSERT_EQUAL(length, 1);
  mTEST_ASSERT_EQUAL(data.GetPointer()[0], 'A');
  mTEST_ASSERT_SUCCESS(mBase64_Encode(data, length, &text2, pAllocator));
  mTEST_ASSERT_TRUE(text == text2);

  for (size_t i = 0; i < 16; i++)
  {
    mTEST_ASSERT_SUCCESS(mString_Append(text, "=", 1));
    mTEST_ASSERT_SUCCESS(mBase64_Decode(text, &data, pAllocator, &length));
    mTEST_ASSERT_EQUAL(length, 1);
    mTEST_ASSERT_EQUAL(data.GetPointer()[0], 'A');
  }

  mTEST_ASSERT_SUCCESS(mString_Create(&text, "TWFuIGlzIGRpc3Rpbmd1aXNoZWQsIG5vdCBvbmx5IGJ5IGhpcyByZWFzb24sIGJ1dCBieSB0aGlzIHNpbmd1bGFyIHBhc3Npb24gZnJvbSBvdGhlciBhbmltYWxzLCB3aGljaCBpcyBhIGx1c3Qgb2YgdGhlIG1pbmQsIHRoYXQgYnkgYSBwZXJzZXZlcmFuY2Ugb2YgZGVsaWdodCBpbiB0aGUgY29udGludWVkIGFuZCBpbmRlZmF0aWdhYmxlIGdlbmVyYXRpb24gb2Yga25vd2xlZGdlLCBleGNlZWRzIHRoZSBzaG9ydCB2ZWhlbWVuY2Ugb2YgYW55IGNhcm5hbCBwbGVhc3VyZS4=", pAllocator));
  mTEST_ASSERT_SUCCESS(mBase64_Decode(text, &data, pAllocator, &length));
  mTEST_ASSERT_EQUAL(length, 269);
  mTEST_ASSERT_SUCCESS(mBase64_Encode(data, length, &text2, pAllocator));
  mTEST_ASSERT_TRUE(text == text2);
  mTEST_ASSERT_SUCCESS(mString_Create(&text, "Man is distinguished, not only by his reason, but by this singular passion from other animals, which is a lust of the mind, that by a perseverance of delight in the continued and indefatigable generation of knowledge, exceeds the short vehemence of any carnal pleasure.", pAllocator));
  mTEST_ASSERT_TRUE(memcmp(text.c_str(), data.GetPointer(), text.bytes - 1) == 0);

  mTEST_ASSERT_SUCCESS(mString_Create(&text, "Q===", pAllocator));
  mTEST_ASSERT_EQUAL(mR_ResourceInvalid, mBase64_Decode(text, &data, pAllocator, &length));

  mTEST_ASSERT_SUCCESS(mString_Create(&text, "Q=Q=", pAllocator));
  mTEST_ASSERT_EQUAL(mR_ResourceInvalid, mBase64_Decode(text, &data, pAllocator, &length));

  mTEST_ASSERT_SUCCESS(mString_Create(&text, "Q=QQ=", pAllocator));
  mTEST_ASSERT_EQUAL(mR_ResourceInvalid, mBase64_Decode(text, &data, pAllocator, &length));

  mTEST_ASSERT_SUCCESS(mString_Create(&text, "Q=QQQ=", pAllocator));
  mTEST_ASSERT_EQUAL(mR_ResourceInvalid, mBase64_Decode(text, &data, pAllocator, &length));

  mTEST_ASSERT_SUCCESS(mString_Create(&text, "=QQ=", pAllocator));
  mTEST_ASSERT_EQUAL(mR_ResourceInvalid, mBase64_Decode(text, &data, pAllocator, &length));

  mTEST_ASSERT_SUCCESS(mString_Create(&text, "@QQ=", pAllocator));
  mTEST_ASSERT_EQUAL(mR_ResourceInvalid, mBase64_Decode(text, &data, pAllocator, &length));

  mTEST_ASSERT_SUCCESS(mString_Create(&text, "QQQQ====QQQQ", pAllocator));
  mTEST_ASSERT_EQUAL(mR_ResourceInvalid, mBase64_Decode(text, &data, pAllocator, &length));

  mTEST_ASSERT_SUCCESS(mString_Create(&text, "====QQQQ", pAllocator));
  mTEST_ASSERT_EQUAL(mR_ResourceInvalid, mBase64_Decode(text, &data, pAllocator, &length));

  mTEST_ALLOCATOR_ZERO_CHECK();
}
