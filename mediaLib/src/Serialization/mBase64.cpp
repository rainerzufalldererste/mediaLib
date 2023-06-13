#include "mBase64.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "YTHgJYRdjfwR0TKL8WQR15znX8VTxKksRxESFI7muAxXROAJIcir/ylvDsYnNQd+xB6cHbLOHVOkZjcr"
#endif

mFUNCTION(mBase64_IsBase64, const mString &text, OUT bool *pIsBase64)
{
  mFUNCTION_SETUP();

  mERROR_IF(pIsBase64 == nullptr, mR_ArgumentNull);

  if (text.c_str() == nullptr || text.count != text.bytes)
  {
    *pIsBase64 = false;
    mRETURN_SUCCESS();
  }

  *pIsBase64 = true;
  bool inPadding = false;

  for (const auto &&_char : text)
  {
    const char c = *_char.character;

    if (_char.characterSize != 1 || !((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '/' && c <= '9') || c == '+' || c == '=') || (inPadding && c != '='))
    {
      *pIsBase64 = false;
      mRETURN_SUCCESS();
    }

    if (c == '=')
      inPadding = true;
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mBase64_GetDecodedLength, const mString &text, OUT size_t *pLength)
{
  mFUNCTION_SETUP();

  mERROR_IF(pLength == nullptr, mR_ArgumentNull);
  mERROR_IF(text.c_str() == nullptr || text.count != text.bytes, mR_InvalidParameter);

  size_t encodedLength = mMin(text.bytes, text.bytes - 1);

  // Remove Padding.
  while (encodedLength > 0 && text.c_str()[encodedLength - 1] == '=')
    encodedLength--;

  mERROR_IF(encodedLength % 4 == 1, mR_ResourceInvalid);

  *pLength = encodedLength / 4 * 3 + (size_t)mMax((int64_t)0, ((int64_t)(encodedLength % 4) - 1));

  mRETURN_SUCCESS();
}

mFUNCTION(mBase64_GetEncodedLength, const size_t inputLength, OUT size_t *pOutChars)
{
  mFUNCTION_SETUP();

  mERROR_IF(pOutChars == nullptr, mR_ArgumentNull);

  *pOutChars = 4 * (mMax(inputLength, inputLength + 2) / 3);

  mRETURN_SUCCESS();
}

mFUNCTION(mBase64_Decode, const mString &text, OUT uint8_t *pData, const size_t capacity)
{
  mFUNCTION_SETUP();

  mERROR_IF(pData == nullptr, mR_ArgumentNull);

  size_t requiredCapacity;
  mERROR_CHECK(mBase64_GetDecodedLength(text, &requiredCapacity));
  mERROR_IF(requiredCapacity > capacity, mR_ArgumentOutOfBounds);
  
  // Which codepoint maps to which hextet?
  // `0xFF` represents invalid codepoints (including the padding codepoint '=', which is handled separately)
  // Doesn't need to be of size 0x100, since we discard multi-byte characters and U+0000 - U+007F are the only single byte Unicode characters.
  const uint8_t lut[0x80] =
  {
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x3E, 0xFF, 0xFF, 0xFF, 0x3F,
    0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E,
    0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F, 0x30, 0x31, 0x32, 0x33, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF
  };

  size_t count = 0;
  char chars[4];
  uint8_t *pNext = pData;
  bool inPadding = false;

  for (const auto &&_char : text)
  {
    mERROR_IF(_char.characterSize != 1, mR_ResourceInvalid);

    chars[count] = *_char.character;
    count++;

    if (count == 4)
    {
      uint32_t decoded[4];

      decoded[0] = lut[(uint8_t)chars[0]];
      decoded[1] = lut[(uint8_t)chars[1]];
      decoded[2] = lut[(uint8_t)chars[2]];
      decoded[3] = lut[(uint8_t)chars[3]];

      size_t unpaddedCount = count;
      count = 0;

      for (size_t i = 0; i < 4; i++)
      {
        if (inPadding)
        {
          mERROR_IF(chars[i] != '=', mR_ResourceInvalid);
          unpaddedCount--;
        }
        else if (decoded[i] == 0xFF)
        {
          mERROR_IF(chars[i] != '=', mR_ResourceInvalid);
          inPadding = true;
          unpaddedCount--;
        }
      }

      // If the entire input consists of padding, simply exit here.
      if (unpaddedCount == 0)
      {
        pNext += 3;
        continue;
      }

      const uint32_t out = (decoded[0] << 3 * 6) | (decoded[1] << 12) | (decoded[2] << 6) | decoded[3];

      pNext[0] = (uint8_t)(out >> 0x10);

      if (unpaddedCount > 2)
      {
        pNext[1] = (uint8_t)(out >> 0x8);

        if (unpaddedCount > 3)
          pNext[2] = (uint8_t)out;
      }

      pNext += 3;
    }
  }

  if (count > 0)
  {
    if (count == 1)
    {
      mERROR_IF(chars[0] != '=', mR_ResourceInvalid);
    }
    else
    {
      uint32_t decoded[4];

      decoded[0] = lut[(uint8_t)chars[0]];
      decoded[1] = lut[(uint8_t)chars[1]];
      decoded[2] = lut[(uint8_t)chars[2]];
      decoded[3] = lut[(uint8_t)chars[3]];

      size_t unpaddedCount = count;

      for (size_t i = 0; i < 2; i++)
      {
        if (inPadding)
        {
          mERROR_IF(chars[i] != '=', mR_ResourceInvalid);
          unpaddedCount--;
        }
        else if (decoded[i] == 0xFF)
        {
          mERROR_IF(chars[i] != '=', mR_ResourceInvalid);
          inPadding = true;
          unpaddedCount--;
        }
      }

      for (size_t i = 2; i < 4; i++)
      {
        if (count <= i)
        {
          decoded[i] = 0;
        }
        else if (inPadding)
        {
          mERROR_IF(chars[i] != '=', mR_ResourceInvalid);
          unpaddedCount--;
        }
        else if (decoded[i] == 0xFF)
        {
          mERROR_IF(chars[i] != '=', mR_ResourceInvalid);
          inPadding = true;
          unpaddedCount--;
        }
      }

      // If the entire input consists of padding, simply exit here.
      mERROR_IF(unpaddedCount == 0, mR_Success);

      const uint32_t out = (decoded[0] << 3 * 6) | (decoded[1] << 12) | (decoded[2] << 6) | decoded[3];

      pNext[0] = (uint8_t)(out >> 0x10);

      if (unpaddedCount > 2)
      {
        pNext[1] = (uint8_t)(out >> 0x8);

        if (unpaddedCount > 3)
          pNext[2] = (uint8_t)out;
      }
    }
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mBase64_Decode, const mString &text, OUT mPtr<uint8_t> &data, const size_t capacity)
{
  mFUNCTION_SETUP();

  mERROR_IF(data == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mBase64_Decode(text, data.GetPointer(), capacity));

  mRETURN_SUCCESS();
}

mFUNCTION(mBase64_Decode, const mString &text, OUT mPtr<uint8_t> *pData, IN mAllocator *pAllocator, OUT OPTIONAL size_t *pCapacity /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_IF(pData == nullptr || pCapacity == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mBase64_GetDecodedLength(text, pCapacity));
  
  mDEFER_CALL_ON_ERROR(pData, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Allocate(pData, pAllocator, mMax(1ULL, *pCapacity)));

  if (*pCapacity > 0)
    mERROR_CHECK(mBase64_Decode(text, pData->GetPointer(), *pCapacity));

  mRETURN_SUCCESS();
}

mFUNCTION(mBase64_Encode, IN const uint8_t *pData, const size_t bytes, OUT mString *pEncoded, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pData == nullptr || pEncoded == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mString_Create(pEncoded, "", pAllocator));

  size_t requiredLength = 0;
  mERROR_CHECK(mBase64_GetEncodedLength(bytes, &requiredLength));

  mERROR_CHECK(mString_Reserve(*pEncoded, requiredLength + 1));

  mDEFER_ON_ERROR(pEncoded->text[0] = '\0');
  pEncoded->text[requiredLength] = '\0';

  char *next = pEncoded->text;

  // Encode To Base64 into `next`.
  {
    const char lut[] =
    {
      'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
      'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
      'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
      'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
      'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
      'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
      'w', 'x', 'y', 'z', '0', '1', '2', '3',
      '4', '5', '6', '7', '8', '9', '+', '/'
    };

    for (size_t i = 0; i < bytes; i += 3)
    {
      uint32_t data[3];

      data[0] = (i + 0) < bytes ? pData[i] : 0;
      data[1] = (i + 1) < bytes ? pData[i + 1] : 0;
      data[2] = (i + 2) < bytes ? pData[i + 2] : 0;

      const uint32_t triple = (data[0] << 0x10) + (data[1] << 0x08) + data[2];

      next[0] = lut[(triple >> 18) & 0x3F];
      next[1] = lut[(triple >> 12) & 0x3F];
      next[2] = lut[(triple >> 6) & 0x3F];
      next[3] = lut[triple & 0x3F];

      next += 4;
    }
  }

  // Add Padding.
  {
    const size_t bytesMod3 = bytes % 3;

    if (bytesMod3 != 0)
      for (size_t i = 0; i < 3 - bytesMod3; i++)
        next[-1 - i] = '=';
  }

  // Change String Length & Count.
  pEncoded->bytes = pEncoded->count = requiredLength + 1;

  mRETURN_SUCCESS();
}

mFUNCTION(mBase64_Encode, const mPtr<uint8_t> &data, const size_t bytes, OUT mString *pEncoded, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(data == nullptr || pEncoded == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mBase64_Encode(data.GetPointer(), bytes, pEncoded, pAllocator));

  mRETURN_SUCCESS();
}
