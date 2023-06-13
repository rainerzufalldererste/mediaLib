#ifndef mBitReader_h__
#define mBitReader_h__

#include "mediaLib.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "QR2tcSiGMaGul/AE7vXao88ONATNm0akcUkcmBmTa6LWN2Flzp+k5Kib/pRhsY0FSHz74YaSOGJn//Lw"
#endif

struct mBitReader
{
  const uint8_t *pData;
  size_t size;
  size_t offsetByte;
  uint8_t offsetBit;
};

inline mFUNCTION(mBitReader_Create, OUT mBitReader *pBitReader, IN const uint8_t *pData, const size_t size)
{
  mFUNCTION_SETUP();

  mERROR_IF(pBitReader == nullptr || pData == nullptr, mR_ArgumentNull);

  pBitReader->pData = pData;
  pBitReader->size = size;
  pBitReader->offsetByte = 0;
  pBitReader->offsetBit = 0;

  mRETURN_SUCCESS();
}

inline mFUNCTION(mBitReader_Destroy, IN_OUT mBitReader *pBitReader)
{
  mFUNCTION_SETUP();

  mERROR_IF(pBitReader == nullptr, mR_ArgumentNull);

  pBitReader->pData = nullptr;
  pBitReader->size = 0;

  mRETURN_SUCCESS();
}

inline mFUNCTION(mBitReader_ReadNextBit, mBitReader &bitReader, OUT bool &bit)
{
  mFUNCTION_SETUP();

  mERROR_IF(bitReader.offsetByte >= bitReader.size, mR_EndOfStream);

  const uint8_t byte = bitReader.pData[bitReader.offsetByte];
  bit = ((byte >> bitReader.offsetBit) & 1);

  bitReader.offsetBit = (bitReader.offsetBit + 1) % CHAR_BIT;
  bitReader.offsetByte += (bitReader.offsetBit == 0);

  mRETURN_SUCCESS();
}

inline mFUNCTION(mBitReader_SkipBit, mBitReader &bitReader)
{
  bool _unused;

  return mBitReader_ReadNextBit(bitReader, _unused);
}

inline mFUNCTION(mBitReader_SetToBit, mBitReader &bitReader, const size_t bitOffset)
{
  mFUNCTION_SETUP();

  const size_t byteOffset = bitOffset / CHAR_BIT;

  mERROR_IF(byteOffset >= bitReader.size, mR_ArgumentOutOfBounds);

  bitReader.offsetByte = byteOffset;
  bitReader.offsetBit = bitOffset % CHAR_BIT;

  mRETURN_SUCCESS();
}

inline mFUNCTION(mBitReader_SetToByte, mBitReader &bitReader, const size_t byteOffset)
{
  mFUNCTION_SETUP();

  mERROR_IF(byteOffset >= bitReader.size, mR_ArgumentOutOfBounds);

  bitReader.offsetByte = byteOffset;
  bitReader.offsetBit = 0;

  mRETURN_SUCCESS();
}

inline mFUNCTION(mBitReader_Reset, mBitReader &bitReader)
{
  return mBitReader_SetToByte(bitReader, 0);
}

inline mFUNCTION(mBitReader_SetLength, mBitReader &bitReader, const size_t newLength)
{
  mFUNCTION_SETUP();

  mERROR_IF(bitReader.offsetByte >= newLength, mR_ResourceStateInvalid);

  bitReader.size = newLength;

  mRETURN_SUCCESS();
}

inline mFUNCTION(mBitReader_ReadBits, mBitReader &bitReader, OUT size_t &value, const size_t bits)
{
  mFUNCTION_SETUP();

  mERROR_IF(bits > sizeof(value) * CHAR_BIT, mR_ArgumentOutOfBounds);
  mERROR_IF(bitReader.offsetByte * CHAR_BIT + bitReader.offsetBit + bits >= bitReader.size * CHAR_BIT, mR_EndOfStream);

  value = 0;
  bool bit;

  for (size_t i = 0; i < bits; i++)
  {
    mERROR_CHECK(mBitReader_ReadNextBit(bitReader, bit));
    value = (value << 1) | (uint8_t)bit;
  }

  mRETURN_SUCCESS();
}

#endif // mBitReader_h__
