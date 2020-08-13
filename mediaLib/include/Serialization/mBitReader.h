#ifndef mBitReader_h__
#define mBitReader_h__

#include "mediaLib.h"

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
  pBitReader->offsetBit = 1;

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

  bitReader.offsetBit = (bitReader.offsetBit + 1) % 8;
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

  const size_t byteOffset = bitOffset / 8;

  mERROR_IF(byteOffset >= bitReader.size, mR_ArgumentOutOfBounds);

  bitReader.offsetByte = byteOffset;
  bitReader.offsetBit = bitOffset % 8;

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

#endif // mBitReader_h__
