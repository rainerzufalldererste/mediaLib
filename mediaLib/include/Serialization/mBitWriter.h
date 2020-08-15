#ifndef mBitWriter_h__
#define mBitWriter_h__

#include "mediaLib.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "NNieKMs2oKEuILOy0M0hIG/AuankgWvftehe2A24dKTxUi+q0LRqMbROOKoZ4i0TRc7gQeWTqThQT2od"
#endif

struct mBitWriter
{
  uint8_t *pData;
  size_t size;
  size_t offsetByte;
  uint8_t offsetBit;
};

inline mFUNCTION(mBitWriter_Create, OUT mBitWriter *pBitWriter, IN uint8_t *pData, const size_t size, const bool setZero = true)
{
  mFUNCTION_SETUP();

  mERROR_IF(pBitWriter == nullptr || pData == nullptr, mR_ArgumentNull);

  pBitWriter->pData = pData;
  pBitWriter->size = size;
  pBitWriter->offsetByte = 0;
  pBitWriter->offsetBit = 0;

  if (setZero)
    mMemset(pData, size, 0);

  mRETURN_SUCCESS();
}

inline mFUNCTION(mBitWriter_Destroy, IN_OUT mBitWriter *pBitWriter)
{
  mFUNCTION_SETUP();

  mERROR_IF(pBitWriter == nullptr, mR_ArgumentNull);

  pBitWriter->pData = nullptr;
  pBitWriter->size = 0;

  mRETURN_SUCCESS();
}

inline mFUNCTION(mBitWriter_WriteNextBit, mBitWriter &bitWriter, const bool bit)
{
  mFUNCTION_SETUP();

  mERROR_IF(bitWriter.offsetByte >= bitWriter.size, mR_EndOfStream);

  bitWriter.pData[bitWriter.offsetByte] |= (bit << bitWriter.offsetBit);

  bitWriter.offsetBit = (bitWriter.offsetBit + 1) % CHAR_BIT;
  bitWriter.offsetByte += (bitWriter.offsetBit == 0);

  mRETURN_SUCCESS();
}

inline mFUNCTION(mBitWriter_SetToBit, mBitWriter &bitWriter, const size_t bitOffset)
{
  mFUNCTION_SETUP();

  const size_t byteOffset = bitOffset / CHAR_BIT;

  mERROR_IF(byteOffset >= bitWriter.size, mR_ArgumentOutOfBounds);

  bitWriter.offsetByte = byteOffset;
  bitWriter.offsetBit = bitOffset % CHAR_BIT;

  mRETURN_SUCCESS();
}

inline mFUNCTION(mBitWriter_SetToByte, mBitWriter &bitWriter, const size_t byteOffset)
{
  mFUNCTION_SETUP();

  mERROR_IF(byteOffset >= bitWriter.size, mR_ArgumentOutOfBounds);

  bitWriter.offsetByte = byteOffset;
  bitWriter.offsetBit = 0;

  mRETURN_SUCCESS();
}

inline mFUNCTION(mBitWriter_Reset, mBitWriter &bitWriter)
{
  return mBitWriter_SetToByte(bitWriter, 0);
}

inline mFUNCTION(mBitWriter_SetLength, mBitWriter &bitWriter, const size_t newLength)
{
  mFUNCTION_SETUP();

  mERROR_IF(bitWriter.offsetByte >= newLength, mR_ResourceStateInvalid);

  bitWriter.size = newLength;

  mRETURN_SUCCESS();
}

inline mFUNCTION(mBitWriter_WriteBits, mBitWriter &bitWriter, const size_t value, const size_t bits)
{
  mFUNCTION_SETUP();

  mERROR_IF(bits > sizeof(value) * CHAR_BIT, mR_ArgumentOutOfBounds);
  mERROR_IF(bitWriter.offsetByte * CHAR_BIT + bitWriter.offsetBit + bits >= bitWriter.size * CHAR_BIT, mR_EndOfStream);

  size_t v = value;

  for (size_t i = 0; i < bits; i++)
  {
    mERROR_CHECK(mBitWriter_WriteNextBit(bitWriter, (bool)(v & 1)));
    v >>= 1;
  }

  mRETURN_SUCCESS();
}

#endif // mBitWriter_h__
