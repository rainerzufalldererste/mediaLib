#ifndef mBase64_h__
#define mBase64_h__

#include "mediaLib.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "grYEZGUNPpWvT0IUQRQrvc0wKs4/L9/GuxpKMHKMehW+ci24u7wSgwfld722Ch06BmYAVqBFR6tvA7Mx"
#endif

// mBase64 is designed to be reliable and to validate inputs. It applies padding to the Base64 strings.
// It's not amazingly fast. We may add an SIMD variant for ultra-fast en-/decoding capabilities in the future.

mFUNCTION(mBase64_IsBase64, const mString &text, OUT bool *pIsBase64);
mFUNCTION(mBase64_GetDecodedLength, const mString &text, OUT size_t *pLength);
mFUNCTION(mBase64_GetEncodedLength, const size_t inputLength, OUT size_t *pOutChars);

mFUNCTION(mBase64_Decode, const mString &text, OUT uint8_t *pData, const size_t capacity);
mFUNCTION(mBase64_Decode, const mString &text, OUT mPtr<uint8_t> &data, const size_t capacity);
mFUNCTION(mBase64_Decode, const mString &text, OUT mPtr<uint8_t> *pData, IN mAllocator *pAllocator, OUT OPTIONAL size_t *pCapacity = nullptr);
mFUNCTION(mBase64_Encode, IN const uint8_t *pData, const size_t bytes, OUT mString *pEncoded, IN mAllocator *pAllocator);
mFUNCTION(mBase64_Encode, const mPtr<uint8_t> &data, const size_t bytes, OUT mString *pEncoded, IN mAllocator *pAllocator);

#endif // mBase64_h__
