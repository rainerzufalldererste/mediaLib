#ifndef mBlobWriter_h__
#define mBlobWriter_h__

#include "mediaLib.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "rJ2f/0165tqSOrF50PPwkfcdbxd4tMyN2R6qrxbybEOMz57loNMtdFHswcbvVJwZsbPUu0SOn/8TZO6l"
#endif

struct mBlobWriter;

mFUNCTION(mBlobWriter_Create, OUT mPtr<mBlobWriter> *pWriter, IN mAllocator *pAllocator);
mFUNCTION(mBlobWriter_Destroy, IN_OUT mPtr<mBlobWriter> *pWriter);

// End all containers and retrieve a pointer to the blob data.
mFUNCTION(mBlobWriter_GetData, mPtr<mBlobWriter> &writer, OUT const uint8_t **ppData, OUT size_t *pBytes);

mFUNCTION(mBlobWriter_BeginContainer, mPtr<mBlobWriter> &writer);
mFUNCTION(mBlobWriter_BeginArray, mPtr<mBlobWriter> &writer);
mFUNCTION(mBlobWriter_BeginBaseTypeArray, mPtr<mBlobWriter> &writer);

mFUNCTION(mBlobWriter_EndContainer, mPtr<mBlobWriter> &writer);
mFUNCTION(mBlobWriter_EndArray, mPtr<mBlobWriter> &writer);
mFUNCTION(mBlobWriter_EndBaseTypeArray, mPtr<mBlobWriter> &writer);

mFUNCTION(mBlobWriter_AddValue, mPtr<mBlobWriter> &writer, const uint8_t value);
mFUNCTION(mBlobWriter_AddValue, mPtr<mBlobWriter> &writer, const uint32_t value);
mFUNCTION(mBlobWriter_AddValue, mPtr<mBlobWriter> &writer, const int32_t value);
mFUNCTION(mBlobWriter_AddValue, mPtr<mBlobWriter> &writer, const uint64_t value);
mFUNCTION(mBlobWriter_AddValue, mPtr<mBlobWriter> &writer, const int64_t value);
mFUNCTION(mBlobWriter_AddValue, mPtr<mBlobWriter> &writer, const float_t value);
mFUNCTION(mBlobWriter_AddValue, mPtr<mBlobWriter> &writer, const double_t value);
mFUNCTION(mBlobWriter_AddValue, mPtr<mBlobWriter> &writer, const bool value);
mFUNCTION(mBlobWriter_AddValue, mPtr<mBlobWriter> &writer, const mVec2s value);
mFUNCTION(mBlobWriter_AddValue, mPtr<mBlobWriter> &writer, const mVec2i value);
mFUNCTION(mBlobWriter_AddValue, mPtr<mBlobWriter> &writer, const mVec2f value);
mFUNCTION(mBlobWriter_AddValue, mPtr<mBlobWriter> &writer, const mVec3f value);
mFUNCTION(mBlobWriter_AddValue, mPtr<mBlobWriter> &writer, const mVec4f value);
mFUNCTION(mBlobWriter_AddValue, mPtr<mBlobWriter> &writer, const mVec2d value);
mFUNCTION(mBlobWriter_AddValue, mPtr<mBlobWriter> &writer, const mVec3d value);
mFUNCTION(mBlobWriter_AddValue, mPtr<mBlobWriter> &writer, const mVec4d value);

mFUNCTION(mBlobWriter_AddValue, mPtr<mBlobWriter> &writer, const mString &text);
mFUNCTION(mBlobWriter_AddValue, mPtr<mBlobWriter> &writer, const char *text, const size_t length); // length includes null terminator.

mFUNCTION(mBlobWriter_AddValue, mPtr<mBlobWriter> &writer, const uint8_t *pData, const size_t bytes);

#endif // mBlobWriter_h__
