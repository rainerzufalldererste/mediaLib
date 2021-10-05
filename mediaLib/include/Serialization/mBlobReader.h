#ifndef mBlobReader_h__
#define mBlobReader_h__

#include "mediaLib.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "Zmm9r6Qpuy4xnePeCjX934p5r2BOtKkfpuljDIh3WpYzgy/mpC/Hjk+lsqB312MmSfQq0SoyDgJLbuKm"
#endif

struct mBlobReader;

mFUNCTION(mBlobReader_Create, OUT mPtr<mBlobReader> *pReader, IN mAllocator *pAllocator, IN const uint8_t *pBlobData, const size_t blobSize);
mFUNCTION(mBlobReader_Destroy, IN_OUT mPtr<mBlobReader> *pReader);

// This function will also enter baseTypeArrays and arrays.
mFUNCTION(mBlobReader_StepIntoContainer, mPtr<mBlobReader> &reader);

// This function will also exit baseTypeArrays and arrays.
mFUNCTION(mBlobReader_ExitContainer, mPtr<mBlobReader> &reader);

// This function will work on both, baseTypeArrays and arrays.
mFUNCTION(mBlobReader_GetArrayCount, mPtr<mBlobReader> &reader, OUT size_t *pCount);

mFUNCTION(mBlobReader_SkipValue, mPtr<mBlobReader> &reader);
mFUNCTION(mBlobReader_ResetToContainerFront, mPtr<mBlobReader> &reader);

mFUNCTION(mBlobReader_ReadValue, mPtr<mBlobReader> &reader, OUT uint8_t *pValue);
mFUNCTION(mBlobReader_ReadValue, mPtr<mBlobReader> &reader, OUT uint32_t *pValue);
mFUNCTION(mBlobReader_ReadValue, mPtr<mBlobReader> &reader, OUT int32_t *pValue);
mFUNCTION(mBlobReader_ReadValue, mPtr<mBlobReader> &reader, OUT uint64_t *pValue);
mFUNCTION(mBlobReader_ReadValue, mPtr<mBlobReader> &reader, OUT int64_t *pValue);
mFUNCTION(mBlobReader_ReadValue, mPtr<mBlobReader> &reader, OUT float_t *pValue);
mFUNCTION(mBlobReader_ReadValue, mPtr<mBlobReader> &reader, OUT double_t *pValue);
mFUNCTION(mBlobReader_ReadValue, mPtr<mBlobReader> &reader, OUT bool *pValue);
mFUNCTION(mBlobReader_ReadValue, mPtr<mBlobReader> &reader, OUT mVec2s *pValue);
mFUNCTION(mBlobReader_ReadValue, mPtr<mBlobReader> &reader, OUT mVec2i *pValue);
mFUNCTION(mBlobReader_ReadValue, mPtr<mBlobReader> &reader, OUT mVec2f *pValue);
mFUNCTION(mBlobReader_ReadValue, mPtr<mBlobReader> &reader, OUT mVec3f *pValue);
mFUNCTION(mBlobReader_ReadValue, mPtr<mBlobReader> &reader, OUT mVec4f *pValue);
mFUNCTION(mBlobReader_ReadValue, mPtr<mBlobReader> &reader, OUT mVec2d *pValue);
mFUNCTION(mBlobReader_ReadValue, mPtr<mBlobReader> &reader, OUT mVec3d *pValue);
mFUNCTION(mBlobReader_ReadValue, mPtr<mBlobReader> &reader, OUT mVec4d *pValue);

mFUNCTION(mBlobReader_ReadValue, mPtr<mBlobReader> &reader, OUT mString *pString);
mFUNCTION(mBlobReader_ReadValue, mPtr<mBlobReader> &reader, OUT const char **pText, OUT size_t *pLength); // length includes null terminator.

mFUNCTION(mBlobReader_ReadValue, mPtr<mBlobReader> &reader, OUT const uint8_t **ppData, OUT size_t *pBytes);

#endif // mBlobReader_h__
