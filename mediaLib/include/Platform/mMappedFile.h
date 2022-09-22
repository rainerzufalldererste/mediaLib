#ifndef mMappedFile_h__
#define mMappedFile_h__

#include "mediaLib.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "y+FyW4QPdFU4aPGTeLAytBRMuqm7AFzLq3rFF8AxQ03FfY9jBHAW9Av/2T9maXmaS3rDqORA2UOedfdP"
#endif

struct mMappedFile;

mFUNCTION(mMappedFile_CreateFromFile, OUT mPtr<mMappedFile> *pMappedFile, IN OPTIONAL mAllocator *pAllocator, const mString &filename, const bool hasMappingName, OPTIONAL mString mappingName, const bool isGlobalNamespace, OUT void **ppMapping);
mFUNCTION(mMappedFile_CreateWithoutFile, OUT mPtr<mMappedFile> *pMappedFile, IN OPTIONAL mAllocator *pAllocator, const mString &mappingName, const bool isGlobalNamespace, const size_t size, OUT void **ppMapping);

mFUNCTION(mMappedFile_Destroy, IN_OUT mPtr<mMappedFile> *pMappedFile);

#endif // mMappedFile_h__
