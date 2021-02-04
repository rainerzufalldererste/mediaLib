#ifndef mDebugSymbolInfo_h__
#define mDebugSymbolInfo_h__

#include "mediaLib.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "3GSFNqDmfSigp3NFRvK/tNXhXF+1dKUUQv1lMy/wCsFbpLocVHwyJuYuIYqpUgTMgTaPpxc+AiejsE+r"
#endif

mFUNCTION(mDebugSymbolInfo_GetNameOfSymbol, IN void *pFunctionPtr, OUT char *name, const size_t length);
mFUNCTION(mDebugSymbolInfo_GetStackTrace, OUT char *stackTrace, const size_t length);

#endif // mDebugSymbolInfo_h__
