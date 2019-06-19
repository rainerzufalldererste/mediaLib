#ifndef mDebugSymbolInfo_h__
#define mDebugSymbolInfo_h__

#include "mediaLib.h"

mFUNCTION(mDebugSymbolInfo_GetNameOfSymbol, IN void *pFunctionPtr, OUT char *name, const size_t length);
mFUNCTION(mDebugSymbolInfo_GetStackTrace, OUT char *stackTrace, const size_t length);

#endif // mDebugSymbolInfo_h__
