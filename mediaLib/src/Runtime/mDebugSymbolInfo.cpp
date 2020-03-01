#include "mDebugSymbolInfo.h"

#include "StackWalker.h"
#include "StackWalker.cpp"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "bR5l8RIj349lnWYK7CNtTS098y3Igupy0MBLdpaAM5oHRcGyuMKUSm+XwKPz/Rqgdm8kLuKUdk3u/oix"
#endif

class StackWalkerToString : public StackWalker
{
public:
  char *output = nullptr;
  size_t length = 0;

protected:
  virtual void OnSymInit(LPCSTR /*szSearchPath*/, DWORD /*symOptions*/, LPCSTR /*szUserName*/) {}
  virtual void OnLoadModule(LPCSTR    /*img*/,
    LPCSTR    /*mod*/,
    DWORD64   /*baseAddr*/,
    DWORD     /*size*/,
    DWORD     /*result*/,
    LPCSTR    /*symType*/,
    LPCSTR    /*pdbName*/,
    ULONGLONG /*fileVersion*/) {};
  virtual void OnDbgHelpErr(LPCSTR /*szFuncName*/, DWORD /*gle*/, DWORD64 /*addr*/) {}
  virtual void OnOutput(LPCSTR szText) { strncat(output, szText, length - strnlen(output, length) - 1); }
};

static StackWalkerToString &mDebugSymbolInfo_GetStackWalker()
{
  static thread_local StackWalkerToString sw;
  return sw;
};

mFUNCTION(mDebugSymbolInfo_GetNameOfSymbol, IN void *pFunctionPtr, OUT char *name, const size_t length)
{
  mFUNCTION_SETUP();

  mERROR_IF(pFunctionPtr == nullptr || name == nullptr, mR_ArgumentNull);

  mDebugSymbolInfo_GetStackWalker().output = name;
  mDebugSymbolInfo_GetStackWalker().length = length;
  mDebugSymbolInfo_GetStackWalker().ShowObject(pFunctionPtr);
  name[length - 1] = '\0';

  mRETURN_SUCCESS();
}

mFUNCTION(mDebugSymbolInfo_GetStackTrace, OUT char *stackTrace, const size_t length)
{
  mFUNCTION_SETUP();

  mERROR_IF(stackTrace == nullptr, mR_ArgumentNull);
  
  mDebugSymbolInfo_GetStackWalker().output = stackTrace;
  mDebugSymbolInfo_GetStackWalker().length = length;
  mDebugSymbolInfo_GetStackWalker().ShowCallstack();
  stackTrace[length - 1] = '\0';

  mRETURN_SUCCESS();
}
