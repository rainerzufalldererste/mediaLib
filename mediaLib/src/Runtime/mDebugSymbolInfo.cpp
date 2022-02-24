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

  inline StackWalkerToString() : StackWalker(GetCurrentProcessId(), GetCurrentProcess()) { }

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

  virtual void OnOutput(LPCSTR szText) 
  {
    const size_t lengthRemaining = strnlen(output, length);

    if (lengthRemaining + 1 < length)
      strncat(output, szText, length - lengthRemaining - 1);
  }
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

  StackWalkerToString &stackWalker = mDebugSymbolInfo_GetStackWalker();

  stackWalker.output = name;
  stackWalker.length = length;
  
  if (FALSE == stackWalker.ShowObject(pFunctionPtr))
    mStringCopy(name, length, "", 1);

  name[length - 1] = '\0';

  mRETURN_SUCCESS();
}

mFUNCTION(mDebugSymbolInfo_GetStackTrace, OUT char *stackTrace, const size_t length)
{
  mFUNCTION_SETUP();

  mERROR_IF(stackTrace == nullptr, mR_ArgumentNull);

  mZeroMemory(stackTrace, length);

  StackWalkerToString &stackWalker = mDebugSymbolInfo_GetStackWalker();

  stackWalker.output = stackTrace;
  stackWalker.length = length;
  
  if (FALSE == stackWalker.ShowCallstack())
    mStringCopy(stackTrace, length, "", 1);

  stackTrace[length - 1] = '\0';

  mRETURN_SUCCESS();
}
