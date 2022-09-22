#include "mDebugSymbolInfo.h"

#include "StackWalker.h"
#include "StackWalker.cpp"

#include <winternl.h>

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

  name[0] = '\0';

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

  stackTrace[length] = '\0';

  StackWalkerToString &stackWalker = mDebugSymbolInfo_GetStackWalker();

  stackWalker.output = stackTrace;
  stackWalker.length = length;
  
  if (FALSE == stackWalker.ShowCallstack())
    mStringCopy(stackTrace, length, "", 1);

  stackTrace[length - 1] = '\0';

  mRETURN_SUCCESS();
}

mFUNCTION(mDebugSymbolInfo_GetCallStack, OUT mDebugSymbolInfo_CallStack *pCallstack)
{
  mFUNCTION_SETUP();

  mERROR_IF(pCallstack == nullptr, mR_ArgumentNull);

  pCallstack->size = CaptureStackBackTrace(0, (DWORD)ARRAYSIZE(pCallstack->callstack), pCallstack->callstack, nullptr);

  mERROR_IF(pCallstack->size == 0, mR_InternalError);

  mRETURN_SUCCESS();
}

mFUNCTION(mDebugSymbolInfo_GetStackTraceFromCallStack, IN const mDebugSymbolInfo_CallStack *pCallstack, OUT char *stackTrace, const size_t length)
{
  mFUNCTION_SETUP();

  mERROR_IF(pCallstack == nullptr || stackTrace == nullptr, mR_ArgumentNull);
  mERROR_IF(length <= 1, mR_IndexOutOfBounds);

  stackTrace[0] = '\0';

  char symbol[512];
  const PEB *pProcessEnvironmentBlock = reinterpret_cast<PEB *>(__readgsqword(0x60));

  for (size_t idx = 0; idx < pCallstack->size; idx++)
  {
    const size_t value = reinterpret_cast<size_t>(pCallstack->callstack[idx]);

    const LIST_ENTRY *pAppMemoryModule = pProcessEnvironmentBlock->Ldr->InMemoryOrderModuleList.Flink;
    const LIST_ENTRY *pNext = pAppMemoryModule;

    const LDR_DATA_TABLE_ENTRY *pTableEntry = CONTAINING_RECORD(pNext, LDR_DATA_TABLE_ENTRY, InMemoryOrderLinks);
    const IMAGE_DOS_HEADER *pDosHeader = reinterpret_cast<const IMAGE_DOS_HEADER *>(pTableEntry->DllBase);

    if (pDosHeader == nullptr)
      break;

    const IMAGE_NT_HEADERS *pNtHeader = reinterpret_cast<const IMAGE_NT_HEADERS *>((size_t)pDosHeader + pDosHeader->e_lfanew);
    const size_t sectionHeaderCount = pNtHeader->FileHeader.NumberOfSections;
    const IMAGE_SECTION_HEADER *pSectionHeader = IMAGE_FIRST_SECTION(pNtHeader);

    bool found = false;

    for (size_t i = 0; i < sectionHeaderCount; i++)
    {
      if (pSectionHeader[i].Characteristics & (IMAGE_SCN_MEM_EXECUTE | IMAGE_SCN_CNT_CODE))
      {
        const size_t moduleBase = reinterpret_cast<size_t>(pDosHeader);

        if (value >= moduleBase + pSectionHeader[i].VirtualAddress && value < moduleBase + pSectionHeader[i].VirtualAddress + pSectionHeader[i].Misc.VirtualSize)
        {
          const size_t offset = value - moduleBase;

          if (pNext != pAppMemoryModule)
          {
            if (0 < WideCharToMultiByte(CP_UTF8, 0, reinterpret_cast<const wchar_t *>(pTableEntry->FullDllName.Buffer), pTableEntry->FullDllName.Length, symbol, (int32_t)mARRAYSIZE(symbol), nullptr, false) || GetLastError() == ERROR_INSUFFICIENT_BUFFER)
            {
              mERROR_CHECK(mStringConcat(stackTrace, length, symbol, mARRAYSIZE(symbol)));
              mERROR_CHECK(mStringConcat(stackTrace, length, " ", 2));
            }
            else
            {
              const char buffer[] = "<Invalid Module Name> ";
              mERROR_CHECK(mStringConcat(stackTrace, length, buffer, mARRAYSIZE(buffer)));
            }
          }

          char buffer[sizeof("0xFFFFFFFFFFFFFFFF ")];
          mERROR_CHECK(mFormatTo(buffer, mARRAYSIZE(buffer), "0x", mFX()(offset), ' '));
          mERROR_CHECK(mStringConcat(stackTrace, length, buffer, mARRAYSIZE(buffer)));
          
          if (mSUCCEEDED(mSILENCE_ERROR(mDebugSymbolInfo_GetNameOfSymbol(reinterpret_cast<void *>(value), symbol, mARRAYSIZE(symbol)))))
            mERROR_CHECK(mStringConcat(stackTrace, length, symbol, mARRAYSIZE(symbol)));

          mERROR_CHECK(mStringConcat(stackTrace, length, "\n", 2));

          found = true;
          break;
        }
      }
    }

    if (!found)
    {
      char buffer[64];
      mERROR_CHECK(mFormatTo(buffer, mARRAYSIZE(buffer), "<Unknown Module> @ 0x", mFX()(value), ' '));
      mERROR_CHECK(mStringConcat(stackTrace, length, buffer, mARRAYSIZE(buffer)));

      if (mSUCCEEDED(mSILENCE_ERROR(mDebugSymbolInfo_GetNameOfSymbol(reinterpret_cast<void *>(value), symbol, mARRAYSIZE(symbol)))))
        mERROR_CHECK(mStringConcat(stackTrace, length, symbol, mARRAYSIZE(symbol)));

      mERROR_CHECK(mStringConcat(stackTrace, length, "\n", 2));
    }
  }

  mRETURN_SUCCESS();
}
