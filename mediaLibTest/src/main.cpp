#include "mTestLib.h"

#include "mFile.h"

int32_t main(int32_t argc, const char **pArgv)
{
  mPrintErrorCallback = nullptr;
  
  // Set working directory.
  {
    mString appDirectory;
    appDirectory.pAllocator = &mDefaultTempAllocator;

    if (mFAILED(mFile_GetCurrentApplicationFilePath(&appDirectory)) ||
      mFAILED(mFile_ExtractDirectoryFromPath(&appDirectory, appDirectory)) ||
      mFAILED(mFile_SetWorkingDirectory(appDirectory)))
    {
      mPRINT("Failed to setup working directory.");
      return -1;
    }
  }

  mPrintLogCallback = mPrintCallback;

  mTestLib_Initialize();

  const mResult result = mTestLib_RunAllTests(&argc, pArgv);

#if defined(mDEBUG_TESTS)
  if (mFAILED(result))
  {
    puts("\nSome tests failed.\nPress any key to quit.");
    getchar();
  }
#endif

  return mSUCCEEDED(result) ? 0 : 1;
}
