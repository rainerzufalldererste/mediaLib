#include "mTestLib.h"

int main(int argc, char **pArgv)
{
  mPrintErrorCallback = nullptr;

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
