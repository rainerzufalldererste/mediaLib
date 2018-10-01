#include "mTestLib.h"

int main(int argc, char **pArgv)
{
  const mResult result = mTestLib_RunAllTests(&argc, pArgv);

#ifdef mDEBUG_TESTS
  if (mFAILED(result))
  {
    puts("\nSome tests failed.\nPress any key to quit.");
    getchar();
  }
#endif

  return result;
}
