#include "gtest/gtest.h"

//#define DEBUG_TESTS

int main(int argc, char **pArgv)
{
  ::testing::InitGoogleTest(&argc, pArgv);

  int result = RUN_ALL_TESTS();

#ifdef DEBUG_TESTS
  if (result != 0)
  {
    puts("\nSome tests failed.\nPress any key to quit.");
    getchar();
  }
#endif

  return result;
}
