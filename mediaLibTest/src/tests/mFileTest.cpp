#include "mTestLib.h"
#include "mFile.h"

mTEST(mFile, TestGetDirectoryContents)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mQueue<mFileInfo>> contents;
  mDEFER_CALL(&contents, mQueue_Destroy);

  mTEST_ASSERT_SUCCESS(mFile_GetDirectoryContents("../testData/", &contents, pAllocator));

  size_t count = 0;
  mTEST_ASSERT_SUCCESS(mQueue_GetCount(contents, &count));

  mTEST_ASSERT_TRUE(count > 0);

  mTEST_ALLOCATOR_ZERO_CHECK();
}
