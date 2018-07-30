#include "default.h"
#include "mVideoFileInputHandler.h"

mFUNCTION(ProcessBufferCallback, IN uint8_t *pBuffer)
{
  mFUNCTION_SETUP();
  
  if(pBuffer)
    printf("Success!\n");
  
  mRETURN_SUCCESS();
}

int main(int, char **)
{
  mFUNCTION_SETUP();

  int k = 0;
  mPtr<int> kptr;
  mDEFER_DESTRUCTION(&kptr, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Create(&kptr, &k, mAT_ForeignRessource));

  uint32_t *pPtr = nullptr;
  mDEFER_DESTRUCTION(pPtr, mFree);
  mERROR_CHECK(mAlloc(&pPtr, 1));

  g_mResult_breakOnError = true;
  mPtr<mVideoFileInputHandler> videoInput;
  mDEFER_DESTRUCTION(&videoInput, mVideoFileInputHandler_Destroy);
  mERROR_CHECK(mVideoFileInputHandler_Create(&videoInput, L"C:\\Users\\cstiller\\Videos\\Original.MP4"));
  
  size_t sizeX = 0, sizeY = 0;
  mERROR_CHECK(mVideoFileInputHandler_GetSize(videoInput, &sizeX, &sizeY));

  mERROR_CHECK(mVideoFileInputHandler_SetCallback(videoInput, ProcessBufferCallback));
  mERROR_CHECK(mVideoFileInputHandler_Play(videoInput));

  mRETURN_SUCCESS();
}