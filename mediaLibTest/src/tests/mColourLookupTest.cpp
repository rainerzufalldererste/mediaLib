#include "mTestLib.h"
#include "mColourLookup.h"
#include "mFile.h"

const char mColourLookupTest_CubeFileName[] = "../testData/TestColourLookup.cube";

const mVec3f mColourLookupTest_FirstBlock[] = 
{
  { 0.0218357, 0.0218357, 0.0218357 },
  { 0.0310674, 0.0214237, 0.0214237 },
  { 0.0421302, 0.0213779, 0.0213779 },
  { 0.0535897, 0.0207523, 0.0207523 },
  { 0.0667735, 0.0199435, 0.0199435 },
  { 0.0821241, 0.0184787, 0.0184787 },
  { 0.10042, 0.016144, 0.016144 },
  { 0.122393, 0.0127108, 0.0127108 },
  { 0.149233, 0.00761425, 0.00761425 },
  { 0.184665, 0.000518807, 0.000518807 },
  { 0.231418, 0, 0 },
  { 0.294728, 0, 0 },
  { 0.375891, 0, 0 },
  { 0.48043, 0, 0 },
  { 0.608835, 0, 0 },
  { 0.675685, 0, 0 },
  { 0.643687, 0, 0 },
  { 0.757122, 0, 0 },
  { 0.871656, 0, 0 },
  { 0.984634, 0, 0 },
  { 1, 0, 0 },
  { 1, 0, 0 },
  { 1, 0, 0 },
  { 1, 0, 0 },
  { 1, 0, 0 },
  { 1, 0, 0 },
  { 1, 0, 0 },
  { 1, 0, 0 },
  { 1, 0, 0 },
  { 1, 0, 0 },
  { 1, 0, 0 },
  { 1, 0, 0 },
  { 1, 0, 0 }
};

const mVec3f mColourLookupTest_LastBlock[] =
{
  { 0, 1, 1 },
  { 0, 1, 1 },
  { 0, 1, 1 },
  { 0, 1, 1 },
  { 0, 1, 1 },
  { 0, 1, 1 },
  { 0, 1, 1 },
  { 0, 1, 1 },
  { 0, 1, 1 },
  { 0, 1, 1 },
  { 0, 1, 1 },
  { 0, 1, 1 },
  { 0, 1, 1 },
  { 0.0432593, 1, 1 },
  { 0.154147, 1, 1 },
  { 0.28014, 1, 1 },
  { 0.255238, 1, 1 },
  { 0.28513, 1, 1 },
  { 0.424109, 1, 1 },
  { 0.590494, 1, 1 },
  { 0.716777, 1, 1 },
  { 0.800275, 1, 1 },
  { 0.855604, 1, 1 },
  { 0.89337, 1, 1 },
  { 0.920409, 1, 1 },
  { 0.942275, 1, 1 },
  { 0.959487, 1, 1 },
  { 0.973251, 1, 1 },
  { 0.98436, 1, 1 },
  { 0.993851, 1, 1 },
  { 1, 1, 1 },
  { 1, 1, 1 },
  { 1, 1, 1 }
};

mTEST(mColourLookup, LoadFromFileTest)
{
  mTEST_ALLOCATOR_SETUP();

  mPtr<mColourLookup> colourLookup;
  mDEFER_CALL(&colourLookup, mColourLookup_Destroy);

  mTEST_ASSERT_SUCCESS(mColourLookup_CreateFromFile(&colourLookup, pAllocator, mColourLookupTest_CubeFileName));

  mTEST_ASSERT_EQUAL(33, colourLookup->resolution.x);
  mTEST_ASSERT_EQUAL(33, colourLookup->resolution.y);
  mTEST_ASSERT_EQUAL(33, colourLookup->resolution.z);

  for (size_t i = 0; i < mARRAYSIZE(mColourLookupTest_FirstBlock); i++)
    mTEST_ASSERT_EQUAL(colourLookup->pData[i], mColourLookupTest_FirstBlock[i]);

  for (size_t i = 0; i < mARRAYSIZE(mColourLookupTest_LastBlock); i++)
    mTEST_ASSERT_EQUAL(colourLookup->pData[33 * 33 * 33 - 33 + i], mColourLookupTest_LastBlock[i]);

  mTEST_ALLOCATOR_ZERO_CHECK();
}
