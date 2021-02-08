#include "mTestLib.h"
#include "mediaLib.h"

mTEST(mIsInt, Test)
{
  mTEST_ALLOCATOR_SETUP();

  mTEST_ASSERT_FALSE(mIsInt(nullptr));
  mTEST_ASSERT_FALSE(mIsInt(nullptr, 100));
  mTEST_ASSERT_FALSE(mIsInt(""));
  mTEST_ASSERT_FALSE(mIsInt("", 1000));
  mTEST_ASSERT_FALSE(mIsInt("/"));
  mTEST_ASSERT_FALSE(mIsInt(":"));
  mTEST_ASSERT_FALSE(mIsInt("0:"));
  mTEST_ASSERT_FALSE(mIsInt("0/"));
  mTEST_ASSERT_FALSE(mIsInt("01234567890/"));
  mTEST_ASSERT_FALSE(mIsInt("/01234567890"));
  mTEST_ASSERT_FALSE(mIsInt("1a2"));
  mTEST_ASSERT_FALSE(mIsInt("1a2", 2));
  mTEST_ASSERT_FALSE(mIsInt("-1a2", 3));
  mTEST_ASSERT_FALSE(mIsInt("--"));
  mTEST_ASSERT_FALSE(mIsInt("-"));
  mTEST_ASSERT_FALSE(mIsInt("1-23456"));
  mTEST_ASSERT_FALSE(mIsInt("-1", 1));
  mTEST_ASSERT_FALSE(mIsInt("00000000000000000a"));
  mTEST_ASSERT_FALSE(mIsInt("00000000000000000aa"));
  mTEST_ASSERT_FALSE(mIsInt("-00000000000000000a"));

  mTEST_ASSERT_TRUE(mIsInt("0123456789"));
  mTEST_ASSERT_TRUE(mIsInt("-0123456789"));
  mTEST_ASSERT_TRUE(mIsInt("0123456789", 1));
  mTEST_ASSERT_TRUE(mIsInt("0123456789", 1));
  mTEST_ASSERT_TRUE(mIsInt("0123456789", 100));
  mTEST_ASSERT_TRUE(mIsInt("1a2", 1));
  mTEST_ASSERT_TRUE(mIsInt("-1a2", 2));

  mTEST_ALLOCATOR_ZERO_CHECK();
}

mTEST(mIsFloat, Test)
{
  mTEST_ALLOCATOR_SETUP();

  mTEST_ASSERT_FALSE(mIsFloat("2-"));
  mTEST_ASSERT_FALSE(mIsFloat("2e-"));
  mTEST_ASSERT_FALSE(mIsFloat("-2e-"));
  mTEST_ASSERT_FALSE(mIsFloat("-e-"));
  mTEST_ASSERT_FALSE(mIsFloat(".e-"));
  mTEST_ASSERT_FALSE(mIsFloat(".e2"));
  mTEST_ASSERT_FALSE(mIsFloat(".e-2"));
  mTEST_ASSERT_FALSE(mIsFloat("-"));
  mTEST_ASSERT_FALSE(mIsFloat(""));
  mTEST_ASSERT_FALSE(mIsFloat("e2"));
  mTEST_ASSERT_FALSE(mIsFloat("e-"));
  mTEST_ASSERT_FALSE(mIsFloat("e-2"));
  mTEST_ASSERT_FALSE(mIsFloat(".e-2"));
  mTEST_ASSERT_FALSE(mIsFloat("-.e-2"));
  mTEST_ASSERT_FALSE(mIsFloat("-e-2"));
  mTEST_ASSERT_FALSE(mIsFloat("-e.2"));
  mTEST_ASSERT_FALSE(mIsFloat("-e2.0"));
  mTEST_ASSERT_FALSE(mIsFloat("-e2.0"));
  mTEST_ASSERT_FALSE(mIsFloat("-.2.e2"));
  mTEST_ASSERT_FALSE(mIsFloat("-2.2.e2"));
  mTEST_ASSERT_FALSE(mIsFloat("-.2.2e2"));
  mTEST_ASSERT_FALSE(mIsFloat("."));
  mTEST_ASSERT_FALSE(mIsFloat("-.2e-2."));

  mTEST_ASSERT_FALSE(mIsFloat("123-"));
  mTEST_ASSERT_FALSE(mIsFloat("123e-"));
  mTEST_ASSERT_FALSE(mIsFloat("-123e-"));
  mTEST_ASSERT_FALSE(mIsFloat("-e-"));
  mTEST_ASSERT_FALSE(mIsFloat(".e-"));
  mTEST_ASSERT_FALSE(mIsFloat(".e123"));
  mTEST_ASSERT_FALSE(mIsFloat(".e-123"));
  mTEST_ASSERT_FALSE(mIsFloat("-"));
  mTEST_ASSERT_FALSE(mIsFloat(""));
  mTEST_ASSERT_FALSE(mIsFloat("e123"));
  mTEST_ASSERT_FALSE(mIsFloat("e-"));
  mTEST_ASSERT_FALSE(mIsFloat("e-123"));
  mTEST_ASSERT_FALSE(mIsFloat(".e-123"));
  mTEST_ASSERT_FALSE(mIsFloat("-.e-123"));
  mTEST_ASSERT_FALSE(mIsFloat("-e-123"));
  mTEST_ASSERT_FALSE(mIsFloat("-e.123"));
  mTEST_ASSERT_FALSE(mIsFloat("-e123.0"));
  mTEST_ASSERT_FALSE(mIsFloat("-e123.0"));
  mTEST_ASSERT_FALSE(mIsFloat("-.123.e123"));
  mTEST_ASSERT_FALSE(mIsFloat("-123.123.e123"));
  mTEST_ASSERT_FALSE(mIsFloat("-.123.123e123"));
  mTEST_ASSERT_FALSE(mIsFloat("."));
  mTEST_ASSERT_FALSE(mIsFloat("-.123e-123."));

  mTEST_ASSERT_TRUE(mIsFloat("2"));
  mTEST_ASSERT_TRUE(mIsFloat(".2"));
  mTEST_ASSERT_TRUE(mIsFloat("-.2"));
  mTEST_ASSERT_TRUE(mIsFloat("2E2"));
  mTEST_ASSERT_TRUE(mIsFloat("2e2"));
  mTEST_ASSERT_TRUE(mIsFloat(".2E2"));
  mTEST_ASSERT_TRUE(mIsFloat("-2E2"));
  mTEST_ASSERT_TRUE(mIsFloat("-2E-2"));
  mTEST_ASSERT_TRUE(mIsFloat(".2e2"));
  mTEST_ASSERT_TRUE(mIsFloat("-2.2e2"));
  mTEST_ASSERT_TRUE(mIsFloat("-.2e2"));
  mTEST_ASSERT_TRUE(mIsFloat(".2e-2"));
  mTEST_ASSERT_TRUE(mIsFloat("-.2E2"));
  mTEST_ASSERT_TRUE(mIsFloat("-.2e-2"));
  mTEST_ASSERT_TRUE(mIsFloat("-.2E-2"));
  mTEST_ASSERT_TRUE(mIsFloat("-2.E-2"));
  mTEST_ASSERT_TRUE(mIsFloat("2.e-2"));

  mTEST_ASSERT_TRUE(mIsFloat("123"));
  mTEST_ASSERT_TRUE(mIsFloat(".123"));
  mTEST_ASSERT_TRUE(mIsFloat("-.123"));
  mTEST_ASSERT_TRUE(mIsFloat("123E123"));
  mTEST_ASSERT_TRUE(mIsFloat("123e123"));
  mTEST_ASSERT_TRUE(mIsFloat(".123E123"));
  mTEST_ASSERT_TRUE(mIsFloat("-123E123"));
  mTEST_ASSERT_TRUE(mIsFloat("-123E-123"));
  mTEST_ASSERT_TRUE(mIsFloat("-123.123e123"));
  mTEST_ASSERT_TRUE(mIsFloat(".123e123"));
  mTEST_ASSERT_TRUE(mIsFloat(".123e-123"));
  mTEST_ASSERT_TRUE(mIsFloat("-.123E123"));
  mTEST_ASSERT_TRUE(mIsFloat("-.123e-123"));
  mTEST_ASSERT_TRUE(mIsFloat("-.123E-123"));
  mTEST_ASSERT_TRUE(mIsFloat("-123.E-123"));
  mTEST_ASSERT_TRUE(mIsFloat("123.e-123"));
  
  mTEST_ALLOCATOR_ZERO_CHECK();
}
