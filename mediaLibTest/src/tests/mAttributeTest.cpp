#include "mTestLib.h"
#include "mAttribute.h"

mTEST(mAttribute, TestSingleAttributeCollection)
{
  mTEST_ALLOCATOR_SETUP();

  mAttributeStore store;
  mTEST_ASSERT_SUCCESS(mAttributeStore_Create(&store, pAllocator));

  mAttributeCollection object;
  mTEST_ASSERT_SUCCESS(mAttributeCollection_Create(&object, &store));

  mAttributeIterator iterator;
  mTEST_ASSERT_SUCCESS(mAttributeCollection_CreateIterator(&object, &iterator));

  mTEST_ASSERT_SUCCESS(mAttributeIterator_AddUInt64(&iterator, "a", 1));
  mTEST_ASSERT_EQUAL(mR_ResourceAlreadyExists, mAttributeIterator_AddInt64(&iterator, "a", -1));

  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mAttributeIterator_AddUInt64(&iterator, "", 1));
  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mAttributeIterator_AddInt64(&iterator, "", -1));
  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mAttributeIterator_AddBool(&iterator, "", true));
  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mAttributeIterator_AddFloat64(&iterator, "", 1.234));
  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mAttributeIterator_AddVector2(&iterator, "", mVec2f(3)));
  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mAttributeIterator_AddVector3(&iterator, "", mVec3f(4)));
  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mAttributeIterator_AddVector4(&iterator, "", mVec4f(5)));
  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mAttributeIterator_AddString(&iterator, "", "test"));
  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mAttributeIterator_AddObject(&iterator, ""));

  mTEST_ASSERT_SUCCESS(mAttributeIterator_AddInt64(&iterator, "b", -1));
  mTEST_ASSERT_SUCCESS(mAttributeIterator_AddBool(&iterator, "c", true));
  mTEST_ASSERT_SUCCESS(mAttributeIterator_AddFloat64(&iterator, "d", 1.234));
  mTEST_ASSERT_SUCCESS(mAttributeIterator_AddVector2(&iterator, "e", mVec2f(3)));
  mTEST_ASSERT_SUCCESS(mAttributeIterator_AddVector3(&iterator, "f", mVec3f(4)));
  mTEST_ASSERT_SUCCESS(mAttributeIterator_AddVector4(&iterator, "g", mVec4f(5)));
  mTEST_ASSERT_SUCCESS(mAttributeIterator_AddString(&iterator, "h", "test"));
  mTEST_ASSERT_SUCCESS(mAttributeIterator_AddObject(&iterator, "i"));

  {
    mTEST_ASSERT_SUCCESS(mAttributeIterator_AddUInt64(&iterator, "a", 1));
    mTEST_ASSERT_EQUAL(mR_ResourceAlreadyExists, mAttributeIterator_AddInt64(&iterator, "a", -1));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_AddInt64(&iterator, "b", -1));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_AddBool(&iterator, "c", true));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_AddFloat64(&iterator, "d", 1.234));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_AddVector2(&iterator, "e", mVec2f(3)));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_AddVector3(&iterator, "f", mVec3f(4)));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_AddVector4(&iterator, "g", mVec4f(5)));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_AddString(&iterator, "h", "test"));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_AddObject(&iterator, "j"));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_LeaveObject(&iterator));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_AddObject(&iterator, "k"));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_AddInt64(&iterator, "a", -1234));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_AddInt64(&iterator, "b", 1234));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_LeaveObject(&iterator));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_Find(&iterator, "j"));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_EnterCurrentObject(&iterator));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_AddInt64(&iterator, "a", -1234));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_AddInt64(&iterator, "b", 1234));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_LeaveObject(&iterator));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_Find(&iterator, "k"));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_EnterCurrentObject(&iterator));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_AddFloat64(&iterator, "c", 0.1234));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_AddFloat64(&iterator, "d", 1.234));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_LeaveObject(&iterator));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_Find(&iterator, "j"));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_EnterCurrentObject(&iterator));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_AddObject(&iterator, "i"));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_AddInt64(&iterator, "a", -1234));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_AddInt64(&iterator, "b", 1234));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_LeaveObject(&iterator));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_LeaveObject(&iterator));
  }

  mTEST_ASSERT_SUCCESS(mAttributeIterator_LeaveObject(&iterator));
  mTEST_ASSERT_EQUAL(mR_ResourceNotFound, mAttributeIterator_Find(&iterator, "x"));
  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mAttributeIterator_AddUInt64(&iterator, "🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅", 32));
  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mAttributeIterator_AddUInt64(&iterator, "0123456789012345678901234567890123456789012345678901234567890123", 32));
  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mAttributeIterator_AddUInt64(&iterator, "01234567890123456789012345678901234567890123456789012345678901234", 32));
  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mAttributeIterator_AddInt64(&iterator, "🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅", 32));
  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mAttributeIterator_AddInt64(&iterator, "0123456789012345678901234567890123456789012345678901234567890123", 32));
  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mAttributeIterator_AddInt64(&iterator, "01234567890123456789012345678901234567890123456789012345678901234", 32));
  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mAttributeIterator_AddBool(&iterator, "🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅", false));
  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mAttributeIterator_AddBool(&iterator, "0123456789012345678901234567890123456789012345678901234567890123", false));
  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mAttributeIterator_AddBool(&iterator, "01234567890123456789012345678901234567890123456789012345678901234", false));
  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mAttributeIterator_AddFloat64(&iterator, "🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅", 1.23));
  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mAttributeIterator_AddFloat64(&iterator, "0123456789012345678901234567890123456789012345678901234567890123", 1.23));
  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mAttributeIterator_AddFloat64(&iterator, "01234567890123456789012345678901234567890123456789012345678901234", 1.23));
  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mAttributeIterator_AddVector2(&iterator, "🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅", mVec2f(1.23f)));
  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mAttributeIterator_AddVector2(&iterator, "0123456789012345678901234567890123456789012345678901234567890123", mVec2f(1.23f)));
  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mAttributeIterator_AddVector2(&iterator, "01234567890123456789012345678901234567890123456789012345678901234", mVec2f(1.23f)));
  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mAttributeIterator_AddVector3(&iterator, "🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅", mVec3f(1.23f)));
  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mAttributeIterator_AddVector3(&iterator, "0123456789012345678901234567890123456789012345678901234567890123", mVec3f(1.23f)));
  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mAttributeIterator_AddVector3(&iterator, "01234567890123456789012345678901234567890123456789012345678901234", mVec3f(1.23f)));
  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mAttributeIterator_AddVector4(&iterator, "🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅", mVec4f(1.23f)));
  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mAttributeIterator_AddVector4(&iterator, "0123456789012345678901234567890123456789012345678901234567890123", mVec4f(1.23f)));
  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mAttributeIterator_AddVector4(&iterator, "01234567890123456789012345678901234567890123456789012345678901234", mVec4f(1.23f)));
  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mAttributeIterator_AddObject(&iterator, "🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅🌵🦎🌵🎅"));
  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mAttributeIterator_AddObject(&iterator, "0123456789012345678901234567890123456789012345678901234567890123"));
  mTEST_ASSERT_EQUAL(mR_InvalidParameter, mAttributeIterator_AddObject(&iterator, "01234567890123456789012345678901234567890123456789012345678901234"));

  mString s;
  s.pAllocator = pAllocator;

  mInplaceString<mAttribute_NameLength> is;

  mAttribute_Type type;

  uint64_t u64;
  int64_t i64;
  bool b;
  double_t f64;
  float_t f32;
  mVec2f v2;
  mVec3f v3;
  mVec4f v4;

  // i64.
  {
    mTEST_ASSERT_SUCCESS(mAttributeIterator_Find(&iterator, "b"));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeName(&iterator, &s));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeName(&iterator, &is));
    mTEST_ASSERT_EQUAL(s, "b");
    mTEST_ASSERT_EQUAL(is, "b");

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeType(&iterator, &type));
    mTEST_ASSERT_EQUAL(type, mA_T_Int64);

    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &u64));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentValue(&iterator, &i64));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &b));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &f64));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &f32));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v2));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v3));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v4));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &s));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentObject(&iterator, &iterator));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_EnterCurrentObject(&iterator));

    mTEST_ASSERT_EQUAL(i64, -1);
  }

  // u64.
  {
    mTEST_ASSERT_SUCCESS(mAttributeIterator_Find(&iterator, "a"));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeName(&iterator, &s));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeName(&iterator, &is));
    mTEST_ASSERT_EQUAL(s, "a");
    mTEST_ASSERT_EQUAL(is, "a");

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeType(&iterator, &type));
    mTEST_ASSERT_EQUAL(type, mA_T_UInt64);

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentValue(&iterator, &u64));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &i64));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &b));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &f64));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &f32));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v2));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v3));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v4));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &s));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentObject(&iterator, &iterator));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_EnterCurrentObject(&iterator));

    mTEST_ASSERT_EQUAL(u64, 1);
  }

  // float64.
  {
    mTEST_ASSERT_SUCCESS(mAttributeIterator_Find(&iterator, "d"));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeName(&iterator, &s));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeName(&iterator, &is));
    mTEST_ASSERT_EQUAL(s, "d");
    mTEST_ASSERT_EQUAL(is, "d");

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeType(&iterator, &type));
    mTEST_ASSERT_EQUAL(type, mA_T_Float64);

    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &u64));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &i64));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &b));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentValue(&iterator, &f64));
    mTEST_ASSERT_EQUAL(f64, 1.234);
    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentValue(&iterator, &f32));
    mTEST_ASSERT_EQUAL(f32, 1.234f);
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v2));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v3));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v4));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &s));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentObject(&iterator, &iterator));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_EnterCurrentObject(&iterator));
  }

  // bool.
  {
    mTEST_ASSERT_SUCCESS(mAttributeIterator_Find(&iterator, "c"));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeName(&iterator, &s));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeName(&iterator, &is));
    mTEST_ASSERT_EQUAL(s, "c");
    mTEST_ASSERT_EQUAL(is, "c");

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeType(&iterator, &type));
    mTEST_ASSERT_EQUAL(type, mA_T_Bool);

    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &u64));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &i64));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentValue(&iterator, &b));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &f64));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &f32));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v2));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v3));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v4));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &s));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentObject(&iterator, &iterator));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_EnterCurrentObject(&iterator));

    mTEST_ASSERT_EQUAL(b, true);
  }

  // Vector2.
  {
    mTEST_ASSERT_SUCCESS(mAttributeIterator_Find(&iterator, "e"));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeName(&iterator, &s));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeName(&iterator, &is));
    mTEST_ASSERT_EQUAL(s, "e");
    mTEST_ASSERT_EQUAL(is, "e");

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeType(&iterator, &type));
    mTEST_ASSERT_EQUAL(type, mA_T_Vector2);

    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &u64));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &i64));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &b));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &f64));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &f32));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentValue(&iterator, &v2));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v3));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v4));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &s));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentObject(&iterator, &iterator));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_EnterCurrentObject(&iterator));

    mTEST_ASSERT_EQUAL(v2, mVec2f(3));
  }

  // Vector3.
  {
    mTEST_ASSERT_SUCCESS(mAttributeIterator_Find(&iterator, "f"));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeName(&iterator, &s));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeName(&iterator, &is));
    mTEST_ASSERT_EQUAL(s, "f");
    mTEST_ASSERT_EQUAL(is, "f");

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeType(&iterator, &type));
    mTEST_ASSERT_EQUAL(type, mA_T_Vector3);

    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &u64));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &i64));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &b));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &f64));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &f32));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v2));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentValue(&iterator, &v3));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v4));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &s));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentObject(&iterator, &iterator));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_EnterCurrentObject(&iterator));

    mTEST_ASSERT_EQUAL(v3, mVec3f(4));
  }

  // Vector4.
  {
    mTEST_ASSERT_SUCCESS(mAttributeIterator_Find(&iterator, "g"));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeName(&iterator, &s));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeName(&iterator, &is));
    mTEST_ASSERT_EQUAL(s, "g");
    mTEST_ASSERT_EQUAL(is, "g");

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeType(&iterator, &type));
    mTEST_ASSERT_EQUAL(type, mA_T_Vector4);

    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &u64));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &i64));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &b));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &f64));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &f32));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v2));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v3));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentValue(&iterator, &v4));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &s));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentObject(&iterator, &iterator));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_EnterCurrentObject(&iterator));

    mTEST_ASSERT_EQUAL(v4, mVec4f(5));
  }

  // String.
  {
    mTEST_ASSERT_SUCCESS(mAttributeIterator_Find(&iterator, "h"));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeName(&iterator, &s));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeName(&iterator, &is));
    mTEST_ASSERT_EQUAL(s, "h");
    mTEST_ASSERT_EQUAL(is, "h");

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeType(&iterator, &type));
    mTEST_ASSERT_EQUAL(type, mA_T_String);

    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &u64));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &i64));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &b));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &f64));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &f32));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v2));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v3));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v4));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentValue(&iterator, &s));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentObject(&iterator, &iterator));

    mTEST_ASSERT_EQUAL(s, "test");
  }

  // Object.
  {
    mTEST_ASSERT_SUCCESS(mAttributeIterator_Find(&iterator, "i"));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeName(&iterator, &s));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeName(&iterator, &is));
    mTEST_ASSERT_EQUAL(s, "i");
    mTEST_ASSERT_EQUAL(is, "i");

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeType(&iterator, &type));
    mTEST_ASSERT_EQUAL(type, mA_T_Object);

    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &u64));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &i64));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &b));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &f64));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &f32));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v2));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v3));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v4));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &s));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_EnterCurrentObject(&iterator));

    // i64.
    {
      mTEST_ASSERT_SUCCESS(mAttributeIterator_Find(&iterator, "b"));

      mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeName(&iterator, &s));
      mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeName(&iterator, &is));
      mTEST_ASSERT_EQUAL(s, "b");
      mTEST_ASSERT_EQUAL(is, "b");

      mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeType(&iterator, &type));
      mTEST_ASSERT_EQUAL(type, mA_T_Int64);

      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &u64));
      mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentValue(&iterator, &i64));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &b));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &f64));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &f32));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v2));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v3));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v4));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &s));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentObject(&iterator, &iterator));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_EnterCurrentObject(&iterator));

      mTEST_ASSERT_EQUAL(i64, -1);
    }

    // u64.
    {
      mTEST_ASSERT_SUCCESS(mAttributeIterator_Find(&iterator, "a"));

      mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeName(&iterator, &s));
      mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeName(&iterator, &is));
      mTEST_ASSERT_EQUAL(s, "a");
      mTEST_ASSERT_EQUAL(is, "a");

      mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeType(&iterator, &type));
      mTEST_ASSERT_EQUAL(type, mA_T_UInt64);

      mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentValue(&iterator, &u64));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &i64));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &b));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &f64));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &f32));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v2));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v3));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v4));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &s));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentObject(&iterator, &iterator));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_EnterCurrentObject(&iterator));

      mTEST_ASSERT_EQUAL(u64, 1);
    }

    // float64.
    {
      mTEST_ASSERT_SUCCESS(mAttributeIterator_Find(&iterator, "d"));

      mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeName(&iterator, &s));
      mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeName(&iterator, &is));
      mTEST_ASSERT_EQUAL(s, "d");
      mTEST_ASSERT_EQUAL(is, "d");

      mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeType(&iterator, &type));
      mTEST_ASSERT_EQUAL(type, mA_T_Float64);

      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &u64));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &i64));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &b));
      mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentValue(&iterator, &f64));
      mTEST_ASSERT_EQUAL(f64, 1.234);
      mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentValue(&iterator, &f32));
      mTEST_ASSERT_EQUAL(f32, 1.234f);
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v2));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v3));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v4));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &s));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentObject(&iterator, &iterator));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_EnterCurrentObject(&iterator));
    }

    // bool.
    {
      mTEST_ASSERT_SUCCESS(mAttributeIterator_Find(&iterator, "c"));

      mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeName(&iterator, &s));
      mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeName(&iterator, &is));
      mTEST_ASSERT_EQUAL(s, "c");
      mTEST_ASSERT_EQUAL(is, "c");

      mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeType(&iterator, &type));
      mTEST_ASSERT_EQUAL(type, mA_T_Bool);

      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &u64));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &i64));
      mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentValue(&iterator, &b));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &f64));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &f32));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v2));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v3));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v4));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &s));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentObject(&iterator, &iterator));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_EnterCurrentObject(&iterator));

      mTEST_ASSERT_EQUAL(b, true);
    }

    // Vector2.
    {
      mTEST_ASSERT_SUCCESS(mAttributeIterator_Find(&iterator, "e"));

      mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeName(&iterator, &s));
      mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeName(&iterator, &is));
      mTEST_ASSERT_EQUAL(s, "e");
      mTEST_ASSERT_EQUAL(is, "e");

      mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeType(&iterator, &type));
      mTEST_ASSERT_EQUAL(type, mA_T_Vector2);

      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &u64));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &i64));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &b));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &f64));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &f32));
      mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentValue(&iterator, &v2));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v3));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v4));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &s));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentObject(&iterator, &iterator));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_EnterCurrentObject(&iterator));

      mTEST_ASSERT_EQUAL(v2, mVec2f(3));
    }

    // Vector3.
    {
      mTEST_ASSERT_SUCCESS(mAttributeIterator_Find(&iterator, "f"));

      mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeName(&iterator, &s));
      mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeName(&iterator, &is));
      mTEST_ASSERT_EQUAL(s, "f");
      mTEST_ASSERT_EQUAL(is, "f");

      mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeType(&iterator, &type));
      mTEST_ASSERT_EQUAL(type, mA_T_Vector3);

      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &u64));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &i64));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &b));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &f64));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &f32));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v2));
      mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentValue(&iterator, &v3));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v4));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &s));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentObject(&iterator, &iterator));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_EnterCurrentObject(&iterator));

      mTEST_ASSERT_EQUAL(v3, mVec3f(4));
    }

    // Vector4.
    {
      mTEST_ASSERT_SUCCESS(mAttributeIterator_Find(&iterator, "g"));

      mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeName(&iterator, &s));
      mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeName(&iterator, &is));
      mTEST_ASSERT_EQUAL(s, "g");
      mTEST_ASSERT_EQUAL(is, "g");

      mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeType(&iterator, &type));
      mTEST_ASSERT_EQUAL(type, mA_T_Vector4);

      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &u64));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &i64));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &b));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &f64));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &f32));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v2));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v3));
      mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentValue(&iterator, &v4));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &s));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentObject(&iterator, &iterator));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_EnterCurrentObject(&iterator));

      mTEST_ASSERT_EQUAL(v4, mVec4f(5));
    }

    // String.
    {
      mTEST_ASSERT_SUCCESS(mAttributeIterator_Find(&iterator, "h"));

      mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeName(&iterator, &s));
      mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeName(&iterator, &is));
      mTEST_ASSERT_EQUAL(s, "h");
      mTEST_ASSERT_EQUAL(is, "h");

      mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeType(&iterator, &type));
      mTEST_ASSERT_EQUAL(type, mA_T_String);

      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &u64));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &i64));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &b));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &f64));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &f32));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v2));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v3));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentValue(&iterator, &v4));
      mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentValue(&iterator, &s));
      mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_GetCurrentObject(&iterator, &iterator));

      mTEST_ASSERT_EQUAL(s, "test");
    }

    mTEST_ASSERT_SUCCESS(mAttributeIterator_Find(&iterator, "j"));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_ReplaceCurrentValueWithUInt64(&iterator, 1));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_ReplaceCurrentValueWithInt64(&iterator, -1));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_ReplaceCurrentValueWithBool(&iterator, true));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_ReplaceCurrentValueWithFloat64(&iterator, 1.234));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_ReplaceCurrentValueWithVector2(&iterator, mVec2f(3)));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_ReplaceCurrentValueWithVector3(&iterator, mVec3f(4)));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_ReplaceCurrentValueWithVector4(&iterator, mVec4f(5)));
    mTEST_ASSERT_EQUAL(mR_ResourceIncompatible, mAttributeIterator_ReplaceCurrentValueWithString(&iterator, "test"));

    mAttributeValueHandle handleA, handleB;
    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentValueGlobalHandle(&iterator, &handleA));

    mTEST_ASSERT_EQUAL(mR_ResourceNotFound, mAttributeIterator_ReactivateValueFromGlobalHandle(&iterator, (mAttributeValueHandle)-1));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_ReactivateValueFromGlobalHandle(&iterator, handleA));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_DeactivateCurrentValue(&iterator));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeName(&iterator, &is));
    mTEST_ASSERT_NOT_EQUAL(is, "j");

    mTEST_ASSERT_EQUAL(mR_ResourceNotFound, mAttributeIterator_Find(&iterator, "z"));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_AddUInt64(&iterator, "y", 2468));
    mTEST_ASSERT_EQUAL(mR_ResourceNotFound, mAttributeIterator_Find(&iterator, "j"));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_ReactivateValueFromGlobalHandle(&iterator, handleA));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_Find(&iterator, "j"));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_DeactivateCurrentValue(&iterator));
    mTEST_ASSERT_EQUAL(mR_ResourceNotFound, mAttributeIterator_Find(&iterator, "j"));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_AddUInt64(&iterator, "j", 12345));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_Find(&iterator, "j"));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_Find(&iterator, "a"));
    mTEST_ASSERT_EQUAL(mR_ResourceAlreadyExists, mAttributeIterator_ReactivateValueFromGlobalHandle(&iterator, handleA));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_Find(&iterator, "j"));
    mTEST_ASSERT_EQUAL(mR_ResourceAlreadyExists, mAttributeIterator_ReactivateValueFromGlobalHandle(&iterator, handleA));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeType(&iterator, &type));
    mTEST_ASSERT_EQUAL(type, mA_T_UInt64);

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentValueGlobalHandle(&iterator, &handleB));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_DeactivateCurrentValue(&iterator));
    mTEST_ASSERT_EQUAL(mR_ResourceNotFound, mAttributeIterator_Find(&iterator, "j"));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_ReactivateValueFromGlobalHandle(&iterator, handleA));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_Find(&iterator, "j"));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeType(&iterator, &type));
    mTEST_ASSERT_EQUAL(type, mA_T_Object);
    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeType(&iterator, &type));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeName(&iterator, &s));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeName(&iterator, &is));
    mTEST_ASSERT_EQUAL(s, "j");
    mTEST_ASSERT_EQUAL(is, "j");

    mTEST_ASSERT_EQUAL(mR_ResourceAlreadyExists, mAttributeIterator_ReactivateValueFromGlobalHandle(&iterator, handleB));

    // What it it's the last (and only) element?
    {
      mTEST_ASSERT_SUCCESS(mAttributeIterator_AddObject(&iterator, "q"));

      mTEST_ASSERT_SUCCESS(mAttributeIterator_AddUInt64(&iterator, "j", 12345));
      mTEST_ASSERT_SUCCESS(mAttributeIterator_Find(&iterator, "j"));
      mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentValueGlobalHandle(&iterator, &handleA));
      mTEST_ASSERT_SUCCESS(mAttributeIterator_DeactivateCurrentValue(&iterator));
      mTEST_ASSERT_EQUAL(mR_ResourceNotFound, mAttributeIterator_Find(&iterator, "j"));
      mTEST_ASSERT_SUCCESS(mAttributeIterator_ReactivateValueFromGlobalHandle(&iterator, handleA));
      mTEST_ASSERT_SUCCESS(mAttributeIterator_Find(&iterator, "j"));

      mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeType(&iterator, &type));
      mTEST_ASSERT_EQUAL(type, mA_T_UInt64);

      mTEST_ASSERT_SUCCESS(mAttributeIterator_LeaveObject(&iterator));
    }

    mTEST_ASSERT_SUCCESS(mAttributeIterator_Find(&iterator, "a"));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_ReplaceCurrentValueWithUInt64(&iterator, 2));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeName(&iterator, &s));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeName(&iterator, &is));
    mTEST_ASSERT_EQUAL(s, "a");
    mTEST_ASSERT_EQUAL(is, "a");

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeType(&iterator, &type));
    mTEST_ASSERT_EQUAL(type, mA_T_UInt64);

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentValue(&iterator, &u64));
    mTEST_ASSERT_EQUAL(u64, 2);

    mTEST_ASSERT_SUCCESS(mAttributeIterator_ReplaceCurrentValueWithInt64(&iterator, -2));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeType(&iterator, &type));
    mTEST_ASSERT_EQUAL(type, mA_T_Int64);

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentValue(&iterator, &i64));
    mTEST_ASSERT_EQUAL(i64, -2);

    mTEST_ASSERT_SUCCESS(mAttributeIterator_ReplaceCurrentValueWithBool(&iterator, false));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeType(&iterator, &type));
    mTEST_ASSERT_EQUAL(type, mA_T_Bool);

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentValue(&iterator, &b));
    mTEST_ASSERT_EQUAL(b, false);

    mTEST_ASSERT_SUCCESS(mAttributeIterator_ReplaceCurrentValueWithFloat64(&iterator, 2.345));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeType(&iterator, &type));
    mTEST_ASSERT_EQUAL(type, mA_T_Float64);

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentValue(&iterator, &f64));
    mTEST_ASSERT_EQUAL(f64, 2.345);

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentValue(&iterator, &f32));
    mTEST_ASSERT_EQUAL(f32, 2.345f);

    mTEST_ASSERT_SUCCESS(mAttributeIterator_ReplaceCurrentValueWithVector2(&iterator, mVec2f(-3)));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeType(&iterator, &type));
    mTEST_ASSERT_EQUAL(type, mA_T_Vector2);

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentValue(&iterator, &v2));
    mTEST_ASSERT_EQUAL(v2, mVec2f(-3));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_ReplaceCurrentValueWithVector3(&iterator, mVec3f(-4)));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeType(&iterator, &type));
    mTEST_ASSERT_EQUAL(type, mA_T_Vector3);

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentValue(&iterator, &v3));
    mTEST_ASSERT_EQUAL(v3, mVec3f(-4));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_ReplaceCurrentValueWithVector4(&iterator, mVec4f(-5)));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeType(&iterator, &type));
    mTEST_ASSERT_EQUAL(type, mA_T_Vector4);

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentValue(&iterator, &v4));
    mTEST_ASSERT_EQUAL(v4, mVec4f(-5));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_ReplaceCurrentValueWithString(&iterator, "not test"));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeType(&iterator, &type));
    mTEST_ASSERT_EQUAL(type, mA_T_String);

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentValue(&iterator, &s));
    mTEST_ASSERT_EQUAL(s, "not test");

    mTEST_ASSERT_SUCCESS(mAttributeIterator_ReplaceCurrentValueWithString(&iterator, "still not test"));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeType(&iterator, &type));
    mTEST_ASSERT_EQUAL(type, mA_T_String);

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentValue(&iterator, &s));
    mTEST_ASSERT_EQUAL(s, "still not test");

    mTEST_ASSERT_SUCCESS(mAttributeIterator_ReplaceCurrentValueWithString(&iterator, "still not test"));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_Find(&iterator, "a"));
    mTEST_ASSERT_SUCCESS(mAttributeIterator_ReplaceCurrentValueWithUInt64(&iterator, 2));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeType(&iterator, &type));
    mTEST_ASSERT_EQUAL(type, mA_T_UInt64);

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentValue(&iterator, &u64));
    mTEST_ASSERT_EQUAL(u64, 2);

    mTEST_ASSERT_SUCCESS(mAttributeIterator_ReplaceCurrentValueWithString(&iterator, "still not test"));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_ReplaceCurrentValueWithInt64(&iterator, -2));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeType(&iterator, &type));
    mTEST_ASSERT_EQUAL(type, mA_T_Int64);

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentValue(&iterator, &i64));
    mTEST_ASSERT_EQUAL(i64, -2);

    mTEST_ASSERT_SUCCESS(mAttributeIterator_ReplaceCurrentValueWithString(&iterator, "still not test"));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_ReplaceCurrentValueWithBool(&iterator, false));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeType(&iterator, &type));
    mTEST_ASSERT_EQUAL(type, mA_T_Bool);

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentValue(&iterator, &b));
    mTEST_ASSERT_EQUAL(b, false);

    mTEST_ASSERT_SUCCESS(mAttributeIterator_ReplaceCurrentValueWithString(&iterator, "still not test"));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_ReplaceCurrentValueWithFloat64(&iterator, 2.345));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeType(&iterator, &type));
    mTEST_ASSERT_EQUAL(type, mA_T_Float64);

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentValue(&iterator, &f64));
    mTEST_ASSERT_EQUAL(f64, 2.345);

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentValue(&iterator, &f32));
    mTEST_ASSERT_EQUAL(f32, 2.345f);

    mTEST_ASSERT_SUCCESS(mAttributeIterator_ReplaceCurrentValueWithString(&iterator, "still not test"));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_ReplaceCurrentValueWithVector2(&iterator, mVec2f(-3)));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeType(&iterator, &type));
    mTEST_ASSERT_EQUAL(type, mA_T_Vector2);

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentValue(&iterator, &v2));
    mTEST_ASSERT_EQUAL(v2, mVec2f(-3));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_ReplaceCurrentValueWithString(&iterator, "still not test"));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_ReplaceCurrentValueWithVector3(&iterator, mVec3f(-4)));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeType(&iterator, &type));
    mTEST_ASSERT_EQUAL(type, mA_T_Vector3);

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentValue(&iterator, &v3));
    mTEST_ASSERT_EQUAL(v3, mVec3f(-4));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_ReplaceCurrentValueWithString(&iterator, "still not test"));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_ReplaceCurrentValueWithVector4(&iterator, mVec4f(-5)));

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentAttributeType(&iterator, &type));
    mTEST_ASSERT_EQUAL(type, mA_T_Vector4);

    mTEST_ASSERT_SUCCESS(mAttributeIterator_GetCurrentValue(&iterator, &v4));
    mTEST_ASSERT_EQUAL(v4, mVec4f(-5));


    mTEST_ASSERT_SUCCESS(mAttributeIterator_LeaveObject(&iterator));
  }

  mTEST_ALLOCATOR_ZERO_CHECK();
}
