#ifndef mAttribute_h__
#define mAttribute_h__

#include "mediaLib.h"
#include "mQueue.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "GWjROKAX2JS1+0yoBIhZ9ksccAF76DXPT8gmZ7Yqo8uu0WCmsowtHU+rC/SqpxyEtQ1ro+RVKOySPAVL"
#endif

struct mAttribute;

constexpr size_t mAttribute_NameLength = 64;

enum mAttribute_Type
{
  mA_T_Null,

  mA_T_UInt64,
  mA_T_Int64,
  mA_T_Bool,
  mA_T_Float64,
  mA_T_Vector2,
  mA_T_Vector3,
  mA_T_Vector4,
  mA_T_String,
  mA_T_Object,

  _mA_T_ActiveValueCount,

  mA_T_SkipTo,
  mA_T_EmptyObject,

  mA_T_InactiveValue = 0x100,

  mA_T_SkipUInt64 = mA_T_InactiveValue | mA_T_UInt64,
  mA_T_SkipInt64 = mA_T_InactiveValue | mA_T_Int64,
  mA_T_SkipBool = mA_T_InactiveValue | mA_T_Bool,
  mA_T_SkipFloat64 = mA_T_InactiveValue | mA_T_Float64,
  mA_T_SkipVector2 = mA_T_InactiveValue | mA_T_Vector2,
  mA_T_SkipVector3 = mA_T_InactiveValue | mA_T_Vector3,
  mA_T_SkipVector4 = mA_T_InactiveValue | mA_T_Vector4,
  mA_T_SkipString = mA_T_InactiveValue | mA_T_String,
  mA_T_SkipObject = mA_T_InactiveValue | mA_T_Object,
};

struct mAttribute
{
  mAttribute_Type type;
  mInplaceString<mAttribute_NameLength> name;

  union
  {
    struct
    {
      uint64_t value;
    } UInt64;

    struct
    {
      int64_t value;
    } Int64;

    struct
    {
      bool value;
    } Bool;

    struct
    {
      double_t value;
    } Float64;

    struct
    {
      mVec2f value;
    } Vector2;

    struct
    {
      mVec3f value;
    } Vector3;

    struct
    {
      mVec4f value;
    } Vector4;

    struct
    {
      mString *pValue;
    } String;

    struct
    {
      uint32_t size, skip;
      size_t parentIndex;
    } Object;

    struct 
    {
      size_t nextAttributeIndex;
    } SkipTo;
  };

  mAttribute();
  mAttribute(mAttribute &&move);
  mAttribute & operator = (mAttribute &&move);
  ~mAttribute();
};

mFUNCTION(mDestruct, IN_OUT mAttribute *pAttribute);

inline mAttribute::mAttribute()
{
  mMemset(this, 1);
}

inline mAttribute::mAttribute(mAttribute &&move)
{
  memmove(this, &move, sizeof(move));

  if (type == mA_T_String || type == mA_T_SkipString)
  {
    move.String.pValue = nullptr;
    move.type = mA_T_Null;
  }
}

inline mAttribute & mAttribute::operator=(mAttribute &&move)
{
  mDestruct(this);

  memmove(this, &move, sizeof(move));

  if (type == mA_T_String || type == mA_T_SkipString)
  {
    move.String.pValue = nullptr;
    move.type = mA_T_Null;
  }

  return *this;
}

inline mAttribute::~mAttribute()
{
  mDestruct(this);
}

struct mAttributeStore;

struct mAttributeCollection
{
  size_t attributeStartIndex;
  mAttributeStore *pStore;
};

struct mAttributeIterator
{
  mAttributeStore *pStore;
  size_t objectIndex, currentContainerAttributeIndex, currentAttributeIndex;
};

struct mAttributeStore
{
  mUniqueContainer<mQueue<mAttribute>> attributes;
};

mFUNCTION(mAttributeCollection_Create, OUT mAttributeCollection *pAttributeCollection, IN mAttributeStore *pAttributeStore);
mFUNCTION(mAttributeCollection_CreateIterator, IN mAttributeCollection *pAttributeCollection, OUT mAttributeIterator *pAttributeIterator);

mFUNCTION(mAttributeStore_Create, OUT mAttributeStore *pAttributeStore, IN mAllocator *pAllocator);
mFUNCTION(mAttributeStore_Destroy, IN_OUT mAttributeStore *pAttributeStore);

mFUNCTION(mAttributeIterator_Reset, IN_OUT mAttributeIterator *pAttributeIterator);
mFUNCTION(mAttributeIterator_Next, IN_OUT mAttributeIterator *pAttributeIterator);
mFUNCTION(mAttributeIterator_Find, IN_OUT mAttributeIterator *pAttributeIterator, const mString &name);

mFUNCTION(mAttributeIterator_AddUInt64, IN_OUT mAttributeIterator *pAttributeIterator, const mString &name, const uint64_t value);
mFUNCTION(mAttributeIterator_AddInt64, IN_OUT mAttributeIterator *pAttributeIterator, const mString &name, const int64_t value);
mFUNCTION(mAttributeIterator_AddBool, IN_OUT mAttributeIterator *pAttributeIterator, const mString &name, const bool value);
mFUNCTION(mAttributeIterator_AddFloat64, IN_OUT mAttributeIterator *pAttributeIterator, const mString &name, const double_t value);
mFUNCTION(mAttributeIterator_AddVector2, IN_OUT mAttributeIterator *pAttributeIterator, const mString &name, const mVec2f value);
mFUNCTION(mAttributeIterator_AddVector3, IN_OUT mAttributeIterator *pAttributeIterator, const mString &name, const mVec3f value);
mFUNCTION(mAttributeIterator_AddVector4, IN_OUT mAttributeIterator *pAttributeIterator, const mString &name, const mVec4f value);
mFUNCTION(mAttributeIterator_AddString, IN_OUT mAttributeIterator *pAttributeIterator, const mString &name, const mString &value);
mFUNCTION(mAttributeIterator_AddObject, IN_OUT mAttributeIterator *pAttributeIterator, const mString &name);

mFUNCTION(mAttributeIterator_GetCurrentAttributeName, IN mAttributeIterator *pAttributeIterator, OUT mString *pName);
mFUNCTION(mAttributeIterator_GetCurrentAttributeType, IN mAttributeIterator *pAttributeIterator, OUT mAttribute_Type *pType);

mFUNCTION(mAttributeIterator_GetCurrentValue, IN mAttributeIterator *pAttributeIterator, OUT uint64_t *pValue);
mFUNCTION(mAttributeIterator_GetCurrentValue, IN mAttributeIterator *pAttributeIterator, OUT int64_t *pValue);
mFUNCTION(mAttributeIterator_GetCurrentValue, IN mAttributeIterator *pAttributeIterator, OUT bool *pValue);
mFUNCTION(mAttributeIterator_GetCurrentValue, IN mAttributeIterator *pAttributeIterator, OUT double_t *pValue);
mFUNCTION(mAttributeIterator_GetCurrentValue, IN mAttributeIterator *pAttributeIterator, OUT float_t *pValue);
mFUNCTION(mAttributeIterator_GetCurrentValue, IN mAttributeIterator *pAttributeIterator, OUT mVec2f *pValue);
mFUNCTION(mAttributeIterator_GetCurrentValue, IN mAttributeIterator *pAttributeIterator, OUT mVec3f *pValue);
mFUNCTION(mAttributeIterator_GetCurrentValue, IN mAttributeIterator *pAttributeIterator, OUT mVec4f *pValue);
mFUNCTION(mAttributeIterator_GetCurrentValue, IN mAttributeIterator *pAttributeIterator, OUT mString *pValue);

mFUNCTION(mAttributeIterator_GetCurrentObject, IN mAttributeIterator *pAttributeIterator, OUT mAttributeIterator *pChildAttributeIterator);

mFUNCTION(mAttributeIterator_DeactivateCurrentValue, IN mAttributeIterator *pAttributeIterator);

mFUNCTION(mAttributeIterator_ReplaceCurrentValueWithUInt64, IN_OUT mAttributeIterator *pAttributeIterator, const uint64_t value);
mFUNCTION(mAttributeIterator_ReplaceCurrentValueWithInt64, IN_OUT mAttributeIterator *pAttributeIterator, const int64_t value);
mFUNCTION(mAttributeIterator_ReplaceCurrentValueWithBool, IN_OUT mAttributeIterator *pAttributeIterator, const bool value);
mFUNCTION(mAttributeIterator_ReplaceCurrentValueWithFloat64, IN_OUT mAttributeIterator *pAttributeIterator, const double_t value);
mFUNCTION(mAttributeIterator_ReplaceCurrentValueWithVector2, IN_OUT mAttributeIterator *pAttributeIterator, const mVec2f value);
mFUNCTION(mAttributeIterator_ReplaceCurrentValueWithVector3, IN_OUT mAttributeIterator *pAttributeIterator, const mVec3f value);
mFUNCTION(mAttributeIterator_ReplaceCurrentValueWithVector4, IN_OUT mAttributeIterator *pAttributeIterator, const mVec4f value);
mFUNCTION(mAttributeIterator_ReplaceCurrentValueWithString, IN_OUT mAttributeIterator *pAttributeIterator, const mString &value);

#endif // mAttribute_h__
