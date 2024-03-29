#ifndef mNamedEnum_h__
#define mNamedEnum_h__

#include "mediaLib.h"
#include "mHashMap.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "fH8wkLaG3mRNYopoqIJj28idLsRbWPTUQORTrukt8IF+ZGlVTYek2VaMXhBZgvbB76U9Bb2hb2UkFC1x"
#endif

template <typename T>
struct mNamedEnumWrapper
{
  ~mNamedEnumWrapper();

  mNamedEnumWrapper<T> & Initialize(const std::initializer_list<T> &values, const char *commaSeparatedNames);
  mNamedEnumWrapper<T> & Initialize(const std::initializer_list<T> &values, const std::initializer_list<const char *> &names);

  size_t m_count = 0;
  mPtr<mHashMap<mString, T>> m_byName = nullptr;
  const char **m_pByValue = nullptr;
  char *m_commaSeparatedNames = nullptr;
  bool isInitialized = false;
  size_t *m_pReferenceCount = nullptr;
};

template<typename T>
inline mNamedEnumWrapper<T> & mNamedEnumWrapper_GetNamedEnumWrapper();

template <typename T>
T mNamedEnumWrapper_GetIndex(const mString &value);

template <typename T>
const char * mNamedEnumWrapper_GetName(const T value);

#define mNAMED_ENUM_WRAPPER_PREFIX __NAMED_ENUM_WRAPPER__
#define mNAMED_ENUM_INDEX_OF(NameOfEnum, value) mNamedEnumWrapper_GetIndex<NameOfEnum>((value))
#define mNAMED_ENUM_NAME_OF(value) mNamedEnumWrapper_GetName((value))

#define mNAMED_ENUM(Name, ...) \
enum Name { mCONCAT_LITERALS(Name, _Invalid), __VA_ARGS__, mCONCAT_LITERALS(Name, _Count) }; \
static mNamedEnumWrapper<Name> mCONCAT_LITERALS(mNAMED_ENUM_WRAPPER_PREFIX, Name) = mNamedEnumWrapper_GetNamedEnumWrapper<Name>().Initialize({ mCONCAT_LITERALS(Name, _Invalid), __VA_ARGS__ }, mCONCAT_LITERALS(mCONCAT_LITERALS(#Name, "_Invalid, "), #__VA_ARGS__))

#define mNAMED_ENUM_COMMA(x) x ,
#define mNAMED_ENUM_STRINGIFY_COMMA(x) #x ,
#define mNAMED_ENUM_FROM_X_MACRO(Name, xmacro) enum Name { mCONCAT_LITERALS(Name, _Invalid), xmacro(mNAMED_ENUM_COMMA) mCONCAT_LITERALS(Name, _Count) }; \
static mNamedEnumWrapper<Name> mCONCAT_LITERALS(mNAMED_ENUM_WRAPPER_PREFIX, Name) = mNamedEnumWrapper_GetNamedEnumWrapper<Name>().Initialize({ mCONCAT_LITERALS(Name, _Invalid), xmacro(mNAMED_ENUM_COMMA) }, { mCONCAT_LITERALS(#Name, "_Invalid, "), xmacro(mNAMED_ENUM_STRINGIFY_COMMA) } )

//////////////////////////////////////////////////////////////////////////

template<typename T>
inline mNamedEnumWrapper<T> & mNamedEnumWrapper<T>::Initialize(const std::initializer_list<T> &values, const char *commaSeparatedNames)
{
  if (isInitialized)
  {
    if (m_pReferenceCount != nullptr)
      ++*m_pReferenceCount;

    return *this;
  }

  m_count = values.size();

  mResult result = mR_Success;

  mERROR_IF_GOTO(commaSeparatedNames == nullptr, mR_ArgumentNull, result, epilogue);

  const size_t stringSize = strlen(commaSeparatedNames) + 1;

  mERROR_CHECK_GOTO(mAllocator_AllocateZero(nullptr, &m_pReferenceCount, 1), result, epilogue);
  ++*m_pReferenceCount;

  mERROR_CHECK_GOTO(mAllocator_AllocateZero(nullptr, &m_commaSeparatedNames, stringSize), result, epilogue);
  mERROR_CHECK_GOTO(mMemcpy(m_commaSeparatedNames, commaSeparatedNames, stringSize), result, epilogue);

  mERROR_CHECK_GOTO(mAllocator_AllocateZero(nullptr, &m_pByValue, m_count), result, epilogue);
  mERROR_CHECK_GOTO(mHashMap_Create(&m_byName, nullptr, 64), result, epilogue);

  char *lastIndex = m_commaSeparatedNames;
  char *nextIndex = lastIndex + 1;
  size_t count = 0;

  for (T value : values)
  {
    // find next value.
    while (true)
    {
      if (*nextIndex == ' ')
      {
        *nextIndex = '\0';
      }
      else if (*nextIndex == ',')
      {
        *nextIndex = '\0';
        ++nextIndex;

        while (true)
        {
          if (*nextIndex == ' ' || *nextIndex == '\t')
          {
            *nextIndex = '\0';
            ++nextIndex;
            continue;
          }
          else if (nextIndex == '\0')
          {
            nextIndex = nullptr;
            break;
          }

          break;
        }

        break;
      }
      else if (*nextIndex == '\0')
      {
        mERROR_IF_GOTO(count + 1 != m_count, mR_ArgumentNull, result, epilogue);
        nextIndex = nullptr;
        break;
      }

      ++nextIndex;
    }

    m_pByValue[count] = lastIndex;

    mString name;
    mERROR_CHECK_GOTO(mString_Create(&name, lastIndex, nullptr), result, epilogue);
    mERROR_CHECK_GOTO(mHashMap_Add(m_byName, std::move(name), &value), result, epilogue);

    if (nextIndex == nullptr || lastIndex == nullptr)
    {
      mERROR_IF_GOTO(count + 1 != m_count, mR_InvalidParameter, result, epilogue);
      break;
    }

    lastIndex = nextIndex;

    ++count;
    ++nextIndex;
  }

  isInitialized = true;
  return *this;

epilogue:
  m_count = 0;
  return *this;
}

template<typename T>
inline mNamedEnumWrapper<T> & mNamedEnumWrapper<T>::Initialize(const std::initializer_list<T> &values, const std::initializer_list<const char *> &names)
{
  if (isInitialized)
  {
    if (m_pReferenceCount != nullptr)
      ++*m_pReferenceCount;

    return *this;
  }

  mResult result = mR_Success;

  m_count = values.size();
  mERROR_IF_GOTO(m_count != names.size(), mR_IndexOutOfBounds, result, epilogue);

  mERROR_CHECK_GOTO(mAllocator_AllocateZero(nullptr, &m_pReferenceCount, 1), result, epilogue);
  ++*m_pReferenceCount;

  mERROR_CHECK_GOTO(mAllocator_AllocateZero(nullptr, &m_pByValue, m_count), result, epilogue);
  mERROR_CHECK_GOTO(mHashMap_Create(&m_byName, nullptr, 256), result, epilogue);

  auto _begin = names.begin();
  auto _end = names.end();

  size_t count = 0;

  for (T value : values)
  {
    mERROR_IF_GOTO(!(_begin != _end), mR_InvalidParameter, result, epilogue);

    const char *currentName = *_begin;

    m_pByValue[count] = currentName;
    count++;

    mString name;
    mERROR_CHECK_GOTO(mString_Create(&name, currentName, nullptr), result, epilogue);
    mERROR_CHECK_GOTO(mHashMap_Add(m_byName, std::move(name), &value), result, epilogue);

    ++_begin;
  }

  isInitialized = true;
  return *this;

epilogue:
  m_count = 0;
  return *this;
}

template<typename T>
inline mNamedEnumWrapper<T>::~mNamedEnumWrapper()
{
  if (m_pReferenceCount != nullptr)
  {
    --*m_pReferenceCount;

    if (m_pReferenceCount == 0)
    {
      mAllocator_FreePtr(nullptr, &m_pReferenceCount);
      mHashMap_Destroy(&m_byName);

      if (m_pByValue != nullptr)
        mAllocator_FreePtr(nullptr, &m_pByValue);

      if (m_commaSeparatedNames != nullptr)
        mAllocator_FreePtr(nullptr, &m_commaSeparatedNames);

      m_count = 0;
    }
  }
}

template<typename T>
inline mNamedEnumWrapper<T> & mNamedEnumWrapper_GetNamedEnumWrapper()
{
  static mNamedEnumWrapper<T> wrapper;

  return wrapper;
}

template<typename T>
inline T mNamedEnumWrapper_GetIndex(const mString &value)
{
  mNamedEnumWrapper<T> &wrapper = mNamedEnumWrapper_GetNamedEnumWrapper<T>();

  if (wrapper.m_count == 0)
    return (T)0;

  T index;
  const mResult result = mHashMap_Get(wrapper.m_byName, value, &index);

  if (mSUCCEEDED(result))
    return index;

  return (T)0;
}

template<typename T>
inline const char * mNamedEnumWrapper_GetName(const T value)
{
  mNamedEnumWrapper<T> &wrapper = mNamedEnumWrapper_GetNamedEnumWrapper<T>();

  if (wrapper.m_count == 0 || wrapper.m_count <= value)
    return "";

  return wrapper.m_pByValue[value];
}

#endif // mNamedEnum_h__
