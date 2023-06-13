#ifndef mTypeTraits_h__
#define mTypeTraits_h__

#include <stdint.h>
#include <type_traits>

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "TR5gR65VlDVpUt04X9YEX1BI8PqeKp8AEBMnKZwxm7Ws60MZoo0BQytATfA63QZG6tmLYFAu9zSIvThz"
#endif

template <typename T>
struct mIsTriviallyMemoryMovable
{
  static constexpr bool value = std::is_trivially_move_constructible<T>::value && std::is_trivially_move_assignable<T>::value;
};

template <typename T>
void mMoveConstructMultiple(T *pDestination, T *pSource, const size_t count)
{
  for (size_t i = 0; i < count; i++)
    new (&pDestination[i]) T(std::move(pSource[i]));
}

template <typename T>
void mCopyConstructMultiple(T *pDestination, T *pSource, const size_t count)
{
  for (size_t i = 0; i < count; i++)
    new (&pDestination[i]) T(pSource[i]);
}

template <typename T, typename T2 = T>
struct mEquals
{
  bool operator ()(const T &a, const T2 &b)
  {
    return a == b;
  }
};

template <typename T, typename T2 = T>
struct mEqualsValue
{
  bool operator ()(const T *pA, const T2 *pB)
  {
    if (pA == pB)
      return true;

    if ((pA == nullptr) ^ (pB == nullptr))
      return false;

    return *pA == *pB;
  }
};

struct mTrue
{
  template <typename... TArgs>
  inline mTrue(TArgs ...) { }

  inline operator bool() const
  {
    return true;
  }

  template <typename... TArgs>
  inline bool operator ()(TArgs ...) const
  {
    return *this;
  }
};

struct mFalse
{
  template <typename... TArgs>
  inline mFalse(TArgs ...) { }

  inline operator bool() const
  {
    return false;
  }

  template <typename... TArgs>
  inline bool operator ()(TArgs ...) const
  {
    return *this;
  }
};

template <typename T>
struct mEqualTo
{
private:
  bool equal;

public:
  inline mEqualTo(const T &a, const T &b) : 
    equal(a == b)
  { }

  inline operator bool() const
  {
    return equal;
  }
};

template <typename T, T TFunc>
struct mFnWrapper
{
  inline mFnWrapper() { }

  template<typename... Args>
  auto operator ()(Args... args) -> decltype(TFunc(std::forward<Args>(args)...)) const
  {
    return TFunc(std::forward<Args>(args)...);
  }
};

#define mFN_WRAPPER(fn) mFnWrapper<decltype((fn)), (fn)>

constexpr inline int64_t mIsEquivalentIntegerTypeSigned(int64_t) { return true; }
constexpr inline int32_t mIsEquivalentIntegerTypeSigned(int32_t) { return true; }
constexpr inline long mIsEquivalentIntegerTypeSigned(long) { return true; }
constexpr inline int16_t mIsEquivalentIntegerTypeSigned(int16_t) { return true; }
constexpr inline int8_t mIsEquivalentIntegerTypeSigned(int8_t) { return true; }
constexpr inline uint64_t mIsEquivalentIntegerTypeSigned(uint64_t) { return false; }
constexpr inline uint32_t mIsEquivalentIntegerTypeSigned(uint32_t) { return false; }
constexpr inline unsigned long mIsEquivalentIntegerTypeSigned(unsigned long) { return false; }
constexpr inline uint16_t mIsEquivalentIntegerTypeSigned(uint16_t) { return false; }
constexpr inline uint8_t mIsEquivalentIntegerTypeSigned(uint8_t) { return false; }

template <typename T>
struct mEnumEquivalentIntegerType
{
  typedef decltype(mIsEquivalentIntegerTypeSigned((T)0)) type;
};

template <typename T>
struct mUnsignedEquivalent
{

};

template <>
struct mUnsignedEquivalent<int64_t>
{
  typedef uint64_t type;
};

template <>
struct mUnsignedEquivalent<int32_t>
{
  typedef uint32_t type;
};

template <>
struct mUnsignedEquivalent<long>
{
  typedef unsigned long type;
};

template <>
struct mUnsignedEquivalent<int16_t>
{
  typedef uint16_t type;
};

template <>
struct mUnsignedEquivalent<int8_t>
{
  typedef uint8_t type;
};

#endif // mTypeTraits_h__
