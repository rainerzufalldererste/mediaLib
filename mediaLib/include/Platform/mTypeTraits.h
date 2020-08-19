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

#endif // mTypeTraits_h__
