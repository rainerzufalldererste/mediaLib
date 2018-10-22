#ifndef mTypeTraits_h__
#define mTypeTraits_h__

#include <stdint.h>
#include <type_traits>

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

#endif // mTypeTraits_h__
