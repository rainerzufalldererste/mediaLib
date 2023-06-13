#ifndef mForwardTuple_h__
#define mForwardTuple_h__

#pragma warning(push)
#pragma warning(disable: 4100) // unreferenced formal parameter.

template <typename TFunction, typename T, typename Tuple, bool Done, int Total, int... N>
struct mTupleForwarder_Internal
{
  static mResult Unpack(TFunction function, T &parameter, Tuple && t)
  {
    return mTupleForwarder_Internal<TFunction, T, Tuple, Total == 1 + sizeof...(N), Total, N..., sizeof...(N)>::Unpack(function, parameter, std::forward<Tuple>(t));
  }
};

template <typename TFunction, typename T, typename Tuple, int Total, int... N>
struct mTupleForwarder_Internal<TFunction, T, Tuple, true, Total, N...>
{
  static mResult Unpack(TFunction function, T &parameter, Tuple && t)
  {
    return function(parameter, std::get<N>(std::forward<Tuple>(t))...);
  }
};

template <typename TFunction, typename T, typename Tuple>
mResult mForwardTuple(TFunction f, T &parameter, Tuple && t)
{
  typedef typename std::decay<Tuple>::type ttype;
  return mTupleForwarder_Internal<TFunction, T, Tuple, 0 == std::tuple_size<ttype>::value, std::tuple_size<ttype>::value>::Unpack(f, parameter, std::forward<Tuple>(t));
}

#pragma warning(pop)

#endif // mForwardTuple_h__
