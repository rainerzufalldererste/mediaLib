// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

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
