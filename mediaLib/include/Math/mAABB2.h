#ifndef mAABB2_h__
#define mAABB2_h__

#include "mediaLib.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "gfnL8axBpKOXND7jioe6bhlbfhrrhPRahwc5ZXPXOVPBPBXaUqACEBWqRcXNUCgFE3Eyj8KXA/Z+yrQ1"
#endif

template <typename T>
struct mAABB2
{
  mAABB2();
  mAABB2(const mVec2t<T> position);
  mAABB2(const mVec2t<T> startPosition, const mVec2t<T> endPosition);

  bool Contains(const mVec2t<T> position) const;
  void GrowToContain(const mVec2t<T> position);

  mVec2t<T> GetStartPosition() const;
  mVec2t<T> GetEndPosition() const;

  void GetCorners(mVec2t<T> corners[4]) const;

  mVec2t<T> GetClosestPoint(const mVec2t<T> position) const;

  template<typename T, typename TRet = mMath_FloatTypeFrom<T>::type>
  inline TRet mAABB2<T>::GetClosestDistance(const mVec2t<T> position) const;

  template<typename T, typename TRet = mMath_FloatTypeFrom<T>::type>
  inline TRet mAABB2<T>::GetClosestDistanceSquared(const mVec2t<T> position) const;

  mVec2t<T> startPos, endPos;
};

template<typename T>
inline mAABB2<T>::mAABB2()
  : startPos(mVec2t<T>(0))
  , endPos(mVec2t<T>(0))
{ }

template<typename T>
inline mAABB2<T>::mAABB2(const mVec2t<T> position)
{
  startPos = position;
  endPos = position;
}

template<typename T>
inline mAABB2<T>::mAABB2(const mVec2t<T> startPosition, const mVec2t<T> endPosition)
{
  startPos = startPosition;
  endPos = endPosition;

  if (startPos.x > endPos.x)
    std::swap(startPos.x, endPos.x);

  if (startPos.y > endPos.y)
    std::swap(startPos.y, endPos.y);
}

template<typename T>
inline void mAABB2<T>::GrowToContain(const mVec2t<T> position)
{
  if (Contains(position))
    return;

  if (startPos.x > position.x)
    startPos.x = position.x;
  else if (endPos.x < position.x)
    endPos.x = position.x;

  if (startPos.y > position.y)
    startPos.y = position.y;
  else if (endPos.y < position.y)
    endPos.y = position.y;
}

template<typename T> inline bool mAABB2<T>::Contains(const mVec2t<T> position) const
{ 
  return position.x >= startPos.x && position.y >= startPos.y && position.x <= endPos.x && position.y <= endPos.y;
}

template<typename T>
void mAABB2<T>::GetCorners(mVec2t<T> corners[4]) const
{
  corners[0] = startPos;
  corners[1] = mVec2t<T>(endPos.x, startPos.y);
  corners[2] = mVec2t<T>(startPos.x, endPos.y);
  corners[3] = endPos;
}

template<typename T>
inline mVec2t<T> mAABB2<T>::GetClosestPoint(const mVec2t<T> position) const
{
  if (Contains(position))
    return position;

  mVec2t<T> ret = position;

  if (position.x > endPos.x)
    ret.x = endPos.x;
  else if (position.x < startPos.x)
    ret.x = startPos.x;

  if (position.y > endPos.y)
    ret.y = endPos.y;
  else if (position.y < startPos.y)
    ret.y = startPos.y;
}

template<typename T, typename TRet>
inline TRet mAABB2<T>::GetClosestDistance(const mVec2t<T> position) const
{
  if (Contains(position))
    return (TRet)0;

  int64_t xcomp = 0;
  int64_t ycomp = 0;

  if (position.x > endPos.x)
    xcomp = 1;
  else if (position.x < startPos.x)
    xcomp = -1;

  if (position.y > endPos.y)
    ycomp = 1;
  else if (position.y < startPos.y)
    ycomp = -1;

  if (xcomp)
  {
    if (!ycomp)
    {
      if (xcomp == 1)
        return (TRet)(position.x - endPos.x);

      return (TRet)(startPos.x - position.x);
    }
    else
    {
      mVec2t<T> comp;

      if (xcomp == 1)
        comp.x = endPos.x;
      else
        comp.x = startPos.x;

      if (ycomp == 1)
        comp.y = endPos.y;
      else
        comp.y = startPos.y;

      return (TRet)mVec2t<T>(comp - position).Length();
    }
  }
  else // ycomp & !xcomp
  {
    if (ycomp == 1)
      return (TRet)(position.y - endPos.y);

    return (TRet)(startPos.y - position.y);
  }
}

template<typename T, typename TRet>
inline TRet mAABB2<T>::GetClosestDistanceSquared(const mVec2t<T> position) const
{
  if (Contains(position))
    return 0.0;

  int64_t xcomp = 0;
  int64_t ycomp = 0;

  if (position.x > endPos.x)
    xcomp = 1;
  else if (position.x < startPos.x)
    xcomp = -1;

  if (position.y > endPos.y)
    ycomp = 1;
  else if (position.y < startPos.y)
    ycomp = -1;

  if (xcomp)
  {
    if (!ycomp)
    {
      if (xcomp == 1)
      {
        const TRet ret = (TRet)(position.x - endPos.x);
        return (TRet)(ret * ret);
      }

      const TRet ret = (TRet)(startPos.x - position.x);
      return (TRet)(ret * ret);
    }
    else
    {
      mVec2t<T> comp;

      if (xcomp == 1)
        comp.x = endPos.x;
      else
        comp.x = startPos.x;

      if (ycomp == 1)
        comp.y = endPos.y;
      else
        comp.y = startPos.y;

      return (TRet)mVec2t<T>(comp - position).LengthSquared();
    }
  }
  else // ycomp & !xcomp
  {
    if (ycomp == 1)
    {
      const TRet ret = (TRet)(position.y - endPos.y);
      return (TRet)(ret * ret);
    }

    const TRet ret = (TRet)(startPos.y - position.y);
    return (TRet)(ret * ret);
  }
}

template<typename T> inline mVec2t<T> mAABB2<T>::GetStartPosition() const
{
  return startPos;
}

template<typename T> inline mVec2t<T> mAABB2<T>::GetEndPosition() const
{
  return endPos;
}

#endif // mAABB2_h__
