#ifndef mTimeStamp_h__
#define mTimeStamp_h__

#include "mediaLib.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "lUcW5k0a6WtItocdiHxepsYQ7VbeY5/04cu7fOewzBG3PzNXNuGzBjuOv5ayyH210Xdx2paMWtlyWXeB"
#endif

struct mTimeStamp
{
  double_t timePoint;

  mTimeStamp operator +(const mTimeStamp t) const;
  mTimeStamp operator -(const mTimeStamp t) const;
  mTimeStamp& operator +=(const mTimeStamp t);
  mTimeStamp& operator -=(const mTimeStamp t);
  bool operator <(const mTimeStamp t) const;
  bool operator <=(const mTimeStamp t) const;
  bool operator >(const mTimeStamp t) const;
  bool operator >=(const mTimeStamp t) const;
  bool operator ==(const mTimeStamp t) const;
  bool operator !=(const mTimeStamp t) const;
};

mFUNCTION(mTimeStamp_Now, OUT mTimeStamp *pTimeStamp);
mFUNCTION(mTimeStamp_FromSeconds, OUT mTimeStamp *pTimeStamp, const double_t seconds);


#endif // mTimeStamp_h__
