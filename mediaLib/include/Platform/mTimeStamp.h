#ifndef mTimeStamp_h__
#define mTimeStamp_h__

#include "mediaLib.h"

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
