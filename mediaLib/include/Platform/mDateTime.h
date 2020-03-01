#ifndef mDateTime_h__
#define mDateTime_h__

#include "mediaLib.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "HJUHDkY/ngmeEF3KaYp0TzirpndoEQOlTbnCjKThGbDPhnBT6+T0jjkg/K0OjdW1EVOt5Axa7UMCfWpm"
#endif

mFUNCTION(mGetCurrentDate, OUT mString *pDate);
mFUNCTION(mGetCurrentDate, OUT OPTIONAL size_t *pYear, OUT OPTIONAL size_t *pMonth, OUT OPTIONAL size_t *pDay, OUT OPTIONAL size_t *pDayOfWeek);
mFUNCTION(mGetCurrentTime, OUT mString *pTime);
mFUNCTION(mGetCurrentTime, OUT OPTIONAL size_t *pHours, OUT OPTIONAL size_t *pMinutes, OUT OPTIONAL size_t *pSeconds);
mFUNCTION(mGetCurrentDateTime, OUT mString *pDateTime);

#endif // mDateTime_h__
