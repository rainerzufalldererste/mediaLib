#ifndef mProfiler_h__
#define mProfiler_h__

#include "mediaLib.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "Da+ha1ASTY4vXT0NRFALTtpfjqZIFX16JTNdyd7AUrqjnpxUB1x7iMePHZF3O73+RjvE6bjXJcbWxZo3"
#endif

// There can only be one global profiler which the `mProfiler_ReportEventStart` and `mProfiler_ReportEventEnd` functions report to. The Profiler cannot be freed because other processes might still be writing to it.
// The Profiler has a fixed amount of events that can be reported until it loops around or is disabled.

#define mPROFILER_MAX_NAME_LENGHT 64

struct mProfiler_Record
{
  char name[mPROFILER_MAX_NAME_LENGHT];
  size_t eventTime;
  uint32_t asyncOrThreadIndex;
  bool eventStart;
};

struct mProfiler
{
  volatile int64_t enabled; // zero if not enabled, non-zero if enabled.
  mProfiler_Record *pRecords;
  size_t count;
  volatile int64_t index;
};

extern mProfiler _GlobalProfiler;

//////////////////////////////////////////////////////////////////////////

// `eventCount`: must be a power of two.
mFUNCTION(mProfiler_Allocate, const size_t eventCount);

mFUNCTION(mProfiler_Enable);
mFUNCTION(mProfiler_Disable);
mFUNCTION(mProfiler_Reset);

mFUNCTION(mProfiler_WriteToChromiumTraceEventFile, const mString &filename);

//////////////////////////////////////////////////////////////////////////

// Maximum Name Length: mPROFILER_MAX_NAME_LENGHT
inline void mProfiler_ReportEventStart(const char *name, const uint32_t asyncOrThreadIndex = GetCurrentThreadId())
{
  if (_GlobalProfiler.enabled == 0 || _GlobalProfiler.pRecords == nullptr)
    return;

  const size_t index = (_InterlockedIncrement64(&_GlobalProfiler.index) - 1) & (_GlobalProfiler.count - 1);

  _GlobalProfiler.pRecords[index].eventTime = mGetCurrentTimeNs();
  _GlobalProfiler.pRecords[index].eventStart = true;
  _GlobalProfiler.pRecords[index].asyncOrThreadIndex = asyncOrThreadIndex;
  
  const size_t stringLength = strnlen(name, mPROFILER_MAX_NAME_LENGHT - 1);

  memcpy(_GlobalProfiler.pRecords[index].name, name, stringLength);
  _GlobalProfiler.pRecords[index].name[stringLength] = '\0';
  _GlobalProfiler.pRecords[index].name[mPROFILER_MAX_NAME_LENGHT - 1] = '\0';
}

inline void mProfiler_ReportEventEnd(const char *name, const uint32_t asyncOrThreadIndex = GetCurrentThreadId())
{
  if (_GlobalProfiler.enabled == 0 || _GlobalProfiler.pRecords == nullptr)
    return;

  const size_t index = (_InterlockedIncrement64(&_GlobalProfiler.index) - 1) & (_GlobalProfiler.count - 1);

  _GlobalProfiler.pRecords[index].eventTime = mGetCurrentTimeNs();
  _GlobalProfiler.pRecords[index].eventStart = false;
  _GlobalProfiler.pRecords[index].asyncOrThreadIndex = asyncOrThreadIndex;

  const size_t stringLength = strnlen(name, mPROFILER_MAX_NAME_LENGHT - 1);

  memcpy(_GlobalProfiler.pRecords[index].name, name, stringLength);
  _GlobalProfiler.pRecords[index].name[stringLength] = '\0';
  _GlobalProfiler.pRecords[index].name[mPROFILER_MAX_NAME_LENGHT - 1] = '\0';
}

//////////////////////////////////////////////////////////////////////////

void mProfiler_ReportEventStart_NoInline(const char *name, const uint32_t asyncOrThreadIndex = GetCurrentThreadId());
void mProfiler_ReportEventEnd_NoInline(const char *name, const uint32_t asyncOrThreadIndex = GetCurrentThreadId());

//////////////////////////////////////////////////////////////////////////

class mScopedProfileEvent
{
private:
  const char *name;
  uint32_t asyncOrThreadIndex;

public:
  inline mScopedProfileEvent(const char *name, const uint32_t asyncOrThreadIndex = GetCurrentThreadId()) :
    name(name),
    asyncOrThreadIndex(asyncOrThreadIndex)
  {
    mProfiler_ReportEventStart(name, asyncOrThreadIndex);
  }

  inline ~mScopedProfileEvent()
  {
    mProfiler_ReportEventEnd(name, asyncOrThreadIndex);
  }
};

//////////////////////////////////////////////////////////////////////////

#ifdef GIT_BUILD
 #define mPROFILER_DO_NOT_PROFILE
#endif

#ifndef mPROFILER_DO_NOT_PROFILE
 #define mPROFILE_SCOPED(...) mScopedProfileEvent mCONCAT_LITERALS(__scoped_profile__, __COUNTER__)(__VA_ARGS__)
#else
 #define mPROFILE_SCOPED(...)
#endif

#endif // mProfiler_h__
