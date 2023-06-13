#ifndef mProcess_h__
#define mProcess_h__

#include "mediaLib.h"
#include "mPipe.h"
#include "mQueue.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "hycThu8Pf6025725zSBbOqBQJfDJqf/Dn2pv3My4fyqZxY9SOG0HcSt6G4xlxJ4U51Q2hWvsw/uPKvEn"
#endif

//////////////////////////////////////////////////////////////////////////

struct mProcessInfo
{
  mString filename;
  size_t processId;
};

mFUNCTION(mProcess_GetRunningProcesses, OUT mPtr<mQueue<mProcessInfo>> *pProcesses, IN mAllocator *pAllocator);

//////////////////////////////////////////////////////////////////////////

typedef size_t mProcess_CreationFlags;

enum mProcess_CreationFlags_
{
  mP_CF_None = 0,
  mP_CF_NoWindow = 1 << 0,
  mP_CF_Restricted = 1 << 1,
  mP_CF_SecureProcess = 1 << 2,
};

struct mProcess;

// The `mProcess` struct contains references to the pipes to make sure they can't be destroyed whilst being in use.
mFUNCTION(mProcess_Create, OUT mPtr<mProcess> *pProcess, IN mAllocator *pAllocator, const mString &executable, const mString &workingDirectory, const mString &params, OPTIONAL mPtr<mPipe> &stdinPipe, OPTIONAL mPtr<mPipe> &stdoutPipe, OPTIONAL mPtr<mPipe> &stderrPipe, const mProcess_CreationFlags flags = mP_CF_None);
mFUNCTION(mProcess_CreateFromProcessId, OUT mPtr<mProcess> *pProcess, IN mAllocator *pAllocator, const size_t processId);
mFUNCTION(mProcess_GetCurrentProcess, OUT mPtr<mProcess> *pProcess, IN mAllocator *pAllocator);
mFUNCTION(mProcess_Destroy, IN_OUT mPtr<mProcess> *pProcess);

mFUNCTION(mProcess_IsRunning, mPtr<mProcess> &process, OUT bool *pIsRunning);
mFUNCTION(mProcess_GetExitCode, mPtr<mProcess> &process, OUT uint32_t *pExitCode);
mFUNCTION(mProcess_Terminate, mPtr<mProcess> &process, const uint32_t exitCode = (uint32_t)-1);
mFUNCTION(mProcess_WaitForExit, mPtr<mProcess> &process, const uint32_t timeoutMs = (uint32_t)-1);
mFUNCTION(mProcess_GetHandle, mPtr<mProcess> &process, OUT void **ppHandle);

mFUNCTION(mProcess_Run, const mString &executable, const mString &workingDirectory, const mString &params, const mProcess_CreationFlags flags = mP_CF_None);
mFUNCTION(mProcess_Run, const mString &executable, const mString &workingDirectory, const mString &params, OPTIONAL mPtr<mPipe> stdinPipe, OPTIONAL mPtr<mPipe> stdoutPipe, OPTIONAL mPtr<mPipe> stderrPipe, const mProcess_CreationFlags flags = mP_CF_None);
mFUNCTION(mProcess_RunAsElevated, const mString &executable, const mString &workingDirectory, const mString &params);

struct mProcessJob;

mFUNCTION(mProcessJob_Create, OUT mPtr<mProcessJob> *pJob, IN mAllocator *pAllocator, const mString &name);
mFUNCTION(mProcessJob_Destroy, IN_OUT mPtr<mProcessJob> *pJob);

mFUNCTION(mProcess_IsInJob, mPtr<mProcess> &process, mPtr<mProcessJob> &job, OUT bool *pIsInJob);
mFUNCTION(mProcess_AssignToJob, mPtr<mProcess> &process, mPtr<mProcessJob> &job);

mFUNCTION(mProcessJob_CloseHandleOnDestroy, mPtr<mProcessJob> &job);
mFUNCTION(mProcessJob_SetKillAllOnClose, mPtr<mProcessJob> &job);

#endif // mProcess_h__
