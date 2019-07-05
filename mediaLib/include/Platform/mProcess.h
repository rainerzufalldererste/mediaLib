#ifndef mProcess_h__
#define mProcess_h__

#include "mediaLib.h"
#include "mPipe.h"

struct mProcess;

// The `mProcess` struct contains references to the pipes to make sure they can't be destroyed whilst being in use.
mFUNCTION(mProcess_Create, OUT mPtr<mProcess> *pProcess, IN mAllocator *pAllocator, const mString &executable, const mString &workingDirectory, const mString &params, OPTIONAL mPtr<mPipe> &stdinPipe, OPTIONAL mPtr<mPipe> &stdoutPipe, OPTIONAL mPtr<mPipe> &stderrPipe);
mFUNCTION(mProcess_Destroy, IN_OUT mPtr<mProcess> *pProcess);

mFUNCTION(mProcess_IsRunning, mPtr<mProcess> &process, OUT bool *pIsRunning);
mFUNCTION(mProcess_GetExitCode, mPtr<mProcess> &process, OUT uint32_t *pExitCode);
mFUNCTION(mProcess_Terminate, mPtr<mProcess> &process, const uint32_t exitCode = (uint32_t)-1);

mFUNCTION(mProcess_Run, const mString &executable, const mString &workingDirectory, const mString &params);
mFUNCTION(mProcess_Run, const mString &executable, const mString &workingDirectory, const mString &params, OPTIONAL mPtr<mPipe> stdinPipe, OPTIONAL mPtr<mPipe> stdoutPipe, OPTIONAL mPtr<mPipe> stderrPipe);
mFUNCTION(mProcess_RunAsElevated, const mString &executable, const mString &workingDirectory, const mString &params);

#endif // mProcess_h__
