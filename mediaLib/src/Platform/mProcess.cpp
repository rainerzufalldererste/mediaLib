#include "mProcess.h"

struct mProcess
{
  mPtr<mPipe> _stdin, _stderr, _stdout;
};

mFUNCTION(mProcess_Destroy_Internal, IN_OUT mProcess *pProcess);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mProcess_Create, OUT mPtr<mProcess> *pProcess, IN mAllocator *pAllocator, const mString &executable, const mString &workingDirectory, const mString &params, OPTIONAL mPtr<mPipe> &stdinPipe, OPTIONAL mPtr<mPipe> &stdoutPipe, OPTIONAL mPtr<mPipe> &stderrPipe)
{
  mFUNCTION_SETUP();

  mERROR_IF(pProcess == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mProcess_Run(executable, workingDirectory, params, stdinPipe, stdoutPipe, stderrPipe));

  mERROR_CHECK(mSharedPointer_Allocate<mProcess>(pProcess, pAllocator, [](mProcess *pData) { mProcess_Destroy_Internal(pData); }, 1));

  (*pProcess)->_stdin = stdinPipe;
  (*pProcess)->_stdout = stdoutPipe;
  (*pProcess)->_stderr = stderrPipe;

  mRETURN_SUCCESS();
}

mFUNCTION(mProcess_Destroy, IN_OUT mPtr<mProcess> *pProcess)
{
  mFUNCTION_SETUP();

  mERROR_IF(pProcess == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Destroy(pProcess));

  mRETURN_SUCCESS();
}

mFUNCTION(mProcess_Run, const mString &executable, const mString &workingDirectory, const mString &params)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mProcess_Run(executable, workingDirectory, params, mPtr<mPipe>(nullptr), mPtr<mPipe>(nullptr), mPtr<mPipe>(nullptr)));

  mRETURN_SUCCESS();
}

mFUNCTION(mProcess_Run, const mString &executable, const mString &workingDirectory, const mString &params, OPTIONAL mPtr<mPipe> stdinPipe, OPTIONAL mPtr<mPipe> stdoutPipe, OPTIONAL mPtr<mPipe> stderrPipe)
{
  mFUNCTION_SETUP();
  
  mERROR_IF(executable.hasFailed || executable.bytes <= 0 || workingDirectory.hasFailed || params.hasFailed, mR_InvalidParameter);

  wchar_t appName[MAX_PATH + 1];
  wchar_t workingDir[MAX_PATH + 1];
  wchar_t commandLine[1024 * 4]; // Theoretically should be max length 32768 according to msdn (https://docs.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-createprocessw).

  mERROR_CHECK(mString_ToWideString(executable, appName, mARRAYSIZE(appName)));
  mERROR_CHECK(mString_ToWideString(workingDirectory, workingDir, mARRAYSIZE(workingDir)));
  mERROR_CHECK(mString_ToWideString(params, commandLine, mARRAYSIZE(commandLine)));

  PROCESS_INFORMATION processInfo;
  STARTUPINFO startInfo;

  mZeroMemory(&processInfo);
  mZeroMemory(&startInfo);

  startInfo.cb = sizeof(STARTUPINFO);

  if (stdinPipe != nullptr)
    mERROR_CHECK(mPipe_GetReadHandle(stdinPipe, &startInfo.hStdInput));

  if (stdoutPipe != nullptr)
    mERROR_CHECK(mPipe_GetWriteHandle(stdoutPipe, &startInfo.hStdOutput));

  if (stderrPipe != nullptr)
    mERROR_CHECK(mPipe_GetWriteHandle(stderrPipe, &startInfo.hStdError));

  startInfo.dwFlags |= STARTF_USESTDHANDLES;

  mDEFER(
    CloseHandle(processInfo.hProcess);
    CloseHandle(processInfo.hThread);
  );

  if (FALSE == CreateProcessW(appName, commandLine, nullptr, nullptr, TRUE, 0, NULL, workingDir, &startInfo, &processInfo))
  {
    const DWORD error = GetLastError();
    mUnused(error);

    mRETURN_RESULT(mR_InternalError);
  }

  mRETURN_SUCCESS();
}

mFUNCTION(mProcess_RunAsElevated, const mString &executable, const mString &workingDirectory, const mString &params)
{
  mFUNCTION_SETUP();

  mERROR_IF(executable.hasFailed || executable.bytes <= 0 || workingDirectory.hasFailed || params.hasFailed, mR_InvalidParameter);

  wchar_t appName[MAX_PATH + 1];
  wchar_t workingDir[MAX_PATH + 1];
  wchar_t commandLine[1024 * 4]; // Should probably be of size 32768 (https://docs.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-createprocessw).

  mERROR_CHECK(mString_ToWideString(executable, appName, mARRAYSIZE(appName)));
  mERROR_CHECK(mString_ToWideString(workingDirectory, workingDir, mARRAYSIZE(workingDir)));
  mERROR_CHECK(mString_ToWideString(params, commandLine, mARRAYSIZE(commandLine)));

  const HINSTANCE result = ShellExecuteW(nullptr, TEXT("runas"), appName, commandLine, workingDir, SW_SHOWNORMAL);
  mERROR_IF((int32_t)((size_t)result) <= 32, mR_Failure);

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mProcess_Destroy_Internal, IN_OUT mProcess *pProcess)
{
  mFUNCTION_SETUP();

  mERROR_IF(pProcess == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mPipe_Destroy(&pProcess->_stdin));
  mERROR_CHECK(mPipe_Destroy(&pProcess->_stdout));
  mERROR_CHECK(mPipe_Destroy(&pProcess->_stderr));

  mRETURN_SUCCESS();
}
