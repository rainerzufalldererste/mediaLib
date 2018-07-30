// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "default.h"
#include <string>

#pragma comment(lib, "dxva2.lib")
#pragma comment(lib, "evr.lib")
#pragma comment(lib, "mf.lib")
#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfplay.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")

struct mVideoFileInputHandler;
typedef mFUNCTION(ProcessBufferFunction, IN uint8_t *pBuffer);

mFUNCTION(mVideoFileInputHandler_Create, OUT mPtr<mVideoFileInputHandler> *pPtr, const std::wstring &fileName);
mFUNCTION(mVideoFileInputHandler_Destroy, IN_OUT mPtr<mVideoFileInputHandler> *pPtr);
mFUNCTION(mVideoFileInputHandler_Play, mPtr<mVideoFileInputHandler> &ptr);
mFUNCTION(mVideoFileInputHandler_GetSize, mPtr<mVideoFileInputHandler> &ptr, OUT size_t *pSizeX, OUT size_t *pSizeY);
mFUNCTION(mVideoFileInputHandler_SetCallback, mPtr<mVideoFileInputHandler> &ptr, IN ProcessBufferFunction *pCallback);
