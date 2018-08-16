// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "mRenderParams.h"
#ifndef mShader_h__
#define mShader_h__

struct mShader
{
  bool initialized;
#if defined(mRENDERER_OPENGL)
  GLuint shaderProgram;
#endif
};

#if defined(mRENDERER_OPENGL)
#define mGLSL(src) "#version 150 core\n" #src
#endif

mFUNCTION(mShader_Create, OUT mShader *pShader, const std::string &vertexShader, const std::string &fragmentShader, IN OPTIONAL const char *fragDataLocation = nullptr);
mFUNCTION(mShader_CreateFromFile, OUT mShader *pShader, const std::wstring &filename, IN OPTIONAL const char *fragDataLocation = nullptr);
mFUNCTION(mShader_CreateFromFile, OUT mShader *pShader, const std::wstring &vertexShaderPath, const std::wstring &fragmentShaderPath, IN OPTIONAL const char *fragDataLocation = nullptr);
mFUNCTION(mShader_Destroy, IN_OUT mShader *pShader);

mFUNCTION(mShader_Bind, mShader &pShader);

#endif // mShader_h__
