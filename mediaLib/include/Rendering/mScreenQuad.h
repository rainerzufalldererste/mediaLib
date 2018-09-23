// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef mScreenQuad_h__
#define mScreenQuad_h__

#include "mRenderParams.h"
#include "mShader.h"

// mScreenQuad.
// Your fragment shader requires inputs for the amount of textures specified called `_texCoord%TextureIndex%` (`_texCoord0`, `_texCoord1`, ...).

struct mScreenQuad
{
  mPtr<mShader> shader;
#if defined(mRENDERER_OPENGL)
  GLuint vbo;
#endif
};

mFUNCTION(mScreenQuad_Create, OUT mPtr<mScreenQuad> *pScreenQuad, IN mAllocator *pAllocator, const std::string &fragmentShader, const size_t textureCount = 1);

// texture is called `_texture0`.
mFUNCTION(mScreenQuad_Create, OUT mPtr<mScreenQuad> *pScreenQuad, IN mAllocator *pAllocator);
mFUNCTION(mScreenQuad_CreateFrom, OUT mPtr<mScreenQuad> *pScreenQuad, IN mAllocator *pAllocator, const std::wstring &fragmentShaderPath, const size_t textureCount = 1);

mFUNCTION(mScreenQuad_Destroy, IN_OUT mPtr<mScreenQuad> *pScreenQuad);
mFUNCTION(mScreenQuad_Render, mPtr<mScreenQuad> &screenQuad);

#endif // mScreenQuad_h__
