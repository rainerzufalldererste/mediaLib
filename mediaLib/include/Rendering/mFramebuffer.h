// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef mFramebuffer_h__
#define mFramebuffer_h__

#include "mRenderParams.h"
#include "mImageBuffer.h"

#if defined (mRENDERER_OPENGL)
extern GLuint mFrameBuffer_ActiveFrameBufferHandle;
#endif

struct mFramebuffer
{
  mVec2s size;
  size_t textureUnit;
#if defined(mRENDERER_OPENGL)
  GLuint frameBufferHandle;
  GLuint texColourBuffer;
  GLuint rboDepthStencil;
#endif
};

mFUNCTION(mFramebuffer_Create, OUT mPtr<mFramebuffer> *pFramebuffer, IN mAllocator *pAllocator, const mVec2s &size);
mFUNCTION(mFramebuffer_Destroy, IN_OUT mPtr<mFramebuffer> *pFramebuffer);
mFUNCTION(mFramebuffer_Bind, mPtr<mFramebuffer> &framebuffer);
mFUNCTION(mFramebuffer_Unbind);

mFUNCTION(mFramebuffer_SetResolution, mPtr<mFramebuffer> &framebuffer, const mVec2s &size);
mFUNCTION(mFramebuffer_Download, mPtr<mFramebuffer> &framebuffer, OUT mPtr<mImageBuffer> *pImageBuffer, IN mAllocator *pAllocator);

mFUNCTION(mTexture_Bind, mPtr<mFramebuffer> &framebuffer, const size_t textureUnit = 0);

#endif // mFramebuffer_h__
