// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef mTexture_h__
#define mTexture_h__

#include "mRenderParams.h"
#include "mImageBuffer.h"

struct mTexture
{
  mVec2s resolution;
  mVec2f resolutionF;
  mRendererParams_UploadState uploadState = mRendererParams_UploadState::mRP_US_NotInitialized;
  mPtr<mImageBuffer> imageBuffer;

#if defined(mRENDERER_OPENGL)
  GLuint textureId;
#endif
};

mFUNCTION(mTexture_Create, OUT mTexture *pTexture, mPtr<mImageBuffer> &imageBuffer, const bool upload = true);
mFUNCTION(mTexture_Destroy, IN_OUT mTexture *pTexture);

mFUNCTION(mTexture_GetUploadState, mTexture &texture, OUT mRendererParams_UploadState *pUploadState);
mFUNCTION(mTexture_Upload, mTexture &texture);
mFUNCTION(mTexture_Bind, mTexture &texture);

#endif // mTexture_h__
