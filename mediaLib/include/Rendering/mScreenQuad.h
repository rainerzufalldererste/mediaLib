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

mFUNCTION(mScreenQuad_Create, OUT mPtr<mScreenQuad> *pScreenQuad, IN mAllocator *pAllocator, const mString &fragmentShader, const size_t textureCount = 1);

// texture is called `_texture0`.
mFUNCTION(mScreenQuad_Create, OUT mPtr<mScreenQuad> *pScreenQuad, IN mAllocator *pAllocator);

// texture is called `_texture0`. sample count is called `_texture0sampleCount`.
mFUNCTION(mScreenQuad_CreateForMultisampleTexture, OUT mPtr<mScreenQuad> *pScreenQuad, IN mAllocator *pAllocator);
mFUNCTION(mScreenQuad_CreateFrom, OUT mPtr<mScreenQuad> *pScreenQuad, IN mAllocator *pAllocator, const mString &fragmentShaderPath, const size_t textureCount = 1);

mFUNCTION(mScreenQuad_Destroy, IN_OUT mPtr<mScreenQuad> *pScreenQuad);
mFUNCTION(mScreenQuad_Render, mPtr<mScreenQuad> &screenQuad);

#endif // mScreenQuad_h__
