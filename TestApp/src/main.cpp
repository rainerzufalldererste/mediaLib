#include "default.h"
#include "mMediaFileInputHandler.h"
#include "SDL.h"

mVec2s resulution;
SDL_Window *pWindow = nullptr;
uint32_t *pPixels = nullptr;

int main(int, char **)
{
  mFUNCTION_SETUP();

  resulution = mVec2s(1600, 900);
  pWindow = SDL_CreateWindow("HoloRoom Software Render", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, (int)resulution.x, (int)resulution.y, 0);
  pPixels = (uint32_t *)SDL_GetWindowSurface(pWindow)->pixels;

  const float_t width = 2.4f;
  const mVector frontWall_0(-width / 2, 0, width);
  const mVector frontWall_1(-width / 2, 0, 0);
  const mVector frontWall_2(width / 2, 0, width);
  const mVector frontWall_3(width / 2, 0, 0);

  const mMatrix rightWallMatrix = mMatrix::Translation(width / 2, 0, 0) * mMatrix::RotationZ(mDEG2RADf * -60) * mMatrix::Translation(width / 2, 0, 0);
  const mMatrix leftWallMatrix = mMatrix::Translation(-width / 2, 0, 0) * mMatrix::RotationZ(mDEG2RADf * 60) * mMatrix::Translation(-width / 2, 0, 0);

  const mVector rightWall_0 = frontWall_2;
  const mVector rightWall_1 = frontWall_3;
  const mVector rightWall_2 = frontWall_2.Transform3(rightWallMatrix);
  const mVector rightWall_3 = frontWall_3.Transform3(rightWallMatrix);

  const mVector leftWall_0 = frontWall_0.Transform3(leftWallMatrix);
  const mVector leftWall_1 = frontWall_1.Transform3(leftWallMatrix);
  const mVector leftWall_2 = frontWall_0;
  const mVector leftWall_3 = frontWall_1;

  const mVector floor_0 = leftWall_0;
  const mVector floor_1 = leftWall_1;
  const mVector floor_2 = rightWall_2;
  const mVector floor_3 = rightWall_3;

  const mMatrix vpMatrix = (mMatrix::LookToRH(mVector(0, -5.0f, 1.75f, 0), mVector(0, 1.0, 0, 0), mVector(0, 0, 1, 0)) * mMatrix::PerspectiveFovRH(mHALFPIf, resulution.x / (float_t)resulution.y, 1e-3f, 1e6f));

  mVector projectedPositions[] = { frontWall_0, frontWall_1, frontWall_2, frontWall_3, rightWall_2, rightWall_3, leftWall_0, leftWall_1 };

  for (size_t i = 0; i < mARRAYSIZE(projectedPositions); i++)
    projectedPositions[i] = projectedPositions[i].Transform3(vpMatrix);

  mVec2f projectedPositions2d[mARRAYSIZE(projectedPositions)];
  const mVec2f halfImageSize = mVec2f(resulution) / 2;

  for (size_t i = 0; i < mARRAYSIZE(projectedPositions); i++)
  {
    projectedPositions2d[i] = halfImageSize + (mVec2f)projectedPositions[i] * halfImageSize / (projectedPositions[i].z);
    projectedPositions2d[i].y = resulution.y - projectedPositions2d[i].y - 1.0f;
  }

  while (true)
  {
    mERROR_CHECK(mMemset(pPixels, resulution.x * resulution.y));

    for (size_t i = 0; i < mARRAYSIZE(projectedPositions); i++)
    {
      for (size_t j = i + 1; j < mARRAYSIZE(projectedPositions); j++)
      {
        for (float_t k = 0.0f; k < 1.0f; k += 1.0f / 1000.0f)
        {
          mVec2i pos = (mVec2i)(mVec2f)mVector::Lerp((mVector)projectedPositions2d[i], (mVector)projectedPositions2d[j], k);

          if (pos.x >= 0 && pos.x < (int64_t)resulution.x && pos.y >= 0 && pos.y < (int64_t)resulution.y)
            pPixels[pos.x + pos.y * resulution.x] = 0xFFFFFF;
        }
      }
    }

    SDL_UpdateWindowSurface(pWindow);

    SDL_Event sdl_event;
    while (SDL_PollEvent(&sdl_event))
      ; // We don't care.
  }

  mRETURN_SUCCESS();
}
