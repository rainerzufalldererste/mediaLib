#ifndef m2DAnimationHandler_h__
#define m2DAnimationHandler_h__

#include "mediaLib.h"
#include "mQueue.h"

enum m2DAH_KeyframeProperty
{
  m2DAH_KP_Invalid,
  m2DAH_KP_Position,
  m2DAH_KP_Size,
  m2DAH_KP_Rotation,
  m2DAH_KP_Colour,
  m2DAH_KP_CenterPointFactor,
};

struct m2DAH_Keyframe
{
  m2DAH_KeyframeProperty type = m2DAH_KP_Invalid;
  mVec4f targetValue;
  float_t animationPosition;

  float_t blendFactor;
  bool reverseBlending;

  float_t bezierPointA, bezierPointB, bezierPointC, bezierPointD;
  bool isBezierCurve;

  m2DAH_Keyframe();
  m2DAH_Keyframe(const m2DAH_KeyframeProperty type, const mVec4f &value, const float_t animationPosition, const float_t blendFactor);

  static m2DAH_Keyframe Position(const mVec2f value, const float_t animationPosition, const float_t blendFactor = 1);
  static m2DAH_Keyframe Size(const mVec2f value, const float_t animationPosition, const float_t blendFactor = 1);
  static m2DAH_Keyframe Rotation(const float_t value, const float_t animationPosition, const float_t blendFactor = 1);
  static m2DAH_Keyframe Colour(const mVec4f value, const float_t animationPosition, const float_t blendFactor = 1);
  static m2DAH_Keyframe CenterPointFactor(const mVec2f value, const float_t animationPosition, const float_t blendFactor = 1);
};

struct m2DAH_Sprite
{
  mString filename;
  mVec2f position, size;
  float_t rotation;
  mVec4f colour;

  struct
  {
    mVec2f position;
    mVec2f centerPointFactor;
  } _internal;

  mPtr<mQueue<m2DAH_Keyframe>> keyframes;
};

mFUNCTION(m2DAH_Sprite_Create, OUT m2DAH_Sprite *pSprite, const mString &filename);
mFUNCTION(m2DAH_Sprite_Update, IN m2DAH_Sprite *pSprite, const float_t animationPosition);

struct m2DAnimationHandler
{
  mPtr<mQueue<m2DAH_Sprite>> sprites;
  
  uint64_t lastUpdateNs;
  float_t timescaleFactor;
  float_t animationPosition;

  bool isPlaying;
};

mFUNCTION(m2DAnimationHandler_Create, OUT mPtr<m2DAnimationHandler> *pAnimationHandler, IN OPTIONAL mAllocator *pAllocator);
mFUNCTION(m2DAnimationHandler_CreateFromFile, OUT mPtr<m2DAnimationHandler> *pAnimationHandler, IN OPTIONAL mAllocator *pAllocator, const mString &filename);

mFUNCTION(m2DAnimationHandler_Destroy, IN_OUT mPtr<m2DAnimationHandler> *pAnimationHandler);

mFUNCTION(m2DAnimationHandler_Play, mPtr<m2DAnimationHandler> &animationHandler);
mFUNCTION(m2DAnimationHandler_Pause, mPtr<m2DAnimationHandler> &animationHandler);
mFUNCTION(m2DAnimationHandler_Stop, mPtr<m2DAnimationHandler> &animationHandler);

mFUNCTION(m2DAnimationHandler_Update, mPtr<m2DAnimationHandler> &animationHandler, const bool forceUpdate = false);

mFUNCTION(m2DAnimationHandler_SetTimescale, mPtr<m2DAnimationHandler> &animationHandler, const float_t timescale = 1.f);
mFUNCTION(m2DAnimationHandler_GetAnimationDuration, mPtr<m2DAnimationHandler> &animationHandler, OUT float_t *pDuration, OUT OPTIONAL float_t *pFirstKeyframe = nullptr, OUT OPTIONAL float_t *pLastKeyframe = nullptr);
mFUNCTION(m2DAnimationHandler_GetFirstKeyframeTimestamp, mPtr<m2DAnimationHandler> &animationHandler, OUT float_t *pTimestamp);
mFUNCTION(m2DAnimationHandler_GetLastKeyframeTimestamp, mPtr<m2DAnimationHandler> &animationHandler, OUT float_t *pTimestamp);

mFUNCTION(m2DAnimationHandler_SaveToFile, mPtr<m2DAnimationHandler> &animationHandler, const mString &filename);

#endif // m2DAnimationHandler_h__
