#include "m2DAnimationHandler.h"

#include "mJson.h"

mFUNCTION(m2DAH_Keyframe_Serialize, IN const m2DAH_Keyframe *pKeyframe, mPtr<mJsonWriter> &jsonWriter);
mFUNCTION(m2DAH_Keyframe_Deserialize, OUT m2DAH_Keyframe *pKeyframe, mPtr<mJsonReader> &jsonReader);
float_t GetKeyframeInfluenceFactor(IN const m2DAH_Keyframe *pPrevious, IN const m2DAH_Keyframe *pNext, const float_t animationPosition);

const char m2DAH_Keyframe_Type_Position[] = "position";
const char m2DAH_Keyframe_Type_Size[] = "_sprite.size";
const char m2DAH_Keyframe_Type_Rotation[] = "rotation";
const char m2DAH_Keyframe_Type_Colour[] = "colour";
const char m2DAH_Keyframe_Type_CenterPointFactor[] = "centerPointFactor";
const char m2DAH_Keyframe_AnimationPosition[] = "animationPosition";
const char m2DAH_Keyframe_BlendFactor[] = "blendFactor";
const char m2DAH_Keyframe_ReverseBlending[] = "reverseBlending";
const char m2DAH_Keyframe_BezierPointA[] = "bezierPointA";
const char m2DAH_Keyframe_BezierPointB[] = "bezierPointB";
const char m2DAH_Keyframe_BezierPointC[] = "bezierPointC";
const char m2DAH_Keyframe_BezierPointD[] = "bezierPointD";

//////////////////////////////////////////////////////////////////////////

m2DAH_Keyframe::m2DAH_Keyframe()
{ }

m2DAH_Keyframe::m2DAH_Keyframe(const m2DAH_KeyframeProperty type, const mVec4f &value, const float_t animationPosition, const float_t blendFactor) :
  type(type),
  targetValue(value),
  animationPosition(animationPosition),
  blendFactor(blendFactor),
  reverseBlending(false),
  isBezierCurve(false),
  bezierPointA(1),
  bezierPointB(0),
  bezierPointC(1),
  bezierPointD(1)
{ }

m2DAH_Keyframe m2DAH_Keyframe::Position(const mVec2f value, const float_t animationPosition, const float_t blendFactor /* = 1 */)
{
  return m2DAH_Keyframe(m2DAH_KP_Position, mVec4f(value, mVec2f(0)), animationPosition, blendFactor);
}

m2DAH_Keyframe m2DAH_Keyframe::Size(const mVec2f value, const float_t animationPosition, const float_t blendFactor /* = 1 */)
{
  return m2DAH_Keyframe(m2DAH_KP_Size, mVec4f(value, mVec2f(0)), animationPosition, blendFactor);
}

m2DAH_Keyframe m2DAH_Keyframe::Rotation(const float_t value, const float_t animationPosition, const float_t blendFactor /* = 1 */)
{
  return m2DAH_Keyframe(m2DAH_KP_Rotation, mVec4f(value, mVec3f(0)), animationPosition, blendFactor);
}

m2DAH_Keyframe m2DAH_Keyframe::Colour(const mVec4f value, const float_t animationPosition, const float_t blendFactor /* = 1 */)
{
  return m2DAH_Keyframe(m2DAH_KP_Colour, value, animationPosition, blendFactor);
}

m2DAH_Keyframe m2DAH_Keyframe::CenterPointFactor(const mVec2f value, const float_t animationPosition, const float_t blendFactor)
{
  return m2DAH_Keyframe(m2DAH_KP_CenterPointFactor, mVec4f(value, mVec2f(0)), animationPosition, blendFactor);
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(m2DAH_Keyframe_Serialize, IN const m2DAH_Keyframe *pKeyframe, mPtr<mJsonWriter> &jsonWriter)
{
  mFUNCTION_SETUP();

  mERROR_IF(pKeyframe == nullptr || jsonWriter == nullptr, mR_ArgumentNull);

  switch (pKeyframe->type)
  {
  case m2DAH_KP_Position:
    mERROR_CHECK(mJsonWriter_AddValueX(jsonWriter, m2DAH_Keyframe_Type_Position, pKeyframe->targetValue.ToVector2()));
    break;

  case m2DAH_KP_Size:
    mERROR_CHECK(mJsonWriter_AddValueX(jsonWriter, m2DAH_Keyframe_Type_Size, pKeyframe->targetValue.ToVector2()));
    break;

  case m2DAH_KP_Rotation:
    mERROR_CHECK(mJsonWriter_AddValue(jsonWriter, m2DAH_Keyframe_Type_Rotation, pKeyframe->targetValue.x));
    break;

  case m2DAH_KP_Colour:
    mERROR_CHECK(mJsonWriter_AddValueX(jsonWriter, m2DAH_Keyframe_Type_Colour, pKeyframe->targetValue));
    break;

  case m2DAH_KP_CenterPointFactor:
    mERROR_CHECK(mJsonWriter_AddValueX(jsonWriter, m2DAH_Keyframe_Type_CenterPointFactor, pKeyframe->targetValue.ToVector2()));
    break;

  default:
    mRETURN_RESULT(mR_InvalidParameter);
  }

  mERROR_CHECK(mJsonWriter_AddValue(jsonWriter, m2DAH_Keyframe_AnimationPosition, pKeyframe->animationPosition));

  if (pKeyframe->isBezierCurve)
  {
    mERROR_CHECK(mJsonWriter_AddValue(jsonWriter, m2DAH_Keyframe_BezierPointA, pKeyframe->bezierPointA));
    mERROR_CHECK(mJsonWriter_AddValue(jsonWriter, m2DAH_Keyframe_BezierPointB, pKeyframe->bezierPointB));
    mERROR_CHECK(mJsonWriter_AddValue(jsonWriter, m2DAH_Keyframe_BezierPointC, pKeyframe->bezierPointC));
    mERROR_CHECK(mJsonWriter_AddValue(jsonWriter, m2DAH_Keyframe_BezierPointD, pKeyframe->bezierPointD));
  }
  else
  {
    mERROR_CHECK(mJsonWriter_AddValue(jsonWriter, m2DAH_Keyframe_BlendFactor, pKeyframe->blendFactor));
    mERROR_CHECK(mJsonWriter_AddValue(jsonWriter, m2DAH_Keyframe_ReverseBlending, pKeyframe->reverseBlending));
  }

  mRETURN_SUCCESS();
}

mFUNCTION(m2DAH_Keyframe_Deserialize, OUT m2DAH_Keyframe *pKeyframe, mPtr<mJsonReader> &jsonReader)
{
  mFUNCTION_SETUP();

  mERROR_IF(pKeyframe == nullptr || jsonReader == nullptr, mR_ArgumentNull);

  double_t valueFloat = 0;
  double_t valueFloat2 = 0;
  double_t valueFloat3 = 0;
  double_t valueFloat4 = 0;
  mVec2f valueVec2;

  if (mSUCCEEDED(mSILENCE_ERROR(mJsonReader_ReadNamedValue(jsonReader, m2DAH_Keyframe_Type_Position, &valueVec2))))
  {
    pKeyframe->type = m2DAH_KP_Position;
    pKeyframe->targetValue = mVec4f(valueVec2, mVec2f(0));
  }
  else if (mSUCCEEDED(mSILENCE_ERROR(mJsonReader_ReadNamedValue(jsonReader, m2DAH_Keyframe_Type_Size, &valueVec2))))
  {
    pKeyframe->type = m2DAH_KP_Size;
    pKeyframe->targetValue = mVec4f(valueVec2, mVec2f(0));
  }
  else if (mSUCCEEDED(mSILENCE_ERROR(mJsonReader_ReadNamedValue(jsonReader, m2DAH_Keyframe_Type_Rotation, &valueFloat))))
  {
    pKeyframe->type = m2DAH_KP_Rotation;
    pKeyframe->targetValue = mVec4f((float_t)valueFloat, mVec3f(0));
  }
  else if (mSUCCEEDED(mSILENCE_ERROR(mJsonReader_ReadNamedValue(jsonReader, m2DAH_Keyframe_Type_Colour, &pKeyframe->targetValue))))
  {
    pKeyframe->type = m2DAH_KP_Colour;
  }
  else if (mSUCCEEDED(mSILENCE_ERROR(mJsonReader_ReadNamedValue(jsonReader, m2DAH_Keyframe_Type_CenterPointFactor, &valueVec2))))
  {
    pKeyframe->type = m2DAH_KP_CenterPointFactor;
    pKeyframe->targetValue = mVec4f(valueVec2, mVec2f(0));
  }
  else
  {
    mRETURN_RESULT(mR_NotSupported);
  }

  mERROR_CHECK(mJsonReader_ReadNamedValue(jsonReader, m2DAH_Keyframe_AnimationPosition, &valueFloat));
  pKeyframe->animationPosition = (float_t)valueFloat;

  pKeyframe->bezierPointA = 0;
  pKeyframe->bezierPointB = 0;
  pKeyframe->bezierPointC = 0;
  pKeyframe->bezierPointD = 0;
  pKeyframe->blendFactor = 1.f;

  if (mSUCCEEDED(mSILENCE_ERROR(mJsonReader_ReadNamedValue(jsonReader, m2DAH_Keyframe_BlendFactor, &valueFloat))))
  {
    pKeyframe->isBezierCurve = false;
    pKeyframe->blendFactor = (float_t)valueFloat;

    pKeyframe->reverseBlending = false;
    mSILENCE_ERROR(mJsonReader_ReadNamedValue(jsonReader, m2DAH_Keyframe_ReverseBlending, &pKeyframe->reverseBlending));
  }
  else if (mSUCCEEDED(mSILENCE_ERROR(mJsonReader_ReadNamedValue(jsonReader, m2DAH_Keyframe_BezierPointA, &valueFloat)))
    && mSUCCEEDED(mSILENCE_ERROR(mJsonReader_ReadNamedValue(jsonReader, m2DAH_Keyframe_BezierPointB, &valueFloat2)))
    && mSUCCEEDED(mSILENCE_ERROR(mJsonReader_ReadNamedValue(jsonReader, m2DAH_Keyframe_BezierPointC, &valueFloat3)))
    && mSUCCEEDED(mSILENCE_ERROR(mJsonReader_ReadNamedValue(jsonReader, m2DAH_Keyframe_BezierPointD, &valueFloat4))))
  {
    pKeyframe->bezierPointA = (float_t)valueFloat;
    pKeyframe->bezierPointB = (float_t)valueFloat2;
    pKeyframe->bezierPointC = (float_t)valueFloat3;
    pKeyframe->bezierPointD = (float_t)valueFloat4;
    pKeyframe->isBezierCurve = true;
  }
  else
  {
    mRETURN_RESULT(mR_NotSupported);
  }

  mRETURN_SUCCESS();
}

float_t GetKeyframeInfluenceFactor(IN const m2DAH_Keyframe *pPrevious, IN const m2DAH_Keyframe *pNext, const float_t animationPosition)
{
  const float_t blendFactor = (animationPosition - pPrevious->animationPosition) / (pNext->animationPosition - pPrevious->animationPosition);

  if (pNext->isBezierCurve)
  {
    const float_t bezierPointA = mLerp(0.f, pNext->bezierPointA, mPow(blendFactor, pNext->bezierPointC));
    const float_t bezierPointB = mLerp(pNext->bezierPointB, 1.f, 1 - mPow(1 - blendFactor, pNext->bezierPointD));
    
    return mLerp(bezierPointA, bezierPointB, blendFactor);
  }
  else
  {
    if (pNext->reverseBlending)
      return 1 - mPow(1 - blendFactor, pNext->blendFactor);
    else
      return mPow(blendFactor, pNext->blendFactor);
  }
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(m2DAH_Sprite_Serialize, IN const m2DAH_Sprite *pSprite, mPtr<mJsonWriter> &jsonWriter);
mFUNCTION(m2DAH_Sprite_Deserialize, OUT m2DAH_Sprite *pSprite, mPtr<mJsonReader> &jsonReader);
mFUNCTION(m2DAH_Sprite_Update, IN m2DAH_Sprite *pSprite, const float_t animationPosition);

const char m2DAH_Sprite_Keyframes[] = "keyframes";
const char m2DAH_Sprite_Filename[] = "filename";

//////////////////////////////////////////////////////////////////////////

mFUNCTION(m2DAH_Sprite_Create, OUT m2DAH_Sprite *pSprite, const mString &filename)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSprite == nullptr, mR_ArgumentNull);
  mERROR_IF(filename.hasFailed || filename.bytes <= 1, mR_InvalidParameter);

  mERROR_CHECK(mQueue_Create(&pSprite->keyframes, &mDefaultAllocator));
  
  pSprite->filename = filename;
  pSprite->position = pSprite->size = mVec2f(0);
  pSprite->rotation = 0;
  pSprite->colour = mVec4f(0);

  mRETURN_SUCCESS();
}

mFUNCTION(m2DAH_Sprite_Update, IN m2DAH_Sprite *pSprite, const float_t animationPosition)
{
  mFUNCTION_SETUP();

  bool hasPositionKeyframe = false, hasSizeKeyframe = false, hasRotationKeyframe = false, hasColourKeyframe = false, hasCenterPointKeyframe = false;
  const m2DAH_Keyframe *pPrevPositionKeyframe = nullptr, *pNextPositionKeyframe = nullptr, *pPrevSizeKeyframe = nullptr, *pNextSizeKeyframe = nullptr, *pPrevRotationKeyframe = nullptr, *pNextRotationKeyframe = nullptr, *pPrevColourKeyframe = nullptr, *pNextColourKeyframe = nullptr, *pPrevCenterPointKeyframe = nullptr, *pNextCenterPointKeyframe = nullptr;

  // Find keyframes.
  {
    for (const auto &_keyframe : pSprite->keyframes->Iterate())
    {
      switch (_keyframe.type)
      {
      case m2DAH_KP_Position:
        hasPositionKeyframe = true;

        if (_keyframe.animationPosition <= animationPosition && (pPrevPositionKeyframe == nullptr || pPrevPositionKeyframe->animationPosition < _keyframe.animationPosition))
          pPrevPositionKeyframe = &_keyframe;
        else if (_keyframe.animationPosition > animationPosition && (pNextPositionKeyframe == nullptr || pNextPositionKeyframe->animationPosition > _keyframe.animationPosition))
          pNextPositionKeyframe = &_keyframe;
        break;

      case m2DAH_KP_Size:
        hasSizeKeyframe = true;

        if (_keyframe.animationPosition <= animationPosition && (pPrevSizeKeyframe == nullptr || pPrevSizeKeyframe->animationPosition < _keyframe.animationPosition))
          pPrevSizeKeyframe = &_keyframe;
        else if (_keyframe.animationPosition > animationPosition && (pNextSizeKeyframe == nullptr || pNextSizeKeyframe->animationPosition > _keyframe.animationPosition))
          pNextSizeKeyframe = &_keyframe;
        break;

      case m2DAH_KP_Rotation:
        hasRotationKeyframe = true;

        if (_keyframe.animationPosition <= animationPosition && (pPrevRotationKeyframe == nullptr || pPrevRotationKeyframe->animationPosition < _keyframe.animationPosition))
          pPrevRotationKeyframe = &_keyframe;
        else if (_keyframe.animationPosition > animationPosition && (pNextRotationKeyframe == nullptr || pNextRotationKeyframe->animationPosition > _keyframe.animationPosition))
          pNextRotationKeyframe = &_keyframe;
        break;

      case m2DAH_KP_Colour:
        hasColourKeyframe = true;

        if (_keyframe.animationPosition <= animationPosition && (pPrevColourKeyframe == nullptr || pPrevColourKeyframe->animationPosition < _keyframe.animationPosition))
          pPrevColourKeyframe = &_keyframe;
        else if (_keyframe.animationPosition > animationPosition && (pNextColourKeyframe == nullptr || pNextColourKeyframe->animationPosition > _keyframe.animationPosition))
          pNextColourKeyframe = &_keyframe;
        break;

      case m2DAH_KP_CenterPointFactor:
        hasCenterPointKeyframe = true;

        if (_keyframe.animationPosition <= animationPosition && (pPrevCenterPointKeyframe == nullptr || pPrevCenterPointKeyframe->animationPosition < _keyframe.animationPosition))
          pPrevCenterPointKeyframe = &_keyframe;
        else if (_keyframe.animationPosition > animationPosition && (pNextCenterPointKeyframe == nullptr || pNextCenterPointKeyframe->animationPosition > _keyframe.animationPosition))
          pNextCenterPointKeyframe = &_keyframe;
        break;

      default:
        break;
      }
    }
  }

  // Interpret keyframes.
  {
    pSprite->_internal.centerPointFactor = mVec2f(.5f);

    if (hasPositionKeyframe)
    {
      if (pNextPositionKeyframe == nullptr && pPrevPositionKeyframe != nullptr)
        std::swap(pNextPositionKeyframe, pPrevPositionKeyframe);

      if (pNextPositionKeyframe != nullptr && pPrevPositionKeyframe == nullptr)
        pSprite->position = pNextPositionKeyframe->targetValue.ToVector2();
      else if (pNextPositionKeyframe != nullptr && pPrevPositionKeyframe != nullptr)
        pSprite->position = mLerp(pPrevPositionKeyframe->targetValue.ToVector2(), pNextPositionKeyframe->targetValue.ToVector2(), GetKeyframeInfluenceFactor(pPrevPositionKeyframe, pNextPositionKeyframe, animationPosition));
    }

    if (hasSizeKeyframe)
    {
      if (pNextSizeKeyframe == nullptr && pPrevSizeKeyframe != nullptr)
        std::swap(pNextSizeKeyframe, pPrevSizeKeyframe);

      if (pNextSizeKeyframe != nullptr && pPrevSizeKeyframe == nullptr)
        pSprite->size = pNextSizeKeyframe->targetValue.ToVector2();
      else if (pNextSizeKeyframe != nullptr && pPrevSizeKeyframe != nullptr)
        pSprite->size = mLerp(pPrevSizeKeyframe->targetValue.ToVector2(), pNextSizeKeyframe->targetValue.ToVector2(), GetKeyframeInfluenceFactor(pPrevSizeKeyframe, pNextSizeKeyframe, animationPosition));
    }

    if (hasRotationKeyframe)
    {
      if (pNextRotationKeyframe == nullptr && pPrevRotationKeyframe != nullptr)
        std::swap(pNextRotationKeyframe, pPrevRotationKeyframe);

      if (pNextRotationKeyframe != nullptr && pPrevRotationKeyframe == nullptr)
        pSprite->rotation = pNextRotationKeyframe->targetValue.x;
      else if (pNextRotationKeyframe != nullptr && pPrevRotationKeyframe != nullptr)
        pSprite->rotation = mLerp(pPrevRotationKeyframe->targetValue.x, pNextRotationKeyframe->targetValue.x, GetKeyframeInfluenceFactor(pPrevRotationKeyframe, pNextRotationKeyframe, animationPosition));
    }

    if (hasColourKeyframe)
    {
      if (pNextColourKeyframe == nullptr && pPrevColourKeyframe != nullptr)
        std::swap(pNextColourKeyframe, pPrevColourKeyframe);

      if (pNextColourKeyframe != nullptr && pPrevColourKeyframe == nullptr)
        pSprite->colour = pNextColourKeyframe->targetValue;
      else if (pNextColourKeyframe != nullptr && pPrevColourKeyframe != nullptr)
        pSprite->colour = mLerp(pPrevColourKeyframe->targetValue, pNextColourKeyframe->targetValue, GetKeyframeInfluenceFactor(pPrevColourKeyframe, pNextColourKeyframe, animationPosition));
    }

    if (hasCenterPointKeyframe)
    {
      if (pNextCenterPointKeyframe == nullptr && pPrevCenterPointKeyframe != nullptr)
        std::swap(pNextCenterPointKeyframe, pPrevCenterPointKeyframe);

      if (pNextCenterPointKeyframe != nullptr && pPrevCenterPointKeyframe == nullptr)
        pSprite->_internal.centerPointFactor = pNextCenterPointKeyframe->targetValue.ToVector2();
      else if (pNextCenterPointKeyframe != nullptr && pPrevCenterPointKeyframe != nullptr)
        pSprite->_internal.centerPointFactor = mLerp(pPrevCenterPointKeyframe->targetValue.ToVector2(), pNextCenterPointKeyframe->targetValue.ToVector2(), GetKeyframeInfluenceFactor(pPrevCenterPointKeyframe, pNextCenterPointKeyframe, animationPosition));
    }

    pSprite->_internal.position = pSprite->position;

    const mVec2f rotatedOffset = pSprite->size * pSprite->_internal.centerPointFactor;

    pSprite->position -= rotatedOffset;
  }

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(m2DAH_Sprite_Serialize, IN const m2DAH_Sprite *pSprite, mPtr<mJsonWriter> &jsonWriter)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSprite == nullptr || jsonWriter == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mJsonWriter_AddValue(jsonWriter, m2DAH_Sprite_Filename, pSprite->filename));

  // Keyframes.
  {
    mERROR_CHECK(mJsonWriter_BeginArray(jsonWriter, m2DAH_Sprite_Keyframes));
    mDEFER(mJsonWriter_EndArray(jsonWriter));

    for (const auto &_keyframe : pSprite->keyframes->Iterate())
    {
      mERROR_CHECK(mJsonWriter_BeginUnnamed(jsonWriter));
      mDEFER(mJsonWriter_EndUnnamed(jsonWriter));

      mERROR_CHECK(m2DAH_Keyframe_Serialize(&_keyframe, jsonWriter));
    }
  }

  mRETURN_SUCCESS();
}

mFUNCTION(m2DAH_Sprite_Deserialize, OUT m2DAH_Sprite *pSprite, mPtr<mJsonReader> &jsonReader)
{
  mFUNCTION_SETUP();

  mERROR_IF(pSprite == nullptr || jsonReader == nullptr, mR_ArgumentNull);

  mString filename;
  mERROR_CHECK(mJsonReader_ReadNamedValue(jsonReader, m2DAH_Sprite_Filename, &filename));

  mERROR_CHECK(m2DAH_Sprite_Create(pSprite, filename));

  // Keyframes.
  {
    mERROR_CHECK(mJsonReader_StepIntoArray(jsonReader, m2DAH_Sprite_Keyframes));
    mDEFER(mJsonReader_ExitArray(jsonReader));

    const auto &keyframeFunc = [&](mPtr<mJsonReader> &_jsonReader, const size_t /* index */) -> mResult 
    {
      mFUNCTION_SETUP();

      m2DAH_Keyframe keyframe = { };
      mERROR_CHECK(m2DAH_Keyframe_Deserialize(&keyframe, _jsonReader));

      mERROR_CHECK(mQueue_PushBack(pSprite->keyframes, std::move(keyframe)));

      mRETURN_SUCCESS();
    };

    mERROR_CHECK(mJsonReader_ArrayForEach(jsonReader, keyframeFunc));
  }

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(m2DAnimationHandler_Destroy_Internal, IN_OUT m2DAnimationHandler *pAnimationHandler);

const char m2DAnimationHandler_Sprites[] = "sprites";

//////////////////////////////////////////////////////////////////////////

mFUNCTION(m2DAnimationHandler_Create, OUT mPtr<m2DAnimationHandler> *pAnimationHandler, IN OPTIONAL mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAnimationHandler == nullptr, mR_ArgumentNull);

  mDEFER_CALL_ON_ERROR(pAnimationHandler, mSharedPointer_Destroy);
  mERROR_CHECK(mSharedPointer_Allocate(pAnimationHandler, pAllocator));

  mERROR_CHECK(mQueue_Create(&(*pAnimationHandler)->sprites, pAllocator));

  (*pAnimationHandler)->lastUpdateNs = mGetCurrentTimeNs();
  (*pAnimationHandler)->timescaleFactor = 1.f;

  mRETURN_SUCCESS();
}

mFUNCTION(m2DAnimationHandler_CreateFromFile, OUT mPtr<m2DAnimationHandler> *pAnimationHandler, IN OPTIONAL mAllocator *pAllocator, const mString &filename)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(m2DAnimationHandler_Create(pAnimationHandler, pAllocator));

  mPtr<mJsonReader> jsonReader;
  mDEFER_CALL(&jsonReader, mJsonReader_Destroy);
  mERROR_CHECK(mJsonReader_CreateFromFile(&jsonReader, pAllocator, filename));

  mERROR_CHECK(mJsonReader_StepIntoArray(jsonReader, m2DAnimationHandler_Sprites));

  const auto &spriteFunc = [&](mPtr<mJsonReader> &_jsonReader, const size_t /* index */) -> mResult 
  {
    mFUNCTION_SETUP();

    m2DAH_Sprite sprite;
    mERROR_CHECK(m2DAH_Sprite_Deserialize(&sprite, _jsonReader));

    mERROR_CHECK(mQueue_PushBack((*pAnimationHandler)->sprites, std::move(sprite)));

    mRETURN_SUCCESS();
  };

  mERROR_CHECK(mJsonReader_ArrayForEach(jsonReader, spriteFunc));

  mRETURN_SUCCESS();
}

mFUNCTION(m2DAnimationHandler_Destroy, IN_OUT mPtr<m2DAnimationHandler> *pAnimationHandler)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAnimationHandler == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Destroy(pAnimationHandler));

  mRETURN_SUCCESS();
}

mFUNCTION(m2DAnimationHandler_Play, mPtr<m2DAnimationHandler> &animationHandler)
{
  mFUNCTION_SETUP();

  mERROR_IF(animationHandler == nullptr, mR_ArgumentNull);
  mERROR_IF(animationHandler->isPlaying, mR_Success);

  animationHandler->isPlaying = true;
  animationHandler->lastUpdateNs = mGetCurrentTimeNs();

  mRETURN_SUCCESS();
}

mFUNCTION(m2DAnimationHandler_Pause, mPtr<m2DAnimationHandler> &animationHandler)
{
  mFUNCTION_SETUP();

  mERROR_IF(animationHandler == nullptr, mR_ArgumentNull);

  animationHandler->isPlaying = false;

  mRETURN_SUCCESS();
}

mFUNCTION(m2DAnimationHandler_Stop, mPtr<m2DAnimationHandler>& animationHandler)
{
  mFUNCTION_SETUP();

  mERROR_IF(animationHandler == nullptr, mR_ArgumentNull);

  animationHandler->isPlaying = false;
  animationHandler->animationPosition = 0;

  mRETURN_SUCCESS();
}

mFUNCTION(m2DAnimationHandler_Update, mPtr<m2DAnimationHandler> &animationHandler, const bool forceUpdate /* = false */)
{
  mFUNCTION_SETUP();

  mERROR_IF(animationHandler == nullptr, mR_ArgumentNull);

  if (!animationHandler->isPlaying)
  {
    if (!forceUpdate)
      mRETURN_SUCCESS();
  }
  else
  {
    const uint64_t updateTimestampNs = mGetCurrentTimeNs();
    const uint64_t updateStepNs = updateTimestampNs - animationHandler->lastUpdateNs;

    animationHandler->lastUpdateNs = updateTimestampNs;
    animationHandler->animationPosition += (updateStepNs * (1.f / 1e+6f)) * animationHandler->timescaleFactor;
  }

  for (auto &_sprite : animationHandler->sprites->Iterate())
    mERROR_CHECK(m2DAH_Sprite_Update(&_sprite, animationHandler->animationPosition));

  mRETURN_SUCCESS();
}

mFUNCTION(m2DAnimationHandler_SetTimescale, mPtr<m2DAnimationHandler> &animationHandler, const float_t timescale)
{
  mFUNCTION_SETUP();

  mERROR_IF(animationHandler == nullptr, mR_ArgumentNull);
  mERROR_IF(timescale == 0, mR_InvalidParameter);

  animationHandler->timescaleFactor = timescale;

  mRETURN_SUCCESS();
}

mFUNCTION(m2DAnimationHandler_GetAnimationDuration, mPtr<m2DAnimationHandler> &animationHandler, OUT float_t *pDuration, OUT OPTIONAL float_t *pFirstKeyframe /* = nullptr */, OUT OPTIONAL float_t *pLastKeyframe /* = nullptr */)
{
  mFUNCTION_SETUP();

  mERROR_IF(animationHandler == nullptr || pDuration == nullptr, mR_ArgumentNull);

  float_t firstAnimationPosition = 0, lastAnimationPosition = 0;

  for (const auto &_sprite : animationHandler->sprites->Iterate())
  {
    for (const auto &_keyframe : _sprite.keyframes->Iterate())
    {
      if (_keyframe.animationPosition < firstAnimationPosition)
        firstAnimationPosition = _keyframe.animationPosition;

      if (_keyframe.animationPosition > lastAnimationPosition)
        lastAnimationPosition = _keyframe.animationPosition;
    }
  }

  *pDuration = lastAnimationPosition - firstAnimationPosition;

  if (pFirstKeyframe != nullptr)
    *pFirstKeyframe = firstAnimationPosition;

  if (pLastKeyframe != nullptr)
    *pLastKeyframe = lastAnimationPosition;

  mRETURN_SUCCESS();
}

mFUNCTION(m2DAnimationHandler_GetFirstKeyframeTimestamp, mPtr<m2DAnimationHandler> &animationHandler, OUT float_t *pTimestamp)
{
  mFUNCTION_SETUP();

  mERROR_IF(animationHandler == nullptr || pTimestamp == nullptr, mR_ArgumentNull);

  float_t firstAnimationPosition = 0;

  for (const auto &_sprite : animationHandler->sprites->Iterate())
  {
    for (const auto &_keyframe : _sprite.keyframes->Iterate())
    {
      if (_keyframe.animationPosition < firstAnimationPosition)
        firstAnimationPosition = _keyframe.animationPosition;
    }
  }

  *pTimestamp = firstAnimationPosition;

  mRETURN_SUCCESS();
}

mFUNCTION(m2DAnimationHandler_GetLastKeyframeTimestamp, mPtr<m2DAnimationHandler> &animationHandler, OUT float_t *pTimestamp)
{
  mFUNCTION_SETUP();

  mERROR_IF(animationHandler == nullptr || pTimestamp == nullptr, mR_ArgumentNull);

  float_t lastAnimationPosition = 0;

  for (const auto &_sprite : animationHandler->sprites->Iterate())
  {
    for (const auto &_keyframe : _sprite.keyframes->Iterate())
    {
      if (_keyframe.animationPosition > lastAnimationPosition)
        lastAnimationPosition = _keyframe.animationPosition;
    }
  }

  *pTimestamp = lastAnimationPosition;

  mRETURN_SUCCESS();
}


mFUNCTION(m2DAnimationHandler_SaveToFile, mPtr<m2DAnimationHandler> &animationHandler, const mString &filename)
{
  mFUNCTION_SETUP();

  mERROR_IF(animationHandler == nullptr, mR_ArgumentNull);

  mPtr<mJsonWriter> jsonWriter;
  mDEFER_CALL(&jsonWriter, mJsonWriter_Destroy);
  mERROR_CHECK(mJsonWriter_Create(&jsonWriter, &mDefaultTempAllocator));

  // Sprites.
  {
    mERROR_CHECK(mJsonWriter_BeginArray(jsonWriter, m2DAnimationHandler_Sprites));
    mDEFER(mJsonWriter_EndArray(jsonWriter));

    for (const auto &_sprite : animationHandler->sprites->Iterate())
    {
      mERROR_CHECK(mJsonWriter_BeginUnnamed(jsonWriter));
      mDEFER(mJsonWriter_EndUnnamed(jsonWriter));

      mERROR_CHECK(m2DAH_Sprite_Serialize(&_sprite, jsonWriter));
    }
  }

  mERROR_CHECK(mJsonWriter_ToFile(jsonWriter, filename));

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(m2DAnimationHandler_Destroy_Internal, IN_OUT m2DAnimationHandler *pAnimationHandler)
{
  mFUNCTION_SETUP();

  mERROR_IF(pAnimationHandler == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Destroy(&pAnimationHandler->sprites));

  mRETURN_SUCCESS();
}
