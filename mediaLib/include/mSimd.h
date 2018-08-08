// Copyright 2018 Christoph Stiller
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef mSimd_h__
#define mSimd_h__

#include "default.h"
#include <intrin.h>

#if defined(SSE) || defined(SSE2)
#include <xmmintrin.h>
#endif

#if defined(SSE2)
#include <emmintrin.h>
#endif

/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2018 Yermalayeu Ihar,
*               2014-2015 Antonenka Mikhail.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

template <class T> mINLINE char GetChar(T value, const size_t index)
{
  return ((char*)&value)[index];
}

#define SIMD_AS_CHAR(a) char(a)

#define SIMD_AS_2CHARS(a) \
	GetChar(int16_t(a), 0), GetChar(int16_t(a), 1)

#define SIMD_AS_4CHARS(a) \
	GetChar(int32_t(a), 0), GetChar(int32_t(a), 1), \
	GetChar(int32_t(a), 2), GetChar(int32_t(a), 3)

#define SIMD_AS_8CHARS(a) \
	GetChar(int64_t(a), 0), GetChar(int64_t(a), 1), \
	GetChar(int64_t(a), 2), GetChar(int64_t(a), 3), \
	GetChar(int64_t(a), 4), GetChar(int64_t(a), 5), \
	GetChar(int64_t(a), 6), GetChar(int64_t(a), 7)

#define SIMD_MM_SET1_EPI8(a) \
    {SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), \
    SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), \
    SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), \
    SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a)}

#define SIMD_MM_SET2_EPI8(a0, a1) \
    {SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), \
    SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), \
    SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), \
    SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1)}

#define SIMD_MM_SETR_EPI8(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af) \
    {SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a2), SIMD_AS_CHAR(a3), \
    SIMD_AS_CHAR(a4), SIMD_AS_CHAR(a5), SIMD_AS_CHAR(a6), SIMD_AS_CHAR(a7), \
    SIMD_AS_CHAR(a8), SIMD_AS_CHAR(a9), SIMD_AS_CHAR(aa), SIMD_AS_CHAR(ab), \
    SIMD_AS_CHAR(ac), SIMD_AS_CHAR(ad), SIMD_AS_CHAR(ae), SIMD_AS_CHAR(af)}

#define SIMD_MM_SET1_EPI16(a) \
    {SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), \
    SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a)}

#define SIMD_MM_SET2_EPI16(a0, a1) \
    {SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1), SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1), \
    SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1), SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1)}

#define SIMD_MM_SETR_EPI16(a0, a1, a2, a3, a4, a5, a6, a7) \
    {SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1), SIMD_AS_2CHARS(a2), SIMD_AS_2CHARS(a3), \
    SIMD_AS_2CHARS(a4), SIMD_AS_2CHARS(a5), SIMD_AS_2CHARS(a6), SIMD_AS_2CHARS(a7)}

#define SIMD_MM_SET1_EPI32(a) \
    {SIMD_AS_4CHARS(a), SIMD_AS_4CHARS(a), SIMD_AS_4CHARS(a), SIMD_AS_4CHARS(a)}

#define SIMD_MM_SET2_EPI32(a0, a1) \
    {SIMD_AS_4CHARS(a0), SIMD_AS_4CHARS(a1), SIMD_AS_4CHARS(a0), SIMD_AS_4CHARS(a1)}

#define SIMD_MM_SETR_EPI32(a0, a1, a2, a3) \
    {SIMD_AS_4CHARS(a0), SIMD_AS_4CHARS(a1), SIMD_AS_4CHARS(a2), SIMD_AS_4CHARS(a3)}

#define SIMD_MM_SET1_EPI64(a) \
    {SIMD_AS_8CHARS(a), SIMD_AS_8CHARS(a)}

#define SIMD_MM_SET2_EPI64(a0, a1) \
    {SIMD_AS_8CHARS(a0), SIMD_AS_8CHARS(a1)}

#define SIMD_MM_SETR_EPI64(a0, a1) \
    {SIMD_AS_8CHARS(a0), SIMD_AS_8CHARS(a1)}
#define SIMD_MM256_SET1_EPI8(a) \
	{SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), \
	SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), \
	SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), \
	SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), \
	SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), \
	SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), \
	SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), \
	SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a), SIMD_AS_CHAR(a)}

#define SIMD_MM256_SET2_EPI8(a0, a1) \
	{SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), \
	SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), \
	SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), \
	SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), \
	SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), \
	SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), \
	SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), \
	SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1)}

#define SIMD_MM256_SETR_EPI8(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, ba, bb, bc, bd, be, bf) \
    {SIMD_AS_CHAR(a0), SIMD_AS_CHAR(a1), SIMD_AS_CHAR(a2), SIMD_AS_CHAR(a3), \
    SIMD_AS_CHAR(a4), SIMD_AS_CHAR(a5), SIMD_AS_CHAR(a6), SIMD_AS_CHAR(a7), \
    SIMD_AS_CHAR(a8), SIMD_AS_CHAR(a9), SIMD_AS_CHAR(aa), SIMD_AS_CHAR(ab), \
    SIMD_AS_CHAR(ac), SIMD_AS_CHAR(ad), SIMD_AS_CHAR(ae), SIMD_AS_CHAR(af), \
    SIMD_AS_CHAR(b0), SIMD_AS_CHAR(b1), SIMD_AS_CHAR(b2), SIMD_AS_CHAR(b3), \
    SIMD_AS_CHAR(b4), SIMD_AS_CHAR(b5), SIMD_AS_CHAR(b6), SIMD_AS_CHAR(b7), \
    SIMD_AS_CHAR(b8), SIMD_AS_CHAR(b9), SIMD_AS_CHAR(ba), SIMD_AS_CHAR(bb), \
    SIMD_AS_CHAR(bc), SIMD_AS_CHAR(bd), SIMD_AS_CHAR(be), SIMD_AS_CHAR(bf)}

#define SIMD_MM256_SET1_EPI16(a) \
	{SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), \
	SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), \
	SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), \
	SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a), SIMD_AS_2CHARS(a)}

#define SIMD_MM256_SET2_EPI16(a0, a1) \
	{SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1), SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1), \
	SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1), SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1), \
	SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1), SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1), \
	SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1), SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1)}

#define SIMD_MM256_SETR_EPI16(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af) \
    {SIMD_AS_2CHARS(a0), SIMD_AS_2CHARS(a1), SIMD_AS_2CHARS(a2), SIMD_AS_2CHARS(a3), \
    SIMD_AS_2CHARS(a4), SIMD_AS_2CHARS(a5), SIMD_AS_2CHARS(a6), SIMD_AS_2CHARS(a7), \
    SIMD_AS_2CHARS(a8), SIMD_AS_2CHARS(a9), SIMD_AS_2CHARS(aa), SIMD_AS_2CHARS(ab), \
    SIMD_AS_2CHARS(ac), SIMD_AS_2CHARS(ad), SIMD_AS_2CHARS(ae), SIMD_AS_2CHARS(af)}

#define SIMD_MM256_SET1_EPI32(a) \
	{SIMD_AS_4CHARS(a), SIMD_AS_4CHARS(a), SIMD_AS_4CHARS(a), SIMD_AS_4CHARS(a), \
	SIMD_AS_4CHARS(a), SIMD_AS_4CHARS(a), SIMD_AS_4CHARS(a), SIMD_AS_4CHARS(a)}

#define SIMD_MM256_SET2_EPI32(a0, a1) \
	{SIMD_AS_4CHARS(a0), SIMD_AS_4CHARS(a1), SIMD_AS_4CHARS(a0), SIMD_AS_4CHARS(a1), \
	SIMD_AS_4CHARS(a0), SIMD_AS_4CHARS(a1), SIMD_AS_4CHARS(a0), SIMD_AS_4CHARS(a1)}

#define SIMD_MM256_SETR_EPI32(a0, a1, a2, a3, a4, a5, a6, a7) \
    {SIMD_AS_4CHARS(a0), SIMD_AS_4CHARS(a1), SIMD_AS_4CHARS(a2), SIMD_AS_4CHARS(a3), \
    SIMD_AS_4CHARS(a4), SIMD_AS_4CHARS(a5), SIMD_AS_4CHARS(a6), SIMD_AS_4CHARS(a7)}

#define SIMD_MM256_SET1_EPI64(a) \
	{SIMD_AS_8CHARS(a), SIMD_AS_8CHARS(a), SIMD_AS_8CHARS(a), SIMD_AS_8CHARS(a)}

#define SIMD_MM256_SET2_EPI64(a0, a1) \
	{SIMD_AS_8CHARS(a0), SIMD_AS_8CHARS(a1), SIMD_AS_8CHARS(a0), SIMD_AS_8CHARS(a1)}

#define SIMD_MM256_SETR_EPI64(a0, a1, a2, a3) \
    {SIMD_AS_8CHARS(a0), SIMD_AS_8CHARS(a1), SIMD_AS_8CHARS(a2), SIMD_AS_8CHARS(a3)}

#pragma warning(push) 
#pragma warning(disable: 4310)
const __m128i mSimdZero = SIMD_MM_SET1_EPI8(0);
const __m128i mSimdInvZero = SIMD_MM_SET1_EPI8(0xFF);

const size_t mSimd128bit = sizeof(__m128i);
const size_t mSimd256bit = 2 * mSimd128bit;
const size_t mSimd512bit = 4 * mSimd128bit;
const size_t mSimd1024bit = 8 * mSimd128bit;
const size_t mSimd64Bit = mSimd128bit / 2;

const __m128i K8_01 = SIMD_MM_SET1_EPI8(0x01);
const __m128i K8_02 = SIMD_MM_SET1_EPI8(0x02);
const __m128i K8_03 = SIMD_MM_SET1_EPI8(0x03);
const __m128i K8_04 = SIMD_MM_SET1_EPI8(0x04);
const __m128i K8_07 = SIMD_MM_SET1_EPI8(0x07);
const __m128i K8_08 = SIMD_MM_SET1_EPI8(0x08);
const __m128i K8_10 = SIMD_MM_SET1_EPI8(0x10);
const __m128i K8_20 = SIMD_MM_SET1_EPI8(0x20);
const __m128i K8_40 = SIMD_MM_SET1_EPI8(0x40);
const __m128i K8_80 = SIMD_MM_SET1_EPI8(0x80);

const __m128i K8_01_FF = SIMD_MM_SET2_EPI8(0x01, 0xFF);

const __m128i K16_0001 = SIMD_MM_SET1_EPI16(0x0001);
const __m128i K16_0002 = SIMD_MM_SET1_EPI16(0x0002);
const __m128i K16_0003 = SIMD_MM_SET1_EPI16(0x0003);
const __m128i K16_0004 = SIMD_MM_SET1_EPI16(0x0004);
const __m128i K16_0005 = SIMD_MM_SET1_EPI16(0x0005);
const __m128i K16_0006 = SIMD_MM_SET1_EPI16(0x0006);
const __m128i K16_0008 = SIMD_MM_SET1_EPI16(0x0008);
const __m128i K16_0020 = SIMD_MM_SET1_EPI16(0x0020);
const __m128i K16_0080 = SIMD_MM_SET1_EPI16(0x0080);
const __m128i K16_00FF = SIMD_MM_SET1_EPI16(0x00FF);
const __m128i K16_FF00 = SIMD_MM_SET1_EPI16(0xFF00);

const __m128i K32_00000001 = SIMD_MM_SET1_EPI32(0x00000001);
const __m128i K32_00000002 = SIMD_MM_SET1_EPI32(0x00000002);
const __m128i K32_00000004 = SIMD_MM_SET1_EPI32(0x00000004);
const __m128i K32_00000008 = SIMD_MM_SET1_EPI32(0x00000008);
const __m128i K32_000000FF = SIMD_MM_SET1_EPI32(0x000000FF);
const __m128i K32_0000FFFF = SIMD_MM_SET1_EPI32(0x0000FFFF);
const __m128i K32_00010000 = SIMD_MM_SET1_EPI32(0x00010000);
const __m128i K32_01000000 = SIMD_MM_SET1_EPI32(0x01000000);
const __m128i K32_FFFFFF00 = SIMD_MM_SET1_EPI32(0xFFFFFF00);
#pragma warning(pop)

#if defined(SSE) || defined(SSE2)

template <bool align> mINLINE void Store(float_t *pData, __m128 value);

template <> mINLINE void Store<false>(float_t *pData, __m128 value)
{
  _mm_storeu_ps(pData, value);
}

template <> mINLINE void Store<true>(float_t *pData, __m128 value)
{
  _mm_store_ps(pData, value);
}

template <int part> mINLINE void StoreHalf(float_t *pData, __m128 value);

template <> mINLINE void StoreHalf<0>(float_t *pData, __m128 value)
{
  _mm_storel_pi((__m64*)pData, value);
}

template <> mINLINE void StoreHalf<1>(float_t *pData, __m128 value)
{
  _mm_storeh_pi((__m64*)pData, value);
}

template <bool align> mINLINE __m128 Load(const float * p);

template <> mINLINE __m128 Load<false>(const float * p)
{
  return _mm_loadu_ps(p);
}

template <> mINLINE __m128 Load<true>(const float * p)
{
  return _mm_load_ps(p);
}

mINLINE __m128 Load(const float * p0, const float * p1)
{
  return _mm_loadh_pi(_mm_loadl_pi(_mm_setzero_ps(), (__m64*)p0), (__m64*)p1);
}

#endif // defined(SSE) || defined(SSE2)

#if defined(SSE2)

template <bool align> mINLINE void Store(__m128i *pData, __m128i value);

template <> mINLINE void Store<false>(__m128i *pData, __m128i value)
{
  _mm_storeu_si128(pData, value);
}

template <> mINLINE void Store<true>(__m128i *pData, __m128i value)
{
  _mm_store_si128(pData, value);
}

template <bool align> mINLINE void StoreMasked(__m128i *pData, __m128i value, __m128i mask)
{
  __m128i old = Load<align>(pData);

  Store<align>(pData, Combine(mask, value, old));
}

template <bool align> mINLINE __m128i Load(const __m128i * p);

template <> mINLINE __m128i Load<false>(const __m128i * p)
{
  return _mm_loadu_si128(p);
}

template <> mINLINE __m128i Load<true>(const __m128i * p)
{
  return _mm_load_si128(p);
}

template <bool align> mINLINE __m128i LoadMaskI8(const __m128i * p, __m128i index)
{
  return _mm_cmpeq_epi8(Load<align>(p), index);
}

template <size_t count> mINLINE __m128i LoadBeforeFirst(__m128i first)
{
  return _mm_or_si128(_mm_slli_si128(first, count), _mm_and_si128(first, _mm_srli_si128(mSimdInvZero, mSimd128bit - count)));
}

template <size_t count> mINLINE __m128i LoadAfterLast(__m128i last)
{
  return _mm_or_si128(_mm_srli_si128(last, count), _mm_and_si128(last, _mm_slli_si128(mSimdInvZero, mSimd128bit - count)));
}

template <bool align, size_t step> mINLINE void LoadNose3(const uint8_t * p, __m128i a[3])
{
  a[1] = Load<align>((__m128i*)p);
  a[0] = LoadBeforeFirst<step>(a[1]);
  a[2] = _mm_loadu_si128((__m128i*)(p + step));
}

template <bool align, size_t step> mINLINE void LoadBody3(const uint8_t * p, __m128i a[3])
{
  a[0] = _mm_loadu_si128((__m128i*)(p - step));
  a[1] = Load<align>((__m128i*)p);
  a[2] = _mm_loadu_si128((__m128i*)(p + step));
}

template <bool align, size_t step> mINLINE void LoadTail3(const uint8_t * p, __m128i a[3])
{
  a[0] = _mm_loadu_si128((__m128i*)(p - step));
  a[1] = Load<align>((__m128i*)p);
  a[2] = LoadAfterLast<step>(a[1]);
}

template <bool align, size_t step> mINLINE void LoadNose5(const uint8_t * p, __m128i a[5])
{
  a[2] = Load<align>((__m128i*)p);
  a[1] = LoadBeforeFirst<step>(a[2]);
  a[0] = LoadBeforeFirst<step>(a[1]);
  a[3] = _mm_loadu_si128((__m128i*)(p + step));
  a[4] = _mm_loadu_si128((__m128i*)(p + 2 * step));
}

template <bool align, size_t step> mINLINE void LoadBody5(const uint8_t * p, __m128i a[5])
{
  a[0] = _mm_loadu_si128((__m128i*)(p - 2 * step));
  a[1] = _mm_loadu_si128((__m128i*)(p - step));
  a[2] = Load<align>((__m128i*)p);
  a[3] = _mm_loadu_si128((__m128i*)(p + step));
  a[4] = _mm_loadu_si128((__m128i*)(p + 2 * step));
}

template <bool align, size_t step> mINLINE void LoadTail5(const uint8_t * p, __m128i a[5])
{
  a[0] = _mm_loadu_si128((__m128i*)(p - 2 * step));
  a[1] = _mm_loadu_si128((__m128i*)(p - step));
  a[2] = Load<align>((__m128i*)p);
  a[3] = LoadAfterLast<step>(a[2]);
  a[4] = LoadAfterLast<step>(a[3]);
}

mINLINE void LoadNoseDx(const uint8_t * p, __m128i a[3])
{
  a[0] = LoadBeforeFirst<1>(_mm_loadu_si128((__m128i*)p));
  a[2] = _mm_loadu_si128((__m128i*)(p + 1));
}

mINLINE void LoadBodyDx(const uint8_t * p, __m128i a[3])
{
  a[0] = _mm_loadu_si128((__m128i*)(p - 1));
  a[2] = _mm_loadu_si128((__m128i*)(p + 1));
}

mINLINE void LoadTailDx(const uint8_t * p, __m128i a[3])
{
  a[0] = _mm_loadu_si128((__m128i*)(p - 1));
  a[2] = LoadAfterLast<1>(_mm_loadu_si128((__m128i*)p));
}

mINLINE __m128i SaturateI16ToU8(__m128i value)
{
  return _mm_min_epi16(K16_00FF, _mm_max_epi16(value, mSimdZero));
}

mINLINE __m128i MaxI16(__m128i a, __m128i b, __m128i c)
{
  return _mm_max_epi16(a, _mm_max_epi16(b, c));
}

mINLINE __m128i MinI16(__m128i a, __m128i b, __m128i c)
{
  return _mm_min_epi16(a, _mm_min_epi16(b, c));
}

mINLINE void SortU8(__m128i & a, __m128i & b)
{
  __m128i t = a;
  a = _mm_min_epu8(t, b);
  b = _mm_max_epu8(t, b);
}

mINLINE __m128i ShiftLeft(__m128i a, size_t shift)
{
  __m128i t = a;
  if (shift & 8)
    t = _mm_slli_si128(t, 8);
  if (shift & 4)
    t = _mm_slli_si128(t, 4);
  if (shift & 2)
    t = _mm_slli_si128(t, 2);
  if (shift & 1)
    t = _mm_slli_si128(t, 1);
  return t;
}

mINLINE __m128i ShiftRight(__m128i a, size_t shift)
{
  __m128i t = a;
  if (shift & 8)
    t = _mm_srli_si128(t, 8);
  if (shift & 4)
    t = _mm_srli_si128(t, 4);
  if (shift & 2)
    t = _mm_srli_si128(t, 2);
  if (shift & 1)
    t = _mm_srli_si128(t, 1);
  return t;
}

mINLINE __m128i HorizontalSum32(__m128i a)
{
  return _mm_add_epi64(_mm_unpacklo_epi32(a, mSimdZero), _mm_unpackhi_epi32(a, mSimdZero));
}

mINLINE __m128i AbsDifferenceU8(__m128i a, __m128i b)
{
  return _mm_sub_epi8(_mm_max_epu8(a, b), _mm_min_epu8(a, b));
}

mINLINE __m128i AbsDifferenceI16(__m128i a, __m128i b)
{
  return _mm_sub_epi16(_mm_max_epi16(a, b), _mm_min_epi16(a, b));
}

mINLINE __m128i MulU8(__m128i a, __m128i b)
{
  __m128i lo = _mm_mullo_epi16(_mm_unpacklo_epi8(a, mSimdZero), _mm_unpacklo_epi8(b, mSimdZero));
  __m128i hi = _mm_mullo_epi16(_mm_unpackhi_epi8(a, mSimdZero), _mm_unpackhi_epi8(b, mSimdZero));
  return _mm_packus_epi16(lo, hi);
}

mINLINE __m128i DivideI16By255(__m128i value)
{
  return _mm_srli_epi16(_mm_add_epi16(_mm_add_epi16(value, K16_0001), _mm_srli_epi16(value, 8)), 8);
}

mINLINE __m128i BinomialSum16(const __m128i & a, const __m128i & b, const __m128i & c)
{
  return _mm_add_epi16(_mm_add_epi16(a, c), _mm_add_epi16(b, b));
}

mINLINE __m128i BinomialSum16(const __m128i & a, const __m128i & b, const __m128i & c, const __m128i & d)
{
  return _mm_add_epi16(_mm_add_epi16(a, d), _mm_mullo_epi16(_mm_add_epi16(b, c), K16_0003));
}

mINLINE __m128i Combine(__m128i mask, __m128i positive, __m128i negative)
{
  return _mm_or_si128(_mm_and_si128(mask, positive), _mm_andnot_si128(mask, negative));
}

mINLINE __m128i AlphaBlendingI16(__m128i src, __m128i dst, __m128i alpha)
{
  return DivideI16By255(_mm_add_epi16(_mm_mullo_epi16(src, alpha), _mm_mullo_epi16(dst, _mm_sub_epi16(K16_00FF, alpha))));
}

template <int part> mINLINE __m128i UnpackU8(__m128i a, __m128i b = mSimdZero);

template <> mINLINE __m128i UnpackU8<0>(__m128i a, __m128i b)
{
  return _mm_unpacklo_epi8(a, b);
}

template <> mINLINE __m128i UnpackU8<1>(__m128i a, __m128i b)
{
  return _mm_unpackhi_epi8(a, b);
}

template <int index> __m128i U8To16(__m128i a);

template <> mINLINE __m128i U8To16<0>(__m128i a)
{
  return _mm_and_si128(a, K16_00FF);
}

template <> mINLINE __m128i U8To16<1>(__m128i a)
{
  return _mm_and_si128(_mm_srli_si128(a, 1), K16_00FF);
}

template <int part> mINLINE __m128i UnpackU16(__m128i a, __m128i b = mSimdZero);

template <> mINLINE __m128i UnpackU16<0>(__m128i a, __m128i b)
{
  return _mm_unpacklo_epi16(a, b);
}

template <> mINLINE __m128i UnpackU16<1>(__m128i a, __m128i b)
{
  return _mm_unpackhi_epi16(a, b);
}

template <int part> mINLINE __m128i UnpackI16(__m128i a);

template <> mINLINE __m128i UnpackI16<0>(__m128i a)
{
  return _mm_srai_epi32(_mm_unpacklo_epi16(a, a), 16);
}

template <> mINLINE __m128i UnpackI16<1>(__m128i a)
{
  return _mm_srai_epi32(_mm_unpackhi_epi16(a, a), 16);
}

mINLINE __m128i DivideBy16(__m128i value)
{
  return _mm_srli_epi16(_mm_add_epi16(value, K16_0008), 4);
}

template <int index> mINLINE __m128 Broadcast(__m128 a)
{
  return _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(a), index * 0x55));
}

template<int imm> mINLINE __m128i Shuffle32i(__m128i lo, __m128i hi)
{
  return _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(lo), _mm_castsi128_ps(hi), imm));
}

template<int imm> mINLINE __m128 Shuffle32f(__m128 a)
{
  return _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(a), imm));
}

mINLINE __m128i Average16(const __m128i & a, const __m128i & b, const __m128i & c, const __m128i & d)
{
  return _mm_srli_epi16(_mm_add_epi16(_mm_add_epi16(_mm_add_epi16(a, b), _mm_add_epi16(c, d)), K16_0002), 2);
}

mINLINE __m128i Merge16(const __m128i & even, __m128i odd)
{
  return _mm_or_si128(_mm_slli_si128(odd, 1), even);
}

#endif // defined(SSE2)

#endif // mSimd_h__
