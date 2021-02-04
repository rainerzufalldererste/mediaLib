#ifndef mKeyboardState_h__
#define mKeyboardState_h__

#include "mediaLib.h"


#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "whWTaA4ClM03L14ux8pTlNy9Nh6E52xfnSexET7VEu1G6R+lQUv4X7ge6TkZa8IUwSOQBPSyl20/2eZS"
#endif

/**
*  \brief The SDL keyboard scancode representation.
*
*  Values of this type are used to represent keyboard keys, among other places
*  in the \link SDL_Keysym::scancode key.keysym.scancode \endlink field of the
*  SDL_Event structure.
*
*  The values in this enumeration are based on the USB usage page standard:
*  http://www.usb.org/developers/hidpage/Hut1_12v2.pdf
*/

// Copied from SDL_scancode.h
enum mKey
{
  mK_UNKNOWN = 0,

  /**
   *  \name Usage page 0x07
   *
   *  These values are from usage page 0x07 (USB keyboard page).
   */
  /* @{ */

  mK_A = 4,
  mK_B = 5,
  mK_C = 6,
  mK_D = 7,
  mK_E = 8,
  mK_F = 9,
  mK_G = 10,
  mK_H = 11,
  mK_I = 12,
  mK_J = 13,
  mK_K = 14,
  mK_L = 15,
  mK_M = 16,
  mK_N = 17,
  mK_O = 18,
  mK_P = 19,
  mK_Q = 20,
  mK_R = 21,
  mK_S = 22,
  mK_T = 23,
  mK_U = 24,
  mK_V = 25,
  mK_W = 26,
  mK_X = 27,
  mK_Y = 28,
  mK_Z = 29,

  mK_1 = 30,
  mK_2 = 31,
  mK_3 = 32,
  mK_4 = 33,
  mK_5 = 34,
  mK_6 = 35,
  mK_7 = 36,
  mK_8 = 37,
  mK_9 = 38,
  mK_0 = 39,

  mK_Return = 40,
  mK_Escape = 41,
  mK_Backspace = 42,
  mK_Tab = 43,
  mK_Space = 44,

  mK_Minus = 45,
  mK_Equals = 46,
  mK_Leftbracket = 47,
  mK_Rightbracket = 48,
  mK_Backslash = 49, /**< located at the lower left of the return
                                *   key on ISO keyboards and at the right end
                                *   of the QWERTY row on aNSI keyboards.
                                *   Produces REVERSE SOLIDUS (backslash) and
                                *   VERTICaL LINE in a US layout, REVERSE
                                *   SOLIDUS and VERTICaL LINE in a UK Mac
                                *   layout, NUMBER SIGN and TILDE in a UK
                                *   Windows layout, DOLLaR SIGN and POUND SIGN
                                *   in a Swiss German layout, NUMBER SIGN and
                                *   aPOSTROPHE in a German layout, GRaVE
                                *   aCCENT and POUND SIGN in a French Mac
                                *   layout, and aSTERISK and MICRO SIGN in a
                                *   French Windows layout.
                                */
  mK_NonUsHash = 50, /**< isO USB keyboards actually use this code
                                *   instead of 49 for the same key, but all
                                *   OSes I've seen treat the two codes
                                *   identically. So, as an implementor, unless
                                *   your keyboard generates both of those
                                *   codes and your OS treats them differently,
                                *   you should generate mK_BaCKSLaSH
                                *   instead of this code. as a user, you
                                *   should not rely on this code because SDL
                                *   will never generate it with most (all?)
                                *   keyboards.
                                */
  mK_Semicolon = 51,
  mK_Apostrophe = 52,
  mK_Grave = 53,
  mK_Tilde = 53, /**< located in the top left corner (on both aNSI
                            *   and ISO keyboards). Produces GRaVE aCCENT and
                            *   TILDE in a US Windows layout and in US and UK
                            *   Mac layouts on aNSI keyboards, GRaVE aCCENT
                            *   and NOT SIGN in a UK Windows layout, SECTION
                            *   SIGN and PLUS-MINUS SIGN in US and UK Mac
                            *   layouts on ISO keyboards, SECTION SIGN and
                            *   DEGREE SIGN in a Swiss German layout (Mac:
                            *   only on ISO keyboards), CIRCUMFLEX aCCENT and
                            *   DEGREE SIGN in a German layout (Mac: only on
                            *   ISO keyboards), SUPERSCRIPT TWO and TILDE in a
                            *   French Windows layout, COMMERCIaL aT and
                            *   NUMBER SIGN in a French Mac layout on ISO
                            *   keyboards, and LESS-THaN SIGN and GREaTER-THaN
                            *   SIGN in a Swiss German, German, or French Mac
                            *   layout on aNSI keyboards.
                            */
  mK_Comma = 54,
  mK_Period = 55,
  mK_Slash = 56,

  mK_Capslock = 57,

  mK_F1 = 58,
  mK_F2 = 59,
  mK_F3 = 60,
  mK_F4 = 61,
  mK_F5 = 62,
  mK_F6 = 63,
  mK_F7 = 64,
  mK_F8 = 65,
  mK_F9 = 66,
  mK_F10 = 67,
  mK_F11 = 68,
  mK_F12 = 69,

  mK_PrintScreen = 70,
  mK_ScrollLock = 71,
  mK_Pause = 72,
  mK_Insert = 73, /**< insert on PC, help on some Mac keyboards (but
                                 does send code 73, not 117) */
  mK_Home = 74,
  mK_PageUp = 75,
  mK_Delete = 76,
  mK_End = 77,
  mK_PageDown = 78,
  mK_Right = 79,
  mK_Left = 80,
  mK_Down = 81,
  mK_Up = 82,

  mK_NumLockClear = 83, /**< num lock on PC, clear on Mac keyboards
                                   */
  mK_Keypad_Divide = 84,
  mK_Keypad_Multiply = 85,
  mK_Keypad_Minus = 86,
  mK_Keypad_Plus = 87,
  mK_Keypad_Enter = 88,
  mK_Keypad_1 = 89,
  mK_Keypad_2 = 90,
  mK_Keypad_3 = 91,
  mK_Keypad_4 = 92,
  mK_Keypad_5 = 93,
  mK_Keypad_6 = 94,
  mK_Keypad_7 = 95,
  mK_Keypad_8 = 96,
  mK_Keypad_9 = 97,
  mK_Keypad_0 = 98,
  mK_Keypad_Period = 99,

  mK_NonUsBackslash = 100, /**< This is the additional key that ISO
                                      *   keyboards have over aNSI ones,
                                      *   located between left shift and Y.
                                      *   Produces GRaVE aCCENT and TILDE in a
                                      *   US or UK Mac layout, REVERSE SOLIDUS
                                      *   (backslash) and VERTICaL LINE in a
                                      *   US or UK Windows layout, and
                                      *   LESS-THaN SIGN and GREaTER-THaN SIGN
                                      *   in a Swiss German, German, or French
                                      *   layout. */
  mK_Application = 101, /**< windows contextual menu, compose */
  mK_Power = 102, /**< the uSB document says this is a status flag,
                             *   not a physical key - but some Mac keyboards
                             *   do have a power key. */
  mK_Keypad_Equals = 103,
  mK_F13 = 104,
  mK_F14 = 105,
  mK_F15 = 106,
  mK_F16 = 107,
  mK_F17 = 108,
  mK_F18 = 109,
  mK_F19 = 110,
  mK_F20 = 111,
  mK_F21 = 112,
  mK_F22 = 113,
  mK_F23 = 114,
  mK_F24 = 115,
  mK_Execute = 116,
  mK_Help = 117,
  mK_Menu = 118,
  mK_Select = 119,
  mK_Stop = 120,
  mK_Again = 121,   /**< redo */
  mK_Undo = 122,
  mK_Cut = 123,
  mK_Copy = 124,
  mK_Paste = 125,
  mK_Find = 126,
  mK_Mute = 127,
  mK_VolumeUp = 128,
  mK_VolumeDown = 129,
/* not sure whether there's a reason to enable these */
/* mK_LockingCapsLock = 130,  */
/* mK_LockingNumLock = 131, */
/* mK_LockingScrollLock = 132, */
  mK_Keypad_Comma = 133,
  mK_Keypad_Equalsas400 = 134,

  mK_International1 = 135, /**< used on asian keyboards, see
                                          footnotes in USB doc */
  mK_International2 = 136,
  mK_International3 = 137, /**< Yen */
  mK_International4 = 138,
  mK_International5 = 139,
  mK_International6 = 140,
  mK_International7 = 141,
  mK_International8 = 142,
  mK_International9 = 143,
  mK_Lang1 = 144, /**< hangul/English toggle */
  mK_Lang2 = 145, /**< hanja conversion */
  mK_Lang3 = 146, /**< katakana */
  mK_Lang4 = 147, /**< hiragana */
  mK_Lang5 = 148, /**< zenkaku/Hankaku */
  mK_Lang6 = 149, /**< reserved */
  mK_Lang7 = 150, /**< reserved */
  mK_Lang8 = 151, /**< reserved */
  mK_Lang9 = 152, /**< reserved */

  mK_AltErase = 153, /**< erase-Eaze */
  mK_SysReq = 154,
  mK_Cancel = 155,
  mK_Clear = 156,
  mK_Prior = 157,
  mK_Return2 = 158,
  mK_Separator = 159,
  mK_Out = 160,
  mK_Oper = 161,
  mK_ClearAgain = 162,
  mK_CrSel = 163,
  mK_ExSel = 164,

  mK_Keypad_00 = 176,
  mK_Keypad_000 = 177,
  mK_ThousandsSeparator = 178,
  mK_DecimalSeparator = 179,
  mK_CurrencyUnit = 180,
  mK_CurrencySubUnit = 181,
  mK_Keypad_LeftParen = 182,
  mK_Keypad_RightParen = 183,
  mK_Keypad_LeftBrace = 184,
  mK_Keypad_RightBrace = 185,
  mK_Keypad_Tab = 186,
  mK_Keypad_Backspace = 187,
  mK_Keypad_A = 188,
  mK_Keypad_B = 189,
  mK_Keypad_C = 190,
  mK_Keypad_D = 191,
  mK_Keypad_E = 192,
  mK_Keypad_F = 193,
  mK_Keypad_XOr = 194,
  mK_Keypad_Power = 195,
  mK_Keypad_Percent = 196,
  mK_Keypad_Less = 197,
  mK_Keypad_Greater = 198,
  mK_Keypad_Ampersand = 199,
  mK_Keypad_DblAmpersand = 200,
  mK_Keypad_Verticalbar = 201,
  mK_Keypad_DblVerticalbar = 202,
  mK_Keypad_Colon = 203,
  mK_Keypad_Hash = 204,
  mK_Keypad_Space = 205,
  mK_Keypad_At = 206,
  mK_Keypad_Exclam = 207,
  mK_Keypad_Nemstore = 208,
  mK_Keypad_Memrecall = 209,
  mK_Keypad_Memclear = 210,
  mK_Keypad_Memadd = 211,
  mK_Keypad_Memsubtract = 212,
  mK_Keypad_Memmultiply = 213,
  mK_Keypad_Memdivide = 214,
  mK_Keypad_PlusMinus = 215,
  mK_Keypad_Clear = 216,
  mK_Keypad_Clearentry = 217,
  mK_Keypad_Binary = 218,
  mK_Keypad_Octal = 219,
  mK_Keypad_Decimal = 220,
  mK_Keypad_Hexadecimal = 221,

  mK_LCtrl = 224,
  mK_LShift = 225,
  mK_LAlt = 226, /**< alt, option */
  mK_LGui = 227, /**< windows, command (apple), meta */
  mK_RCtrl = 228,
  mK_RShift = 229,
  mK_RAlt = 230, /**< alt gr, option */
  mK_RGui = 231, /**< windows, command (apple), meta */

  mK_Mode = 257,    /**< i'm not sure if this is really not covered
                               *   by any of the above, but since there's a
                               *   special KMOD_MODE for it I'm adding it here
                               */

  /* @} *//* usage page 0x07 */

  /**
   *  \name usage page 0x0c
   *
   *  these values are mapped from usage page 0x0C (USB consumer page).
   */
  /* @{ */

  mK_AudioNext = 258,
  mK_AudioPrev = 259,
  mK_AudioStop = 260,
  mK_AudioPlay = 261,
  mK_AudioMute = 262,
  mK_MediaSelect = 263,
  mK_WWW = 264,
  mK_Mail = 265,
  mK_Calculator = 266,
  mK_Computer = 267,
  mK_AC_Search = 268,
  mK_AC_Home = 269,
  mK_AC_Back = 270,
  mK_AC_Forward = 271,
  mK_AC_Stop = 272,
  mK_AC_Refresh = 273,
  mK_AC_Bookmarks = 274,

  /* @} *//* usage page 0x0c */

  /**
   *  \name walther keys
   *
   *  these are values that Christian Walther added (for mac keyboard?).
   */
  /* @{ */

  mK_BrightnessDown = 275,
  mK_BrightnessUp = 276,
  mK_DisplaySwitch = 277, /**< display mirroring/dual display
                                         switch, video mode switch */
  mK_KbdillumToggle = 278,
  mK_KbdillumDown = 279,
  mK_KbdillumUp = 280,
  mK_Eject = 281,
  mK_Sleep = 282,

  mK_App1 = 283,
  mK_App2 = 284,

  /* @} *//* walther keys */

  /**
   *  \name usage page 0x0c (additional media keys)
   *
   *  these values are mapped from usage page 0x0C (USB consumer page).
   */
  /* @{ */

  mK_AudioRewind = 285,
  mK_AudioFastForward = 286,

  /* @} *//* Usage page 0x0C (additional media keys) */

  /* Add any other keys here. */

  mKey_Count = 512 /**< not a key, just marks the number of scancodes
                        for array bounds */
};

struct mKeyboardState
{
  const uint8_t *pKeys = nullptr;
  size_t keyCount = 0;
  uint8_t currentKeys[mKey_Count];
  uint8_t lastKeys[mKey_Count];
  bool hasPreviousState = false;
};

mFUNCTION(mKeyboardState_Update, OUT mKeyboardState *pKeyboardState);
bool mKeyboardState_IsKeyDown(const mKeyboardState &keyboardState, const mKey key);
bool mKeyboardState_IsKeyUp(const mKeyboardState &keyboardState, const mKey key);
bool mKeyboardState_KeyPress(const mKeyboardState &keyboardState, const mKey key);
bool mKeyboardState_KeyLift(const mKeyboardState &keyboardState, const mKey key);

#endif // mKeyboardState_h__
