# Coding-Standard

## General
 * We use two spaces as indentation instead of tabs.
 * Files end with an empty line.
 * Avoid more than one empty line at any time in implementations. To distinguish between sections more clearly, use comments.
 * Avoid spurious free spaces. For example avoid `if (someVar == 0)...`, where the dots mark the spurious free spaces. Consider enabling "View White Space (Ctrl+E, S)" if using Visual Studio, to aid detection.
 * For non code files (like xml) our current best guidance is consistency. When editing files, keep new code and changes consistent with the style in the files. For new files, it should conform to the style for that component. Last, if there's a completely new component, anything that is reasonably broadly accepted is fine.
 * If a file happens to differ in style from these guidelines, the existing style in that file takes precedence.
 * Functions should be as transactional as possible. If an operation can not be executed don't leave the object state corrupted by the changes that function made.
 * Rather than using the `epilogue`-Paradigm we prefer to use deferred function calls to clean up after functions where possible (`mDEFER`, `mDEFER_CALL`, `mDEFER_CALL_ON_ERROR`, etc.). Use deferred calls to your advantage by placing them next to the object creation to improve readability. Always keep the scope of a deferred call in mind.
 * We use the c-style naming convention rather than using member functions to improve readability. (This also has some advantages for templated code.)
 * Only data-only types (and some special types like `mString` or `mSharedPointer`) use constructors, operators or member functions because these heavily influence the possibility, ease and quality of error handling. (Even for strings not using operators & constructors is preferred.)
 * We use `mSharedPointer` (or its alias `mPtr`) to ensure reference counting for objects in single-threaded code.
 * We don't ever use native exceptions because it's incredibly difficuly to write code that handles exception states properly. They can also compromise application security & speed.
 * We avoid using STL functions & objects. Exceptions are `std::move`, `std::forward`, `std::swap`, `std::function`, `std::atomic` and objects used for template specialization.
 * Memory is always allocated through the provided allocators. Remember to placement `new` objects from other libraries that use constructors inside your objects because zeroed memory might not be a valid object state for them.
 * If a function takes pointers as a parameter mark them with `IN`, `OUT`, `IN_OUT` and `OPTIONAL` to clarify the interface.
 * Take error handling very seriously. Internal functions however can be considered pre-error-handled if they can only be called by functions that already checked for this error. If not manually dealing with error codes every function call should be inside a `mERROR_CHECK` (or `mERROR_IF`) macro (or one of its derivatives for `epilogue` based code).
 * If returning from a function use the `mRETURN_RESULT` or `mRETURN_SUCCESS` macro, which ensures that `mDEFER_ON_ERROR` and `mDEFER_ON_ERROR` know about the returned `mResult`.
 
 We follow the following few guidelines when implementing objects:
 * If an objects memory is zero it is valid (as a null-object) and can be destroyed successfully.
 * Moving(!) an objects memory is equivalent to `memmove`ing it. Copy-constructors should be called however when copying into memory that is not zero when implementing data-structures. (Because e.g. `mSharedPointer` would otherwise try to free the object that previously lived at this memory location.) Alternatively you can zero the memory before copy-assigning to it.
 * If an object is used in collections (without being wrapped in e.g. a `mSharedPointer`) provide a `mDestruct(T *pObject)` function, that knows how to destruct this type.
 
## Naming
 * Name your variables something meaningful so that your code is self-explanatory and you don't need a lot of comments.
 * Variable names are always in camelCase; this includes `static` & `const` variables.
 * Pointers start with 'p' as in `T *pSize`, double pointers start with 'pp' etc.
 * Objects that don't live in CPU-addressable memory like CUDA pointers should be prefixed appropriately. (`__cuda__` for CUDA, wherever a pointer enters GPU memory. `__cuda__pData` is a GPU pointer to data; `p__cuda__pData` is a CPU-addressable pointer to a pointer to GPU data)
 * `char` pointers don't start with 'p' as in `char *name`, double `char` pointers start with 'p' etc. (keep in mind: _`char` is for strings, not for data. When dealing with byte sized data use `uint8_t` instead._)
 * Enums that are closely related to an object start with the name of the object followed by an underscore and the purpose of the `enum` in PascalCase (like `mSystemError_MessageBoxButton`; if there is no related object: `mResult`) Elements in an `enum` start with the prefix, followed by the initials of the related object, underscore, the initials of the `enum`, underscore, purpose of the element in PascalCase (like `mSE_MBB_OK`; if there is no related object: `mR_Success`).
 * `#define` constant literals in UPPERCASE with underscores (like `MAX_INT64`).
 * Function names are always in PascalCase and follow the `prefixObject_FunctionName` naming scheme (like `mPixelFormat_TransformBuffer`).
 * Internal functions should be postfixed with `_Internal` and declared as `static`.
 
## Variables
Global variables:
 * Try to minimize use of any global variables as their initialization order can cause horrible bugs that come down to order of compilation.
 * Declare global variables in source-files not in headers.
 * The `constexpr` / `const` qualifier should be used wherever possible.
 
Local variables:
 * Should be declared near first use for better readability.

## Functions
 * Generally all functions follow the `mFUNCTION` scheme. The only exceptions are performance critical sections and data-only objects.
 
```c++
// Template parameters on separate line.
template<typename T>
// Function definition wrapped in mFUNCTION macro, pointer parameter marked as `OUT` for readability.
inline mFUNCTION(mQueue_PopBack, mPtr<mQueue<T>> &queue, OUT T *pItem)
{
  // mFUNCTION_SETUP macro to setup mResult.
  mFUNCTION_SETUP();

  // Checking parameters for possible invalid inputs.
  mERROR_IF(queue == nullptr || pItem == nullptr, mR_ArgumentNull);

  // Calling a function wrapped in mERROR_CHECK macro.
  mERROR_CHECK(mQueue_PopAt_Internal(queue, queue->count - 1, pItem));

  --queue->count;

  // return is wrapped in mRETURN_SUCCESS or mRETURN_RESULT macro to ensure functionality of mDEFER_CALL_ON_ERROR and mDEFER_ON_ERROR.
  mRETURN_SUCCESS();
}
```

 * For longer or more complex functions use the following scheme to help with readability:
 
```c++
// Parse command line arguments.
{
  ParseCommandLineArguments();
}

// Decode input.
{
  bool hasInput = true;

  while (hasInput)
    DecodeInput(&hasInput);
}
```

 * In combination with such blocks, prefer `do`-`while`(0) over doing lots of nested `if`-`else` statements for error-handling:
 
```c++
// Good:
do
{
  if (SomeErrorOccurs())
    break;
  
  if (SomeOtherErrorOccurs())
    break;

  // ...

  Success();
} while (0)

// Bad:
{
  if (SomeErrorOccurs())
  {
    break;
  }
  else
  {
    if (SomeOtherErrorOccurs())
    {
      break;
    }
    else
    {
      // ...

      Success();
    }
  }
}
```

 * Function implementations should include all qualifiers for the parameters (e.g. `const`) and the default parameters (commented out from the equals sign: `void mSomeObject_SomeFunction(const size_t someVariable /* = defaultValue */)`).
 * Creation and destruction functions take double pointers to avoid copies and to clean up the actual object in question rather than a copy of it.
 * When a function is only accessible inside one function and will not be of any use to other functions, turn them into a static function inside a struct called `_internal` that lives in the scope of the function that uses it.

Lambda Functions:
 * If no external variables or values are being used inside the lambda function, don't capture automatically or explicitly.
 * If external variables or values are being used inside the lambda, try simplifying the captures as much as possible.
 * If a lambda function executes multiple operations it should have a multi line body.

```c++
  size_t a, b, c;
  
  // ...
  
  for (int64_t i = 0; i < count; ++i)
  {
    std::function<bool (void)> lambda = [&, i]() // `i` is passed as a copy, all other captured objects are passed by reference.
    {
      // Use captured resources.
      a = b = c = i;

      RETURN true;
    };

    // ...
  }

  return;
```

## Comments
 * Comments start with a space and usually end with a `.` (like `// Comment.`).
 * If applicable use the `TODO: ` or `HACK: ` tags in your comments.
 * Name your variables something meaningful so that your code is self-explanatory and you don't need a lot of comments.
 * Generally prefer using `//` over `/**/`.

```c++
  // Calculate the one-dimensional discrete cosine transform.
  {
    const float_t a07 = pBuffer[0] + pBuffer[7];
    const float_t a16 = pBuffer[1] + pBuffer[6];
    const float_t a25 = pBuffer[2] + pBuffer[5];
    const float_t a34 = pBuffer[3] + pBuffer[4];
    
    const float_t s07 = pBuffer[0] - pBuffer[7];
    const float_t s61 = pBuffer[6] - pBuffer[1];
    const float_t s25 = pBuffer[2] - pBuffer[5];
    const float_t s43 = pBuffer[4] - pBuffer[3];
    
    const float_t v0 = a07 + a34;
    const float_t v1 = a07 - a34;
    const float_t v2 = a16 + a25;
    const float_t v3 = a16 - a25;

    // ...
  }
```

```c++
// Removes the last element from the collection.
template <typename T>
mFUNCTION(mCollection_PopBack, mPtr<mCollection<T>> &collection, OUT T *pItem);
```

## Types
 * When interfacing with other libraries use their native types in your implementation files (like `GLint`, `HANDLE`, ...).
 * The default types for unsigned integers is `size_t`.
 * The default type for data is `uint8_t`. (Not `char`!)
 * The default type for floating point data is `float_t`.
 * The default type for boolean expressions is `bool`.
 * The default type for strings is `mString` (consider using `mInplaceString<Length>`) or `char *`.
 * Otherwise use the unambiguously sized variants of integer types (`int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `int64_t`, `uint64_t`) and `float_t` & `double_t` for floating point numbers.
 * When dealing with SIMD code use the `__m128i`, `__m128` and `__m128d` types e.g. for SSE.
 * When string formatting integer values use the defines from `inttypes.h`. (like `printf("The value is %" PRIu64 ".\n", (uint64_t)value);`)
 
Integer types:
 * Always use unsigned types for bit manipulations, packed values & flags.
 * The type `size_t` should not be serialized to binary data.
 * Use uppercase hexadecimal letters (like `0xFFFFFFFF`).

Floating-point types:
 * Don't use floating-point numbers if integers would do the job just as well.

Casting:
 * Use c-style casts for integers values. (`(int32_t)value` instead of `int32_t(value)`)
 * Use `static_cast` when casting to derived types.
 * Use `reinterpret_cast` where applicable, consider using `mMemcpy` instead.
 * Try to not use `const_cast` unless absolutely necessary.
 
Type Punning:
 * Minimize usage of type punning since it can be harder to maintain & debug if not used in obvious ways.
 * When type punning use `reinterpret_cast`.
 * Consider using `mMemcpy` instead.

## Pointers
 * When pointing to elements in containers keep in mind that the elements might be moved.
 * Pointers (`*`) belong on the right side (`size_t *pValue`).

## References
 * Pass complex types as (`const`) reference to avoid copying.
 * References (`&`) belong on the right side (function parameter as `mPtr<Something> &value`).
 * Avoid the use of references outside of function parameters because they can complicate behavior enormously if used incorrectly.

## Headers
 * Always use Header-Guards.
 * Avoid forward declarations just include headers.

Header Files:
 * Every .cpp file generally has an associated .h file.
 * Implement templated functions in separate .inl files.

## Braces
 * We use [Allman style](http://en.wikipedia.org/wiki/Indent_style#Allman_style) braces.
 * Braces are not required for single line statements.
 * Single line `if` statements that have a multiline `else` statement (and vice versa) should have brackets to be more consistent.
 
```c++
  if (a)
    SingleLineIfStatement();

  if (b)
  {
    SingleLineIfStatement();
  }
  else
  {
    Multi();
    line = ElseStatement();
  }
```

 * Switch statements should generally not have braces for each `case`. If it is cleaner to have braces (e.g. due to declaring variables inside cases) then include the `break` inside that scope. If you do use braces, use braces for every `case` in that `switch` statement. If it cleaner to use a single line for each `case` (provided that there is a single statement or small amount of code per line), do so.
 * Chained / nested `for` loops & `if` statements which are all single lines do not require braces unless it helps readability.

```c++
// Does not require braces:
for (;;)
  if (expr)
    if (expr)
      statement;

// Does require braces:
for (;;)
{
  if (expr)
    statement;
  else if (expr)
    statement;
  else
    statement;
}
```
 
## Source File Structure
 * When using headers from multiple libraries try to group them for better readability.
 * Source files follow the following structure.
 
```c++
#include "htAssociatedHeaderOfThisCpp.h"

#include "mHeaderFromLibraryOne.h"
#include "mHeader.h"
#include "mAnotherHeaderFromTheSameLibrary.h"

#include "htTheseHeadersAreFromADifferentLibrary.h"
#include "htAnotherHeaderFromThisLibrary.h"
#include "htHeader.h"

// Internal variables.

// Declarations of internal functions.

//////////////////////////////////////////////////////////////////////////

// Implementation

//////////////////////////////////////////////////////////////////////////

// Implementation of internal functions

```
 
## Empty lines
 * Use empty lines to ensure readability of functions.
 * Group code sections by logical or semantical differences.
 
```c++
// Statements grouped by different categories
{
  const size_t a = 0;
  const size_t b = 1;
  const size_t c = 2;
  
  size_t d = 3;
  size_t e = 4;
  
  mERROR_CHECK(mDataStructure_DoSomethingWithD(dataStructure, &d));
  mERROR_CHECK(mDataStructure_DoSomethingElseWithEAndD(dataStructure, e, d));
  mERROR_CHECK(mDataStructure_DoSomethingWithE(dataStructure, e));
}

// These statements are logical tightly coupled and therefore belong together.
{
  size_t count;
  mERROR_CHECK(mDataStructure_GetCount(dataStructure, &count));
}

// Multiple initializations can either be grouped by category
{
  size_t a;
  size_t b;
  size_t c;
  
  mERROR_CHECK(mDataStructure_GetA(dataStructure, &a));
  mERROR_CHECK(mDataStructure_GetB(dataStructure, &b));
  mERROR_CHECK(mDataStructure_GetC(dataStructure, &c));
}

// ... or by logical context.
{
  size_t a;
  mERROR_CHECK(mDataStructure_GetA(dataStructure, &a));
  
  size_t b;
  mERROR_CHECK(mDataStructure_GetB(dataStructure, &b));
  
  size_t c;
  mERROR_CHECK(mDataStructure_GetC(dataStructure, &c));
}

// The same applies here: Grouped by category ...
{
  size_t value = 0;
  
  mERROR_CHECK(mDataStructure_SetValue(dataStructure, value));
  mERROR_CHECK(mDataStructure_DoSomethingUnrelated(dataStructure));
  mERROR_CHECK(mDataStructure_DoSomethingElse(dataStructure));
}

// ... could also be grouped by logical context.
{
  size_t value = 0;
  mERROR_CHECK(mDataStructure_SetValue(dataStructure, value));
  
  mERROR_CHECK(mDataStructure_DoSomethingUnrelated(dataStructure));
  mERROR_CHECK(mDataStructure_DoSomethingElse(dataStructure));
}

// The lines before and after a loop or if-else-block should be empty (or a comment, where the line above the comment is empty).
// If we just entered a new scope or are leaving the current scope afterwards this does not apply. (See example) 
{
  if (a == b)
    b = c;
  
  if (c != b)
  {
    if (a == (b - c))
      ++c;
    
    a = c;
    b = a;
  }
  else
  {
    c = a;

    for (size_t i = 0; i < b; ++i)
    {
      ++a;
      --c;
    }
  }
  
  for (size_t i; i < a; ++i)
    ++b;
   
  while (b > a)
  {
    b -= c;
    ++c;
  }
}
```

## Projects and libraries
 * Only use external libraries if there is a very good reason to. Once added to a project it's usually very hard to get rid of external dependencies, and it might bloat the repository forever.
 * Whenever you link to libraries, do it in a way that doesn't limit other projects that might rely on your project in the future.
 * When writing a feature for a library try to keep it as generic as possible and don't make assumptions about the use-case. Document the assumptions that you made.
 * Unit-Test all generic functionality (like data structures, hashing, ...) are mandatory. Preferably all library code should be unit-tested.
