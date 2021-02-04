#ifndef mXml_h__
#define mXml_h__

#include "mediaLib.h"

#ifdef GIT_BUILD // Define __M_FILE__
  #ifdef __M_FILE__
    #undef __M_FILE__
  #endif
  #define __M_FILE__ "6WQGckaDcdFgVisiVDlfzRxG3GmqnX0gT24250s3EbpgYMAZYUajuedZT7QNh4fkufuJvhe8UHhjhzPk"
#endif

struct mXmlReader;

mFUNCTION(mXmlReader_CreateFromString, OUT mPtr<mXmlReader> *pXmlReader, IN mAllocator *pAllocator, const mString &string);
mFUNCTION(mXmlReader_CreateFromString, OUT mPtr<mXmlReader> *pXmlReader, IN mAllocator *pAllocator, const char *string, const size_t length);
mFUNCTION(mXmlReader_CreateFromFile, OUT mPtr<mXmlReader> *pXmlReader, IN mAllocator *pAllocator, const mString &filename);

mFUNCTION(mXmlReader_Destroy, IN_OUT mPtr<mXmlReader> *pXmlReader);

// Returns `mR_ResourceNotFound` if no child elements with that tag could be found.
mFUNCTION(mXmlReader_StepInto, mPtr<mXmlReader> &xmlReader, const mString &tag);
mFUNCTION(mXmlReader_ExitTag, mPtr<mXmlReader> &xmlReader);

// Leave current element & step into the next element with the same name.
// If no other one can be found, leave the current element and exit with `mR_ResourceNotFound`.
mFUNCTION(mXmlReader_StepIntoNext, mPtr<mXmlReader> &xmlReader);

// Steps into the first child element of the current element.
// `pTag` will be set to it's tag.
// Returns `mR_ResourceNotFound` if no child elements.
mFUNCTION(mXmlReader_Visit, mPtr<mXmlReader> &xmlReader, OUT mString *pTag);

// Leave current element & step into the next element.
// If no other one can be found, leave the current element and exit with `mR_ResourceNotFound`.
mFUNCTION(mXmlReader_VisitNext, mPtr<mXmlReader> &xmlReader, OUT mString *pTag);

// Returns `mR_ResourceNotFound` if no attribute with that name could be found.
mFUNCTION(mXmlReader_GetAttributeProperty, mPtr<mXmlReader> &xmlReader, const mString &attributeKey, OUT mString *pValue);

// Returns `mR_ResourceNotFound` if no attribute could be found.
mFUNCTION(mXmlReader_VisitAttribute, mPtr<mXmlReader> &xmlReader, OUT mString *pKey, OUT mString *pValue);

// Returns `mR_ResourceNotFound` if no further attributes could be found.
mFUNCTION(mXmlReader_VisitNextAttribute, mPtr<mXmlReader> &xmlReader, OUT mString *pKey, OUT mString *pValue);

// `pContent` will be set to an empty string if the current element doesn't have any content.
mFUNCTION(mXmlReader_GetContent, mPtr<mXmlReader> &xmlReader, OUT mString *pContent);

struct mXmlWriter;

mFUNCTION(mXmlWriter_Create, OUT mPtr<mXmlWriter> *pXmlWriter, IN mAllocator *pAllocator, const mString &rootElementTag);
mFUNCTION(mXmlWriter_Destroy, IN_OUT mPtr<mXmlWriter> *pXmlWriter);

mFUNCTION(mXmlWriter_ToString, mPtr<mXmlWriter> &xmlWriter, OUT mString *pXmlString, IN mAllocator *pAllocator);
mFUNCTION(mXmlWriter_ToFile, mPtr<mXmlWriter> &xmlWriter, const mString &filename);

mFUNCTION(mXmlWriter_BeginElement, mPtr<mXmlWriter> &xmlWriter, const mString &tag);
mFUNCTION(mXmlWriter_EndElement, mPtr<mXmlWriter> &xmlWriter);

mFUNCTION(mXmlWriter_AddAttribute, mPtr<mXmlWriter> &xmlWriter, const mString &key, const mString &value);
mFUNCTION(mXmlWriter_AddContent, mPtr<mXmlWriter> &xmlWriter, const mString &content);

#endif // mXML_h__
