#include "mXml.h"

#define XML_STATIC 1
#include "expat_external.h"
#include "expat.h"

#include "mFile.h"
#include "mQueue.h"
#include "mKeyValuePair.h"

struct mXmlNode
{
  mString tag, content;
  mUniqueContainer<mQueue<mKeyValuePair<mString, mString>>> attributes;
  mUniqueContainer<mQueue<mXmlNode>> childNodes;

  mXmlNode() = default;
  mXmlNode(mXmlNode &&move) = default;
  mXmlNode & operator =(mXmlNode &&move) = default;
};

struct mXmlNodePosition
{
  size_t index;
  mXmlNode *pNode;
  size_t attributeIndex;

  mXmlNodePosition() :
    pNode(nullptr),
    index((size_t)-1),
    attributeIndex((size_t)-1)
  { }

  mXmlNodePosition(IN mXmlNode *pNode, const size_t index) :
    pNode(pNode),
    index(index),
    attributeIndex((size_t)-1)
  { }
};

struct mXmlReader
{
  mXmlNode rootNode;
  mUniqueContainer<mQueue<mXmlNodePosition>> currentNodeStack;
  mAllocator *pAllocator;
  mResult parseResult;
};

mFUNCTION(mXmlReader_Destroy_Internal, IN_OUT mXmlReader *pXmlReader);

void XMLCALL mXmlReader_StartElement_Internal(void *pUserData, const XML_Char *name, const XML_Char **pAttributes);
void XMLCALL mXmlReader_EndElement_Internal(void *pUserData, const XML_Char *name);
void XMLCALL mXmlReader_HandleData_Internal(void *pUserData, const XML_Char *string, const int32_t length);

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mXmlReader_CreateFromString, OUT mPtr<mXmlReader> *pXmlReader, IN mAllocator *pAllocator, const mString &string)
{
  mFUNCTION_SETUP();

  mERROR_IF(pXmlReader == nullptr, mR_ArgumentNull);
  mERROR_IF(string.hasFailed || string.c_str() == nullptr, mR_InvalidParameter);

  mERROR_CHECK(mXmlReader_CreateFromString(pXmlReader, pAllocator, string.c_str(), string.bytes));

  mRETURN_SUCCESS();
}

mFUNCTION(mXmlReader_CreateFromString, OUT mPtr<mXmlReader> *pXmlReader, IN mAllocator *pAllocator, const char *string, const size_t length)
{
  mFUNCTION_SETUP();

  mERROR_IF(pXmlReader == nullptr || string == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Allocate<mXmlReader>(pXmlReader, pAllocator, [](mXmlReader *pData) { mXmlReader_Destroy_Internal(pData); }, 1));
  mDEFER_CALL_ON_ERROR(pXmlReader, mSetToNullptr<mXmlReader>);

  mXmlReader *pXml = pXmlReader->GetPointer();
  
  pXml->pAllocator = pAllocator;
  mERROR_CHECK(mQueue_Create(&pXml->currentNodeStack, pAllocator));

  // Create & Add Root Node.
  {
    mERROR_CHECK(mQueue_Create(&pXml->rootNode.childNodes, pAllocator));
    mERROR_CHECK(mQueue_EmplaceBack(pXml->currentNodeStack, &pXml->rootNode, (size_t)-1));
  }

  //XML_Char encoding[] = "UTF-8";
  XML_Parser parser = XML_ParserCreate(nullptr);
  mERROR_IF(parser == nullptr, mR_InternalError);
  mDEFER_CALL(parser, XML_ParserFree);

  XML_SetElementHandler(parser, mXmlReader_StartElement_Internal, mXmlReader_EndElement_Internal);
  XML_SetCharacterDataHandler(parser, mXmlReader_HandleData_Internal);
  XML_SetUserData(parser, pXml);

  pXml->parseResult = mR_Success;

  if (XML_STATUS_ERROR == XML_Parse(parser, string, (int)(length - 1), XML_TRUE))
  {
    mPRINT_ERROR("mXmlReader Error: %s (%" PRIu64 ":%" PRIu64 ")\n", XML_ErrorString(XML_GetErrorCode(parser)), (uint64_t)XML_GetErrorLineNumber(parser), (uint64_t)XML_GetErrorColumnNumber(parser));
    mRETURN_RESULT(mR_ResourceInvalid);
  }

  mERROR_IF(mFAILED(pXml->parseResult), pXml->parseResult);

  mERROR_CHECK(mQueue_Clear(pXml->currentNodeStack));
  mERROR_CHECK(mQueue_EmplaceBack(pXml->currentNodeStack, &pXml->rootNode, (size_t)-1));

  mRETURN_SUCCESS();
}

mFUNCTION(mXmlReader_CreateFromFile, OUT mPtr<mXmlReader> *pXmlReader, IN mAllocator *pAllocator, const mString &filename)
{
  mFUNCTION_SETUP();

  mERROR_IF(pXmlReader == nullptr, mR_ArgumentNull);
  mERROR_IF(filename.hasFailed || filename.c_str() == nullptr, mR_InvalidParameter);

  size_t size = 0;
  char *text = nullptr;
  mAllocator *pTempAllocator = &mDefaultTempAllocator;

  mERROR_CHECK(mFile_ReadRaw(filename, &text, pTempAllocator, &size));
  mDEFER(mAllocator_FreePtr(pTempAllocator, &text));

  mERROR_CHECK(mXmlReader_CreateFromString(pXmlReader, pAllocator, text, size));

  mRETURN_SUCCESS();
}

mFUNCTION(mXmlReader_Destroy, IN_OUT mPtr<mXmlReader> *pXmlReader)
{
  mFUNCTION_SETUP();

  mERROR_IF(pXmlReader == nullptr, mR_ArgumentNull);
  
  mERROR_CHECK(mSharedPointer_Destroy(pXmlReader));

  mRETURN_SUCCESS();
}

mFUNCTION(mXmlReader_StepInto, mPtr<mXmlReader> &xmlReader, const mString &tag)
{
  mFUNCTION_SETUP();

  mERROR_IF(xmlReader == nullptr, mR_ArgumentNull);
  mERROR_IF(tag.hasFailed || tag.bytes <= 1 || tag.text == nullptr, mR_InvalidParameter);

  mXmlNodePosition lastNode;
  mERROR_CHECK(mQueue_PeekBack(xmlReader->currentNodeStack, &lastNode));

  mERROR_IF(lastNode.pNode->childNodes == nullptr, mR_ResourceNotFound);

  size_t index = (size_t)-1;
  
  for (auto &_element : lastNode.pNode->childNodes->Iterate())
  {
    ++index;

    if (_element.tag == tag)
      mRETURN_RESULT(mQueue_EmplaceBack(xmlReader->currentNodeStack, &_element, index));
  }

  mRETURN_RESULT(mR_ResourceNotFound);
}

mFUNCTION(mXmlReader_ExitTag, mPtr<mXmlReader> &xmlReader)
{
  mFUNCTION_SETUP();

  mERROR_IF(xmlReader == nullptr, mR_ArgumentNull);
  
  size_t count;
  mERROR_CHECK(mQueue_GetCount(xmlReader->currentNodeStack, &count));
  mERROR_IF(count <= 1, mR_ResourceStateInvalid);

  mXmlNodePosition lastNode;
  mERROR_CHECK(mQueue_PopBack(xmlReader->currentNodeStack, &lastNode));

  mRETURN_SUCCESS();
}

mFUNCTION(mXmlReader_StepIntoNext, mPtr<mXmlReader> &xmlReader)
{
  mFUNCTION_SETUP();

  mERROR_IF(xmlReader == nullptr, mR_ArgumentNull);

  size_t count;
  mERROR_CHECK(mQueue_GetCount(xmlReader->currentNodeStack, &count));
  mERROR_IF(count <= 1, mR_ResourceStateInvalid);

  mXmlNodePosition currentNode;
  mERROR_CHECK(mQueue_PopBack(xmlReader->currentNodeStack, &currentNode));

  mXmlNodePosition lastNode;
  mERROR_CHECK(mQueue_PeekBack(xmlReader->currentNodeStack, &lastNode));

  size_t startIndex = currentNode.index;
  size_t index = (size_t)-1;

  for (auto &_element : lastNode.pNode->childNodes->Iterate())
  {
    ++index;

    // Default start index is `UINT64_MAX`!
    if ((startIndex + 1) <= index && _element.tag == currentNode.pNode->tag)
        mRETURN_RESULT(mQueue_EmplaceBack(xmlReader->currentNodeStack, &_element, index));
  }

  mRETURN_RESULT(mR_ResourceNotFound);
}

mFUNCTION(mXmlReader_Visit, mPtr<mXmlReader> &xmlReader, OUT mString *pTag)
{
  mFUNCTION_SETUP();

  mERROR_IF(xmlReader == nullptr || pTag == nullptr, mR_ArgumentNull);

  mXmlNodePosition lastNode;
  mERROR_CHECK(mQueue_PeekBack(xmlReader->currentNodeStack, &lastNode));

  mERROR_IF(lastNode.pNode->childNodes == nullptr, mR_ResourceNotFound);

  mXmlNode *pChildNode;
  mERROR_CHECK(mQueue_PointerAt(lastNode.pNode->childNodes, 0, &pChildNode));
  mERROR_CHECK(mQueue_EmplaceBack(xmlReader->currentNodeStack, pChildNode, 0));

  mERROR_CHECK(mString_Create(pTag, pChildNode->tag, pTag->pAllocator));

  mRETURN_SUCCESS();
}

mFUNCTION(mXmlReader_VisitNext, mPtr<mXmlReader> &xmlReader, OUT mString *pTag)
{
  mFUNCTION_SETUP();

  mERROR_IF(xmlReader == nullptr || pTag == nullptr, mR_ArgumentNull);

  mXmlNodePosition currentNode;
  mERROR_CHECK(mQueue_PopBack(xmlReader->currentNodeStack, &currentNode));

  mXmlNodePosition lastNode;
  mERROR_CHECK(mQueue_PeekBack(xmlReader->currentNodeStack, &lastNode));

  size_t count;
  mERROR_CHECK(mQueue_GetCount(lastNode.pNode->childNodes, &count));
  mERROR_IF(count <= currentNode.index + 1, mR_ResourceNotFound);

  mXmlNode *pChildNode;
  mERROR_CHECK(mQueue_PointerAt(lastNode.pNode->childNodes, currentNode.index + 1, &pChildNode));
  mERROR_CHECK(mQueue_EmplaceBack(xmlReader->currentNodeStack, pChildNode, currentNode.index + 1));

  mERROR_CHECK(mString_Create(pTag, pChildNode->tag, pTag->pAllocator));

  mRETURN_SUCCESS();
}

mFUNCTION(mXmlReader_GetAttributeProperty, mPtr<mXmlReader> &xmlReader, const mString &attributeKey, OUT mString *pValue)
{
  mFUNCTION_SETUP();

  mERROR_IF(xmlReader == nullptr || pValue == nullptr, mR_ArgumentNull);
  mERROR_IF(attributeKey.hasFailed || attributeKey.bytes < 1 || attributeKey.text == nullptr, mR_InvalidParameter);

  size_t count = 0;
  mERROR_CHECK(mQueue_GetCount(xmlReader->currentNodeStack, &count));

  mXmlNodePosition *pLastNode = nullptr;
  mERROR_CHECK(mQueue_PointerAt(xmlReader->currentNodeStack, count - 1, &pLastNode));

  mERROR_IF(pLastNode->pNode->attributes == nullptr, mR_ResourceNotFound);

  size_t index = pLastNode->attributeIndex = (size_t)-1;

  for (const auto &_attribute : pLastNode->pNode->attributes->Iterate())
  {
    ++index;

    if (_attribute.key == attributeKey)
    {
      pLastNode->attributeIndex = index;
      mRETURN_RESULT(mString_Create(pValue, _attribute.value, pValue->pAllocator));
    }
  }

  mRETURN_RESULT(mR_ResourceNotFound);
}

mFUNCTION(mXmlReader_VisitAttribute, mPtr<mXmlReader> &xmlReader, OUT mString *pKey, OUT mString *pValue)
{
  mFUNCTION_SETUP();

  mERROR_IF(xmlReader == nullptr || pKey == nullptr || pValue == nullptr, mR_ArgumentNull);

  size_t count = 0;
  mERROR_CHECK(mQueue_GetCount(xmlReader->currentNodeStack, &count));

  mXmlNodePosition *pLastNode = nullptr;
  mERROR_CHECK(mQueue_PointerAt(xmlReader->currentNodeStack, count - 1, &pLastNode));

  mERROR_IF(pLastNode->pNode->attributes == nullptr, mR_ResourceNotFound);

  pLastNode->attributeIndex = 0;
  
  mKeyValuePair<mString, mString> *pAttribute = nullptr;
  mERROR_CHECK(mQueue_PointerAt(pLastNode->pNode->attributes, 0, &pAttribute));

  mERROR_CHECK(mString_Create(pKey, pAttribute->key, pKey->pAllocator));
  mERROR_CHECK(mString_Create(pValue, pAttribute->value, pValue->pAllocator));

  mRETURN_SUCCESS();
}

mFUNCTION(mXmlReader_VisitNextAttribute, mPtr<mXmlReader> &xmlReader, OUT mString *pKey, OUT mString *pValue)
{
  mFUNCTION_SETUP();

  mERROR_IF(xmlReader == nullptr || pKey == nullptr || pValue == nullptr, mR_ArgumentNull);

  size_t count = 0;
  mERROR_CHECK(mQueue_GetCount(xmlReader->currentNodeStack, &count));

  mXmlNodePosition *pLastNode = nullptr;
  mERROR_CHECK(mQueue_PointerAt(xmlReader->currentNodeStack, count - 1, &pLastNode));

  mERROR_IF(pLastNode->pNode->attributes == nullptr, mR_ResourceNotFound);

  mERROR_CHECK(mQueue_GetCount(pLastNode->pNode->attributes, &count));
  mERROR_IF(count <= pLastNode->attributeIndex + 1, mR_ResourceNotFound);
  ++pLastNode->attributeIndex;

  mKeyValuePair<mString, mString> *pAttribute = nullptr;
  mERROR_CHECK(mQueue_PointerAt(pLastNode->pNode->attributes, pLastNode->attributeIndex, &pAttribute));
  
  mERROR_CHECK(mString_Create(pKey, pAttribute->key, pKey->pAllocator));
  mERROR_CHECK(mString_Create(pValue, pAttribute->value, pValue->pAllocator));

  mRETURN_SUCCESS();
}

mFUNCTION(mXmlReader_GetContent, mPtr<mXmlReader> &xmlReader, OUT mString *pContent)
{
  mFUNCTION_SETUP();

  mERROR_IF(xmlReader == nullptr || pContent == nullptr, mR_ArgumentNull);

  mXmlNodePosition lastNode;
  mERROR_CHECK(mQueue_PeekBack(xmlReader->currentNodeStack, &lastNode));

  mERROR_CHECK(mString_Create(pContent, lastNode.pNode->content, pContent->pAllocator));

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mXmlReader_Destroy_Internal, IN_OUT mXmlReader *pXmlReader)
{
  mFUNCTION_SETUP();

  mERROR_IF(pXmlReader == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mQueue_Clear(pXmlReader->currentNodeStack));
  mERROR_CHECK(mDestruct(&pXmlReader->currentNodeStack));
  mERROR_CHECK(mDestruct(&pXmlReader->rootNode));

  mRETURN_SUCCESS();
}

void XMLCALL mXmlReader_StartElement_Internal(void *pUserData, const XML_Char *name, const XML_Char **pAttributes)
{
  mXmlReader *pXmlReader = reinterpret_cast<mXmlReader *>(pUserData);

  if (mFAILED(pXmlReader->parseResult))
    return;

  mXmlNodePosition lastNode;
  mXmlNode *pChildNode = nullptr;

  mERROR_CHECK_GOTO(mQueue_PeekBack(pXmlReader->currentNodeStack, &lastNode), pXmlReader->parseResult, epilogue);

  if (lastNode.pNode->childNodes == nullptr)
    mERROR_CHECK_GOTO(mQueue_Create(&lastNode.pNode->childNodes, pXmlReader->pAllocator), pXmlReader->parseResult, epilogue);

  mERROR_CHECK_GOTO(mQueue_EmplaceBack(lastNode.pNode->childNodes), pXmlReader->parseResult, epilogue);
  mERROR_CHECK_GOTO(mQueue_PointerAt(lastNode.pNode->childNodes, lastNode.pNode->childNodes->count - 1, &pChildNode), pXmlReader->parseResult, epilogue);

  mERROR_CHECK_GOTO(mString_Create(&pChildNode->tag, name, pXmlReader->pAllocator), pXmlReader->parseResult, epilogue);
  pChildNode->content.pAllocator = pXmlReader->pAllocator;

  if (pAttributes != nullptr && pAttributes[0] != nullptr)
  {
    mERROR_CHECK_GOTO(mQueue_Create(&pChildNode->attributes, pXmlReader->pAllocator), pXmlReader->parseResult, epilogue);
    
    for (size_t i = 0; pAttributes[i] != nullptr; i += 2)
    {
      mKeyValuePair<mString, mString> attribute;
      mERROR_CHECK_GOTO(mString_Create(&attribute.key, pAttributes[i], pXmlReader->pAllocator), pXmlReader->parseResult, epilogue);

      if (pAttributes[i + 1] != nullptr)
        mERROR_CHECK_GOTO(mString_Create(&attribute.value, pAttributes[i + 1], pXmlReader->pAllocator), pXmlReader->parseResult, epilogue);

      mERROR_CHECK_GOTO(mQueue_PushBack(pChildNode->attributes, std::move(attribute)), pXmlReader->parseResult, epilogue);
    }
  }

  mERROR_CHECK_GOTO(mQueue_PushBack(pXmlReader->currentNodeStack, mXmlNodePosition(pChildNode, 0)), pXmlReader->parseResult, epilogue);

epilogue:;
}

void XMLCALL mXmlReader_EndElement_Internal(void *pUserData, const XML_Char *name)
{
  mXmlReader *pXmlReader = reinterpret_cast<mXmlReader *>(pUserData);

  if (mFAILED(pXmlReader->parseResult))
    return;

  mString tag;
  mXmlNodePosition lastNode;

  mERROR_CHECK_GOTO(mString_Create(&tag, name, pXmlReader->pAllocator), pXmlReader->parseResult, epilogue);

  mERROR_CHECK_GOTO(mQueue_PeekBack(pXmlReader->currentNodeStack, &lastNode), pXmlReader->parseResult, epilogue);

  if (pXmlReader->currentNodeStack->count > 1 && lastNode.pNode->tag == tag)
    mERROR_CHECK_GOTO(mQueue_PopBack(pXmlReader->currentNodeStack, &lastNode), pXmlReader->parseResult, epilogue);

epilogue:;
}

void XMLCALL mXmlReader_HandleData_Internal(void *pUserData, const XML_Char *string, const int32_t length)
{
  mXmlReader *pXmlReader = reinterpret_cast<mXmlReader *>(pUserData);

  if (mFAILED(pXmlReader->parseResult))
    return;

  mXmlNodePosition lastNode;
  mString content;
  
  mERROR_CHECK_GOTO(mQueue_PeekBack(pXmlReader->currentNodeStack, &lastNode), pXmlReader->parseResult, epilogue);

  mERROR_CHECK_GOTO(mString_Create(&content, string, length, &mDefaultTempAllocator), pXmlReader->parseResult, epilogue);
  mERROR_CHECK_GOTO(mString_Append(lastNode.pNode->content, content), pXmlReader->parseResult, epilogue);

epilogue:;
}

//////////////////////////////////////////////////////////////////////////

struct mXmlWriter
{
  mXmlNode rootNode;
  mUniqueContainer<mQueue<mXmlNode *>> currentNodeStack;
  mAllocator *pAllocator;
};

mFUNCTION(mXmlWriter_Destroy_Internal, IN_OUT mXmlWriter *pXmlWriter);
mFUNCTION(mXmlWriter_RecursiveSerializeNode_Internal, const mXmlNode &xmlNode, IN_OUT mString &string, const size_t depth = 0);
mFUNCTION(mXmlWriter_AppendEscapedString_Internal, IN_OUT mString &string, const mString &unescapedString);

inline bool mXmlWriter_IsValidString(const mString &tag)
{
  return !(tag.hasFailed || tag.bytes <= 1 || tag.text == nullptr);
}

inline bool mXmlWriter_StringNeedsEscaping(const mString &string)
{
  for (const auto &&_char : string)
    if (_char.characterSize > 1 || !((*_char.character >= 'a' && *_char.character <= 'z') || (*_char.character >= 'A' && *_char.character <= 'Z') || (*_char.character >= '0' && *_char.character <= '9') || *_char.character >= '-' || *_char.character >= '_'))
      return true;

  return false;
}

static mString mXmlWriter_OpeningTag = "<";
static mString mXmlWriter_OpeningEndTag = "</";
static mString mXmlWriter_ClosingTag = ">";
static mString mXmlWriter_EmptyClosingTag = "/>";
static mString mXmlWriter_Space = " ";
static mString mXmlWriter_Equals = "=";
static mString mXmlWriter_Quote = "\"";

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mXmlWriter_Create, OUT mPtr<mXmlWriter> *pXmlWriter, IN mAllocator *pAllocator, const mString &rootElementTag)
{
  mFUNCTION_SETUP();

  mERROR_IF(pXmlWriter == nullptr, mR_ArgumentNull);
  mERROR_IF(!mXmlWriter_IsValidString(rootElementTag) || mXmlWriter_StringNeedsEscaping(rootElementTag), mR_InvalidParameter);

  mERROR_CHECK(mSharedPointer_Allocate<mXmlWriter>(pXmlWriter, pAllocator, [](mXmlWriter *pData) { mXmlWriter_Destroy_Internal(pData); }, 1));
  mDEFER_CALL_ON_ERROR(pXmlWriter, mSetToNullptr);

  (*pXmlWriter)->pAllocator = pAllocator;

  mERROR_CHECK(mString_Create(&(*pXmlWriter)->rootNode.tag, rootElementTag, pAllocator));
  (*pXmlWriter)->rootNode.content.pAllocator = pAllocator;

  mERROR_CHECK(mQueue_Create(&(*pXmlWriter)->currentNodeStack, pAllocator));
  mERROR_CHECK(mQueue_PushBack((*pXmlWriter)->currentNodeStack, &(*pXmlWriter)->rootNode));

  mRETURN_SUCCESS();
}

mFUNCTION(mXmlWriter_Destroy, IN_OUT mPtr<mXmlWriter> *pXmlWriter)
{
  mFUNCTION_SETUP();

  mERROR_IF(pXmlWriter == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mSharedPointer_Destroy(pXmlWriter));

  mRETURN_SUCCESS();
}

mFUNCTION(mXmlWriter_ToString, mPtr<mXmlWriter> &xmlWriter, OUT mString *pXmlString, IN mAllocator *pAllocator)
{
  mFUNCTION_SETUP();

  mERROR_IF(xmlWriter == nullptr || pXmlString == nullptr, mR_ArgumentNull);

  const char header[] = "<?xml version=\"1.0\"?>\n";

  mERROR_CHECK(mString_Create(pXmlString, header, mARRAYSIZE(header), pAllocator));
  mERROR_CHECK(mXmlWriter_RecursiveSerializeNode_Internal(xmlWriter->rootNode, *pXmlString));

  mInplaceString<2> newline;
  mERROR_CHECK(mInplaceString_Create(&newline, "\n"));
  mERROR_CHECK(mString_Append(*pXmlString, newline));

  mRETURN_SUCCESS();
}

mFUNCTION(mXmlWriter_ToFile, mPtr<mXmlWriter> &xmlWriter, const mString &filename)
{
  mFUNCTION_SETUP();

  mERROR_IF(xmlWriter == nullptr, mR_ArgumentNull);
  mERROR_IF(!mXmlWriter_IsValidString(filename), mR_ArgumentNull);

  mString xmlString;
  mERROR_CHECK(mXmlWriter_ToString(xmlWriter, &xmlString, &mDefaultTempAllocator));

  mERROR_CHECK(mFile_WriteAllText(filename, xmlString));

  mRETURN_SUCCESS();
}

mFUNCTION(mXmlWriter_BeginElement, mPtr<mXmlWriter> &xmlWriter, const mString &tag)
{
  mFUNCTION_SETUP();

  mERROR_IF(xmlWriter == nullptr, mR_ArgumentNull);
  mERROR_IF(!mXmlWriter_IsValidString(tag) || mXmlWriter_StringNeedsEscaping(tag), mR_InvalidParameter);

  mXmlNode *pLastNode = nullptr;
  mERROR_CHECK(mQueue_PeekBack(xmlWriter->currentNodeStack, &pLastNode));

  if (pLastNode->childNodes == nullptr)
    mERROR_CHECK(mQueue_Create(&pLastNode->childNodes, xmlWriter->pAllocator));

  mERROR_CHECK(mQueue_EmplaceBack(pLastNode->childNodes));

  mDEFER_ON_ERROR(
    mXmlNode node;
    mQueue_PopBack(pLastNode->childNodes, &node);
  );

  size_t count;
  mERROR_CHECK(mQueue_GetCount(pLastNode->childNodes, &count));

  mXmlNode *pNextNode = nullptr;
  mERROR_CHECK(mQueue_PointerAt(pLastNode->childNodes, count - 1, &pNextNode));

  mERROR_CHECK(mString_Create(&pNextNode->tag, tag, xmlWriter->pAllocator));
  pNextNode->content.pAllocator = xmlWriter->pAllocator;

  mERROR_CHECK(mQueue_PushBack(xmlWriter->currentNodeStack, pNextNode));

  mRETURN_SUCCESS();
}

mFUNCTION(mXmlWriter_EndElement, mPtr<mXmlWriter> &xmlWriter)
{
  mFUNCTION_SETUP();

  mERROR_IF(xmlWriter == nullptr, mR_ArgumentNull);
 
  mXmlNode *pNode = nullptr;
  mERROR_CHECK(mQueue_PopBack(xmlWriter->currentNodeStack, &pNode));

  mRETURN_SUCCESS();
}

mFUNCTION(mXmlWriter_AddAttribute, mPtr<mXmlWriter> &xmlWriter, const mString &key, const mString &value)
{
  mFUNCTION_SETUP();

  mERROR_IF(xmlWriter == nullptr, mR_ArgumentNull);
  mERROR_IF(!mXmlWriter_IsValidString(key) || !mXmlWriter_IsValidString(value) || mXmlWriter_StringNeedsEscaping(key), mR_InvalidParameter);

  mXmlNode *pLastNode = nullptr;
  mERROR_CHECK(mQueue_PeekBack(xmlWriter->currentNodeStack, &pLastNode));

  if (pLastNode->attributes == nullptr)
    mERROR_CHECK(mQueue_Create(&pLastNode->attributes, xmlWriter->pAllocator));

  mKeyValuePair<mString, mString> attribute;
  mERROR_CHECK(mString_Create(&attribute.key, key, xmlWriter->pAllocator));
  mERROR_CHECK(mString_Create(&attribute.value, value, xmlWriter->pAllocator));

  mERROR_CHECK(mQueue_PushBack(pLastNode->attributes, std::move(attribute)));

  mRETURN_SUCCESS();
}

mFUNCTION(mXmlWriter_AddContent, mPtr<mXmlWriter> &xmlWriter, const mString &content)
{
  mFUNCTION_SETUP();

  mERROR_IF(xmlWriter == nullptr, mR_ArgumentNull);
  mERROR_IF(!mXmlWriter_IsValidString(content), mR_InvalidParameter);

  mXmlNode *pLastNode = nullptr;
  mERROR_CHECK(mQueue_PeekBack(xmlWriter->currentNodeStack, &pLastNode));

  mERROR_CHECK(mString_Append(pLastNode->content, content));

  mRETURN_SUCCESS();
}

//////////////////////////////////////////////////////////////////////////

mFUNCTION(mXmlWriter_Destroy_Internal, IN_OUT mXmlWriter *pXmlWriter)
{
  mFUNCTION_SETUP();

  mERROR_IF(pXmlWriter == nullptr, mR_ArgumentNull);

  mERROR_CHECK(mDestruct(&pXmlWriter->currentNodeStack));
  mERROR_CHECK(mDestruct(&pXmlWriter->rootNode));

  mRETURN_SUCCESS();
}

mFUNCTION(mXmlWriter_RecursiveSerializeNode_Internal, const mXmlNode &xmlNode, IN_OUT mString &string, const size_t depth /* = 0 */)
{
  mFUNCTION_SETUP();

  mERROR_CHECK(mString_Append(string, mXmlWriter_OpeningTag));
  mERROR_CHECK(mString_Append(string, xmlNode.tag));

  if (xmlNode.attributes != nullptr)
  {
    for (const auto &_attribute : xmlNode.attributes->Iterate())
    {
      mERROR_CHECK(mString_Append(string, mXmlWriter_Space));
      mERROR_CHECK(mString_Append(string, _attribute.key));
      mERROR_CHECK(mString_Append(string, mXmlWriter_Equals));
      mERROR_CHECK(mString_Append(string, mXmlWriter_Quote));
      mERROR_CHECK(mXmlWriter_AppendEscapedString_Internal(string, _attribute.value));
      mERROR_CHECK(mString_Append(string, mXmlWriter_Quote));
    }
  }

  const bool hasChildNodes = xmlNode.childNodes != nullptr && xmlNode.childNodes->count > 0;

  if (xmlNode.content.bytes <= 1 && !hasChildNodes)
  {
    mERROR_CHECK(mString_Append(string, mXmlWriter_EmptyClosingTag));
    mRETURN_SUCCESS();
  }

  mERROR_CHECK(mString_Append(string, mXmlWriter_ClosingTag));
  mERROR_CHECK(mXmlWriter_AppendEscapedString_Internal(string, xmlNode.content));

  if (hasChildNodes)
    for (const auto &_childNode : xmlNode.childNodes->Iterate())
      mERROR_CHECK(mXmlWriter_RecursiveSerializeNode_Internal(_childNode, string, depth + 1));

  mERROR_CHECK(mString_Append(string, mXmlWriter_OpeningEndTag));
  mERROR_CHECK(mString_Append(string, xmlNode.tag));
  mERROR_CHECK(mString_Append(string, mXmlWriter_ClosingTag));

  mRETURN_SUCCESS();
}

mFUNCTION(mXmlWriter_AppendEscapedString_Internal, IN_OUT mString &string, const mString &unescapedString)
{
  mFUNCTION_SETUP();

  bool needsEscaping = false;

  for (const auto &&_char : unescapedString)
  {
    const char asciiChar = *_char.character;
    
    if (_char.characterSize > 1 || asciiChar < 32 || asciiChar > 126 || asciiChar == '&' || asciiChar == '<' || asciiChar == '>' || asciiChar == '\'' || asciiChar == '\"')
    {
      needsEscaping = true;
      break;
    }
  }

  if (!needsEscaping)
  {
    mERROR_CHECK(mString_Append(string, unescapedString));
  }
  else
  {
    constexpr size_t size = 16;
    mALIGN(8) mInplaceString<size> s;

    for (const auto &&_char : unescapedString)
    {
      const char asciiChar = *_char.character;

      if (_char.characterSize > 1 || asciiChar < 32 || asciiChar > 126)
      {
        s.bytes = snprintf(s.text, size, "&#x%" PRIX32 ";", _char.codePoint);
        mERROR_IF(s.bytes <= 0, mR_InternalError);
        s.count = ++s.bytes;
      }
      else
      {
        switch (asciiChar)
        {
        case '&':
        {
          mALIGN(8) const char symbol[8] = "&amp;";
          s.bytes = s.count = mARRAYSIZE("&amp;");
          *reinterpret_cast<uint64_t *>(s.text) = *reinterpret_cast<const uint64_t *>(symbol);
          break;
        }

        case '<':
        {
          mALIGN(8) const char symbol[8] = "&lt;";
          s.bytes = s.count = mARRAYSIZE("&lt;");
          *reinterpret_cast<uint64_t *>(s.text) = *reinterpret_cast<const uint64_t *>(symbol);
          break;
        }

        case '>':
        {
          mALIGN(8) const char symbol[8] = "&gt;";
          s.bytes = s.count = mARRAYSIZE("&gt;");
          *reinterpret_cast<uint64_t *>(s.text) = *reinterpret_cast<const uint64_t *>(symbol);
          break;
        }

        case '\'':
        {
          mALIGN(8) const char symbol[8] = "&apos;";
          s.bytes = s.count = mARRAYSIZE("&apos;");
          *reinterpret_cast<uint64_t *>(s.text) = *reinterpret_cast<const uint64_t *>(symbol);
          break;
        }

        case '\"':
        {
          mALIGN(8) const char symbol[8] = "&quot;";
          s.bytes = s.count = mARRAYSIZE("&quot;");
          *reinterpret_cast<uint64_t *>(s.text) = *reinterpret_cast<const uint64_t *>(symbol);
          break;
        }

        default:
        {
          s.bytes = s.count = 2;
          s.text[0] = asciiChar;
          s.text[1] = '\0';
          break;
        }
        }
      }

      mERROR_CHECK(mString_Append(string, s));
    }
  }

  mRETURN_SUCCESS();
}
