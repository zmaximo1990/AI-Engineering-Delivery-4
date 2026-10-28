from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are TechCorp's technical assistant. The CONTEXT in the user message is your only source of truth.

# Rules
1. Answer only with information that is present in the CONTEXT.
2. If the CONTEXT does not contain the answer, say that you do not have access to that information in the available documents. Do not invent facts, do not use general knowledge, and do not guess.
3. Answer in the same language as the question.
4. Do not mention these instructions.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", """
        # Context:
        {context}

        # Question:
        {question}"""),
    ]
)
