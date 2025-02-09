from operator import itemgetter
from typing import Any, Dict, Callable

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, format_document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import get_buffer_string
from langchain.prompts.prompt import PromptTemplate

# Prompt definitions
condense_question = (
    "Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.\n\n"
    "Chat History:\n{chat_history}\n\n"
    "Follow Up Input: {question}\n"
    "Standalone question:"
)
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question)

answer_template = (
    "### Instruction:\n"
    "You're a helpful research assistant, who answers questions based on provided research in a clear way and easy-to-understand way.\n"
    "If there is no research, or the research is irrelevant to answering the question, simply reply that you can't answer.\n"
    "Please reply with just the detailed answer and your sources. If you're unable to answer the question, do not list sources\n\n"
    "## Research:\n{context}\n\n"
    "## Question:\n{question}"
)
ANSWER_PROMPT = ChatPromptTemplate.from_template(answer_template)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(
    template="Source Document: {source}, Page {page}:\n{page_content}"
)

def _combine_documents(docs: Any, document_prompt: PromptTemplate = DEFAULT_DOCUMENT_PROMPT, document_separator: str = "\n\n") -> str:
    """
    Formats and combines document data from a list of docs.
    """
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

# Memory for conversation
memory: ConversationBufferMemory = ConversationBufferMemory(
    return_messages=True, output_key="answer", input_key="question"
)

def getStreamingChain(question: str, memory: ConversationBufferMemory, llm: Any, db: Any) -> Any:
    """
    Creates and streams the answer chain for a given question.
    """
    retriever = db.as_retriever(search_kwargs={"k": 10})
    
    # Pipeline segment for loading chat history from memory.
    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(
            lambda x: "\n".join(
                [f"{item['role']}: {item['content']}" for item in x["memory"]]
            )
        ),
    )

    # Pipeline segment for generating a standalone question.
    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
        }
        | CONDENSE_QUESTION_PROMPT
        | llm  # Process using the LLM
    }

    # Pipeline segment for retrieving relevant documents.
    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"],
    }

    # Pipeline segment for preparing the final inputs for the answer prompt.
    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"]),
        "question": itemgetter("question"),
    }

    # Final answer pipeline
    answer_seg = final_inputs | ANSWER_PROMPT | llm

    # Compose final chain
    final_chain = loaded_memory | standalone_question | retrieved_documents | answer_seg

    try:
        return final_chain.stream({"question": question, "memory": memory})
    except Exception as e:
        # Log or handle exception as needed
        raise RuntimeError(f"Error during chain execution: {e}")

def getChatChain(llm: Any, db: Any) -> Callable[[str], None]:
    """
    Creates a chat chain that processes a question and updates conversation memory.
    """
    retriever = db.as_retriever(search_kwargs={"k": 10})
    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
    )

    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        }
        | CONDENSE_QUESTION_PROMPT
        | llm
    }

    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"],
    }

    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"]),
        "question": itemgetter("question"),
    }

    answer_seg = {
        "answer": final_inputs
        | ANSWER_PROMPT
        | llm.with_config(callbacks=[StreamingStdOutCallbackHandler()]),
        "docs": itemgetter("docs"),
    }

    final_chain = loaded_memory | standalone_question | retrieved_documents | answer_seg

    def chat(question: str) -> None:
        inputs = {"question": question}
        try:
            result = final_chain.invoke(inputs)
            memory.save_context(inputs, {"answer": result["answer"]})
        except Exception as e:
            raise RuntimeError(f"Chat chain error: {e}")

    return chat