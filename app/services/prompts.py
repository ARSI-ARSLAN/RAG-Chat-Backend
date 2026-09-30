"""Prompt construction - kept separate so it is easy to tweak and test."""
from app.llm.base import Messages
from app.schemas import ChatMessage
from app.rag.vector_store import RetrievedChunk

CONTEXT_MARKER = "CONTEXT:\n"

SYSTEM_TEMPLATE = (
    "You are a helpful assistant that answers questions using ONLY the context below.\n"
    "- If the context does not contain the answer, say you don't know based on the "
    "available documents. Do not invent facts.\n"
    "- Be concise. Mention the source file name in square brackets, e.g. [handbook.md].\n\n"
    + CONTEXT_MARKER
    + "{context}"
)


def format_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(no relevant documents found)"
    return "\n\n---\n\n".join(f"[source: {c.source}]\n{c.text}" for c in chunks)


def build_messages(
    question: str, chunks: list[RetrievedChunk], history: list[ChatMessage]
) -> Messages:
    messages: Messages = [
        {"role": "system", "content": SYSTEM_TEMPLATE.format(context=format_context(chunks))}
    ]
    messages += [{"role": m.role, "content": m.content} for m in history]
    messages.append({"role": "user", "content": question})
    return messages
