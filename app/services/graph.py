# graph.py

import os
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, END
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition


# --- 1. Define the final output schema ---
class ResumeAnalysis(BaseModel):
    """The result of a resume analysis against a job description."""

    match_score: float = Field(
        ...,
        description="The overall match score from 0 to 100, representing the resume's alignment with the job description.",
    )
    summary: str = Field(
        ...,
        description="A brief summary of how the resume matches the job description, based on the retrieved context.",
    )
    compatible_keywords: List[str] = Field(
        ...,
        description="A list of keywords from the job description that are present in the retrieved resume context.",
    )
    missing_keywords: List[str] = Field(
        ...,
        description="A list of important keywords from the job description that appear to be missing from the retrieved resume context.",
    )
    suggestions: List[str] = Field(
        ...,
        description="Actionable suggestions for how the candidate could improve their resume for this specific job, based on the retrieved context.",
    )


# --- 2. Define the Graph State ---
# We extend MessagesState to hold our final analysis result.
class AgenticRagState(MessagesState):
    job_description: str
    analysis_result: Dict[str, Any]


# --- 3. Define the Nodes of the graph ---
def agent_node(state: AgenticRagState, llm):
    """
    The "brain" of the agent. It decides what to do next based on the conversation history.
    """
    print("--- Agent Node ---")
    return {"messages": [llm.invoke(state["messages"])]}


def analyzer_node(state: AgenticRagState, llm_with_tool):
    """
    This node takes the retrieved context and performs the final analysis.
    """
    print("--- Analyzer Node ---")

    # The retrieved context is in the last message, which is a ToolMessage
    retrieved_context = state["messages"][-1].content

    # We create a new prompt that includes the retrieved context
    prompt_text = (
        "Based on the following retrieved context from a candidate's resume, "
        "and the original job description, please perform a detailed analysis. "
        f"\n\n**Job Description:**\n{state['job_description']}"
        f"\n\n**Retrieved Resume Context:**\n{retrieved_context}"
    )

    # We invoke the structured LLM to get our final JSON output
    analysis = llm_with_tool.invoke(prompt_text)

    return {"analysis_result": analysis.dict()}


# --- 4. Build the Graph ---


def build_analysis_graph(retriever_tool, api_key: str):
    """Builds the Agentic LangGraph for the resume analysis."""

    # The agent's brain: an LLM that can call the retriever tool
    agent_llm = ChatOpenAI(
        model="gpt-4.1-mini", temperature=0, api_key=api_key
    ).bind_tools([retriever_tool])

    # The analyzer's brain: an LLM that produces the final structured output
    analyzer_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=api_key)
    structured_analyzer_llm = analyzer_llm.with_structured_output(ResumeAnalysis)

    workflow = StateGraph(AgenticRagState)

    # Define the nodes
    workflow.add_node("agent", lambda state: agent_node(state, agent_llm))
    workflow.add_node("retriever", ToolNode([retriever_tool]))
    workflow.add_node(
        "analyzer", lambda state: analyzer_node(state, structured_analyzer_llm)
    )

    # Define the edges
    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "retriever", END: END},
    )
    workflow.add_edge("retriever", "analyzer")
    workflow.add_edge("analyzer", END)

    return workflow.compile()


# --- 5. Create a convenience function to run the graph ---


def run_analysis_graph(
    api_key: str, job_description: str, resume_text: str
) -> Dict[str, Any]:
    """
    Initializes and runs the agentic analysis graph with the given inputs.
    """
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OpenAI API key is required. Please provide it in the form or set the OPENAI_API_KEY environment variable."
        )

    # --- Vector Store and Retriever Creation ---
    # This happens for every request, creating a temporary, in-memory knowledge base from the resume.
    print("--- Creating In-Memory Vector Store for Resume ---")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=100
    )
    doc_splits = text_splitter.split_documents([Document(page_content=resume_text)])

    embeddings = OpenAIEmbeddings(api_key=api_key)
    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits, embedding=embeddings
    )
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )  # Retrieve top 4 relevant chunks

    # Create the tool that the agent will use
    retriever_tool = create_retriever_tool(
        retriever,
        "search_resume",
        "Search the candidate's resume for information relevant to the job description.",
    )
    # --- End of Vector Store Creation ---

    graph = build_analysis_graph(retriever_tool, api_key)

    user_message = (
        "Please analyze the candidate's resume against the provided job description. "
        "Use the `search_resume` tool to find relevant skills, experiences, and keywords from the resume. "
        "After gathering sufficient information, provide a detailed analysis."
    )

    inputs = {"messages": [("user", user_message)], "job_description": job_description}

    # Invoke the graph
    final_state = graph.invoke(inputs)

    if not final_state.get("analysis_result"):
        raise Exception("Graph did not produce a final analysis.")

    return final_state["analysis_result"]
