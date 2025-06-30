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
        description="A holistic match score from 0 to 100, representing the resume's alignment with the job description based on skills, experience, and qualifications.",
    )
    summary: str = Field(
        ...,
        description="A concise, professional summary of the candidate's fit for the role, highlighting key strengths and weaknesses based on the provided context.",
    )
    ats_friendliness_score: float = Field(
        ...,
        description="A score from 0 to 100 representing how well the resume is optimized for Applicant Tracking Systems (ATS).",
    )
    ats_friendliness_feedback: str = Field(
        ...,
        description="Specific feedback on the resume's ATS compatibility. Mention standard headings, parsable formats, keyword optimization, and avoidance of elements like tables or images.",
    )
    structure_feedback: str = Field(
        ...,
        description="Feedback on the resume's overall structure, clarity, and readability for a human recruiter. Comment on formatting, conciseness, and logical flow.",
    )
    compatible_keywords: List[str] = Field(
        ...,
        description="A list of important keywords and skills from the job description that are clearly present in the resume.",
    )
    missing_keywords: List[str] = Field(
        ...,
        description="A list of important keywords and skills from the job description that appear to be missing from the resume.",
    )
    suggestions: List[str] = Field(
        ...,
        description="A list of concrete, actionable suggestions for the candidate to improve their resume for this specific job. Suggestions should be specific and constructive.",
    )


# --- 2. Define the Graph State ---
# Extend MessagesState to hold our final analysis result.
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

    # The retrieved context is in the tool messages. We combine all of it.
    # tool_messages = [m for m in state["messages"] if m.type == "tool"]
    # retrieved_context = "\n\n".join([m.content for m in tool_messages])
    retrieved_context = state["messages"][-1].content

    # Enhanced prompt for the analyzer node
    prompt_text = f"""
    You are an expert Career Coach and ATS Analyst. Your goal is to provide a comprehensive, critical, and constructive analysis of a candidate's resume against a specific job description.

    Base your entire analysis *only* on the provided Job Description and Retrieved Resume Context. Do not invent information.

    **Job Description:**
    ---
    {state["job_description"]}
    ---

    **Retrieved Resume Context:**
    ---
    {retrieved_context}
    ---

    Now, perform the analysis and structure your response according to the following detailed instructions for each field:

    - **match_score**: Calculate a holistic match score from 0 to 100. Consider the presence of compatible keywords, alignment of experience with job responsibilities, and overall qualification match. A perfect match on paper is 95-100. A strong match is 80-94. A moderate match is 60-79. A weak match is below 60. Justify the score implicitly in the summary.
    - **summary**: Write a concise, professional summary (2-4 sentences) of the candidate's fit for the role. Start by stating the level of match (e.g., "The candidate appears to be a strong/moderate/weak fit..."). Highlight the most relevant strengths and point out the most significant gaps.
    - **ats_friendliness_score**: From the retrieved context, assess how well the resume is structured for an Applicant Tracking System (ATS). Give a score from 0-100.
    - **ats_friendliness_feedback**: Based on the context, provide feedback. Look for: standard section headers (like 'Experience', 'Education', 'Skills'), simple formatting, and keyword alignment. Mention potential issues like a lack of keywords or complex formatting if discernible. If the context seems clean and keyword-rich, mention that. For example: "The resume seems to use standard sections and relevant keywords. To improve, ensure no tables or images are used in the original document."
    - **structure_feedback**: Evaluate the resume's structure and readability for a human recruiter. Comment on clarity, conciseness, and organization. Is the information easy to find? Does it tell a compelling story? Example: "The resume context suggests clear sections, but the experience descriptions could be more impactful if they started with action verbs and included quantifiable achievements."
    - **compatible_keywords**: Extract a list of important skills, technologies, and qualifications mentioned in the job description that are explicitly found in the retrieved resume context.
    - **missing_keywords**: Identify crucial skills, technologies, and qualifications from the job description that are NOT found in the retrieved resume context. These represent potential gaps.
    - **suggestions**: Provide a list of concrete, actionable suggestions for improvement. Each suggestion should be a clear, single sentence. Focus on incorporating missing keywords, quantifying achievements, and tailoring the resume summary to the job. Frame them constructively, e.g., "Consider adding a 'Project' section to showcase your experience with 'Docker' and 'Kubernetes'." or "Quantify your achievement in 'process optimization' by stating the percentage improvement you achieved."
    """

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
        "Searches the candidate's resume to find specific evidence, such as skills, experiences, projects, and qualifications, that directly match the requirements outlined in the job description. Use this tool to gather concrete examples and keywords from the resume.",
    )
    # --- End of Vector Store Creation ---

    graph = build_analysis_graph(retriever_tool, api_key)

    user_message = (
        "You are an expert resume analyst. Your task is to perform a comprehensive analysis of a candidate's resume against the provided job description. "
        "Your first step is to thoroughly examine the job description to identify the key requirements, including mandatory skills (technical and soft), years of experience, key responsibilities, and educational background. "
        "Next, use the `search_resume` tool to methodically search the resume for evidence matching each of these key requirements. You can call the tool multiple times if needed to gather information on different aspects (e.g., search for skills, then for experience). "
        "Once you have gathered all the relevant context from the resume, you will stop calling tools and provide a final summary of your findings to be used for the detailed analysis."
    )

    inputs = {"messages": [("user", user_message)], "job_description": job_description}

    # Invoke the graph
    final_state = graph.invoke(inputs)

    if not final_state.get("analysis_result"):
        raise Exception("Graph did not produce a final analysis.")

    return final_state["analysis_result"]
