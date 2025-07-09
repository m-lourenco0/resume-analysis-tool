# graph.py
import os
from typing import Any, Dict, List

from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from app.services.utils import format_prompt_string


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
    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    retrieved_context = "\n\n---\n\n".join([m.content for m in tool_messages])

    if not retrieved_context:
        # Handle cases where the agent decides not to call the tool at all
        retrieved_context = "No specific context was retrieved from the resume. The analysis will be based on the initial prompt and job description only."

    prompt_text = f"""
## Persona
You are a highly analytical and evidence-based Career Coach and ATS Analyst. Your analysis must be objective, critical, and directly tied to the context provided.

## Core Task
Analyze the Retrieved Resume Context against the Job Description and generate a single, clean JSON object with no extraneous text or explanations outside of the JSON structure. Base your entire analysis *strictly* on the provided context. Do not invent skills, experiences, or formatting details.

## Important Context Limitation
The 'Retrieved Resume Context' consists of text snippets that matched specific search queries. It is **NOT** the full, formatted resume document. Therefore, your analysis of ATS friendliness and structure must be based only on the parseable text and keywords present, not on visual layout, fonts, or columns, which you cannot see.

## Input Data

**Job Description:**
---
{state["job_description"]}
---

**Retrieved Resume Context:**
---
{retrieved_context}
---

## JSON Output Schema and Instructions

Your response MUST be a single JSON object matching this schema:

```json
{{
  "match_score": "integer",
  "summary": "string",
  "ats_friendliness_score": "integer",
  "ats_friendliness_feedback": "string",
  "structure_feedback": "string",
  "compatible_keywords": "list[string]",
  "missing_keywords": "list[string]",
  "suggestions": "list[string]"
}}
```

Field-by-Field Instructions:
match_score: (Integer) Calculate a holistic match score from 0 to 100 based on the evidence. The score should directly reflect the balance of compatible vs. missing keywords and experience.

95-100: Perfect match on paper.

80-94: Strong match.

60-79: Moderate match.

<60: Weak match.

summary: (String) A concise, professional summary (2-4 sentences) of the candidate's fit. Begin by stating the match level (e.g., 'This candidate is a strong fit...'). Directly reference the key strengths and most significant gaps identified in your keyword analysis.

ats_friendliness_score: (Integer) Score from 0-100. Base this score only on keyword alignment and the apparent presence of standard, parseable sections within the provided text snippets.

ats_friendliness_feedback: (String) Provide feedback on keyword relevance and text structure, acknowledging the limitation that you cannot see the full document. Example: 'Based on the text provided, the resume contains many relevant keywords. To ensure ATS compatibility, the original document should use standard section headings (like 'Experience') and avoid tables or images.'

structure_feedback: (String) Evaluate the clarity and impact of the language for a human recruiter, based only on the retrieved text. Comment on the use of action verbs and quantifiable results. Example: 'The experience descriptions are clear, but would be more powerful if achievements were quantified (e.g., 'managed a team' vs. 'managed a team of 5 engineers').'

compatible_keywords: (List of Strings) Extract a list of skills, technologies, and qualifications from the job description that are explicitly present in the Retrieved Resume Context.

missing_keywords: (List of Strings) Identify crucial skills, technologies, and qualifications from the job description that are NOT found in the Retrieved Resume Context.

suggestions: (List of Strings) Provide a list of concrete, actionable suggestions for improvement, with each suggestion as a separate string. These suggestions must directly address the findings in missing_keywords and structure_feedback. Example: 'Incorporate the terms 'SaaS' and 'CI/CD' to better align with the job description.' or 'Revise the bullet point on 'process optimization' to include the specific percentage of improvement achieved.'
"""

    analysis = llm_with_tool.invoke(prompt_text)
    return {"analysis_result": analysis.dict()}


def router(state: AgenticRagState) -> str:
    """
    This function determines the next step for the agent.

    If the last message from the agent contains tool calls, it routes to the 'retriever'.
    Otherwise, it means the agent has finished its research and is ready to perform
    the final analysis, so it routes to the 'analyzer'.
    """
    print("--- Router ---")
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        print("Decision: Call retriever")
        return "retriever"
    else:
        print("Decision: Proceed to analyzer")
        return "analyzer"


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
        router,
        {
            "retriever": "retriever",
            "analyzer": "analyzer",
        },
    )
    workflow.add_edge("retriever", "agent")
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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Create the tool that the agent will use
    retriever_tool = create_retriever_tool(
        retriever,
        "search_resume",
        "Searches the candidate's resume to find specific evidence, such as skills, experiences, projects, and qualifications, that directly match the requirements outlined in the job description. Use this tool to gather concrete examples and keywords from the resume.",
    )

    graph = build_analysis_graph(retriever_tool, api_key)

    user_message = f"""
## Persona
You are a methodical AI assistant acting as a Resume Data Analyst.

## Primary Goal
Your sole objective in this step is to systematically gather all relevant facts from a candidate's resume that correspond to the requirements in a provided job description. You will use a search tool for this task.

## Context
- **Job Description:** The source of truth for all requirements.
- **Available Tool:** You have access to a single tool: `search_resume(query: str)`.

## Workflow & Strict Instructions

1.  **Internal Analysis First:** Before using any tools, perform a silent, internal analysis of the job description below. Create a mental checklist of all key requirements, such as:
    * Technical Skills (e.g., specific languages, software, frameworks)
    * Years and Types of Experience (e.g., "5+ years", "management experience")
    * Educational Background (e.g., degrees, certifications)
    * Key Responsibilities (e.g., "client-facing", "budget management")

2.  **Execute Search Plan:** Based on your internal checklist, methodically use the `search_resume` tool to find evidence for each requirement.
    * **Be Systematic:** Query for each distinct requirement or group of related requirements.
    * **Be Efficient:** Combine searches for related skills into a single tool call where logical (e.g., `search_resume(query="Python, Django, Flask")`).
    * **Be Thorough:** Do not stop until you have attempted to find evidence for **all** the key requirements you identified in your initial analysis.

3.  **Completion and Final Output:**
    * Once your systematic search is complete, you MUST stop calling tools.
    * Your final and ONLY response for this task must be the exact sentence: **"I have gathered sufficient information to proceed with the analysis."** Do not add any other text or explanation.

## Job Description to Analyze
---
{job_description}
---
"""

    inputs = {"messages": [("user", user_message)], "job_description": job_description}

    # Invoke the graph
    final_state = graph.invoke(inputs)

    if not final_state.get("analysis_result"):
        raise Exception("Graph did not produce a final analysis.")

    result = final_state["analysis_result"]
    result["prompt"] = format_prompt_string(
        analysis_result=result, job_description=job_description
    )

    return result
