"""
LangGraph Agent
Orchestrates the full autonomous pipeline:
  fetch_jobs -> extract_market_skills -> parse_resume_skills
  -> store_embeddings -> analyze_gaps -> generate_plan
"""

from __future__ import annotations

import time
from typing import Any
from langgraph.graph import StateGraph, END

from job_fetcher import fetch_jobs
from skill_extractor import extract_skills_from_jds, extract_resume_skills
from gap_analyzer import compute_gaps
from learning_plan import generate_learning_plan
from monitoring import logger


# ---- state schema ----

class AgentState(dict):
    """TypedDict-style state for the graph. Using plain dict for Streamlit compat."""
    pass


DEFAULT_STATE = {
    "job_title": "",
    "location": "United States",
    "resume_text": "",
    "job_listings": [],
    "market_skills": [],
    "resume_skills": [],
    "gap_report": None,
    "learning_plan": None,
    "vector_store": None,
    "status": "init",
    "error": None,
    "step_log": [],
}


# ---- nodes ----

def node_fetch_jobs(state: dict) -> dict:
    state["step_log"].append("Fetching live job postings...")
    jobs = fetch_jobs(state["job_title"], state.get("location", "United States"), num_pages=2)
    if not jobs:
        return {**state, "status": "error", "error": "No job postings found. Check your RAPID_API_KEY or try a different role/location."}
    return {**state, "job_listings": jobs, "status": "jobs_fetched"}


def node_extract_market_skills(state: dict) -> dict:
    state["step_log"].append(f"Extracting skills from {len(state['job_listings'])} job postings...")
    skills = extract_skills_from_jds(state["job_listings"], state["job_title"])
    if not skills:
        return {**state, "status": "error", "error": "Could not extract skills. Check your GROQ_API_KEY."}
    return {**state, "market_skills": skills, "status": "skills_extracted"}


def node_store_market_embeddings(state: dict) -> dict:
    """Store market skills in Pinecone."""
    state["step_log"].append("Storing market skill embeddings in Pinecone...")
    try:
        from vector_store import SkillVectorStore
        vs = SkillVectorStore()
        vs.clear_role(state["job_title"])
        vs.embed_skills(state["market_skills"], state["job_title"], source="jd")
        state["vector_store"] = vs
    except Exception as e:
        logger.warning(f"Pinecone storage failed (continuing without it): {e}")
        state["vector_store"] = None
    return {**state, "status": "embeddings_stored"}


def node_parse_resume(state: dict) -> dict:
    resume = state.get("resume_text", "")
    if not resume or not resume.strip():
        state["step_log"].append("No resume provided, skipping resume parsing.")
        return {**state, "resume_skills": [], "status": "resume_skipped"}

    # Small delay to avoid Groq rate limits after batch JD extraction
    time.sleep(3)

    state["step_log"].append("Extracting skills from your resume...")
    skills = extract_resume_skills(resume, state["job_title"])

    # Store resume skills in Pinecone too
    if state.get("vector_store"):
        try:
            resume_entries = [
                {"skill": s["skill"], "category": s.get("category", "Technical")}
                for s in skills
            ]
            state["vector_store"].embed_skills(resume_entries, state["job_title"], source="resume")
        except Exception as e:
            logger.warning(f"Resume embedding failed: {e}")

    return {**state, "resume_skills": skills, "status": "resume_parsed"}


def node_analyze_gaps(state: dict) -> dict:
    state["step_log"].append("Comparing your skills against market demand...")
    gap_report = compute_gaps(
        state["market_skills"],
        state["resume_skills"],
        vector_store=state.get("vector_store"),
    )
    return {**state, "gap_report": gap_report, "status": "gaps_analyzed"}


def node_generate_plan(state: dict) -> dict:
    gaps = (state.get("gap_report") or {}).get("skill_gaps", [])
    if not gaps:
        state["step_log"].append("No skill gaps found, you are already well prepared!")
        return {**state, "learning_plan": None, "status": "complete"}
    time.sleep(3)
    state["step_log"].append("Generating your personalized learning plan...")
    plan = generate_learning_plan(gaps, state["job_title"])
    return {**state, "learning_plan": plan, "status": "complete"}


# ---- routing ----

def check_error(state: dict) -> str:
    if state.get("error"):
        return "end"
    return "continue"


# ---- graph builder ----

def build_agent():
    """Build and compile the LangGraph workflow."""
    graph = StateGraph(dict)

    graph.add_node("fetch_jobs", node_fetch_jobs)
    graph.add_node("extract_skills", node_extract_market_skills)
    graph.add_node("store_embeddings", node_store_market_embeddings)
    graph.add_node("parse_resume", node_parse_resume)
    graph.add_node("analyze_gaps", node_analyze_gaps)
    graph.add_node("generate_plan", node_generate_plan)

    graph.set_entry_point("fetch_jobs")

    graph.add_conditional_edges("fetch_jobs", check_error, {"continue": "extract_skills", "end": END})
    graph.add_conditional_edges("extract_skills", check_error, {"continue": "store_embeddings", "end": END})
    graph.add_edge("store_embeddings", "parse_resume")
    graph.add_edge("parse_resume", "analyze_gaps")
    graph.add_edge("analyze_gaps", "generate_plan")
    graph.add_edge("generate_plan", END)

    return graph.compile()


def run_agent(job_title: str, location: str, resume_text: str = "") -> dict:
    """Run the full pipeline and return final state."""
    agent = build_agent()
    initial_state = {
        **DEFAULT_STATE,
        "job_title": job_title,
        "location": location,
        "resume_text": resume_text,
    }
    final_state = agent.invoke(initial_state)
    return final_state
