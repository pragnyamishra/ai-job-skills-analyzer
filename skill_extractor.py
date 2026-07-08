"""
Skill Extractor
Uses Groq API (Llama 3.3 70B) with few-shot and chain-of-thought prompting
to extract, categorize, and normalize technical skills from text.
"""

import json
import os
import time
import requests
from monitoring import monitor, logger

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"


def _call_groq(system_prompt: str, user_prompt: str, max_tokens: int = 3000, temperature: float = 0.2) -> dict | None:
    """Shared Groq caller with monitoring baked in."""
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        logger.error("GROQ_API_KEY not set")
        return None

    start = time.time()
    try:
        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=60,
        )
        latency = (time.time() - start) * 1000

        if resp.status_code != 200:
            logger.error(f"Groq API {resp.status_code}: {resp.text[:300]}")
            return None

        result = resp.json()
        usage = result.get("usage", {})

        monitor.track_llm_call(
            func_name="groq_call",
            model=MODEL,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=latency,
        )

        text = result["choices"][0]["message"]["content"].strip()
        # Strip markdown fences
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        return json.loads(text)

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
    except Exception as e:
        logger.error(f"Groq call error: {e}")
    return None


# ---- public functions ----


def extract_skills_from_jds(job_listings: list[dict], job_title: str) -> list[dict]:
    """Extract and aggregate skills from a batch of job descriptions.

    Uses few-shot examples and chain-of-thought in the prompt so the model
    normalizes skill names (e.g. 'pytorch' and 'torch' both map to 'PyTorch').
    """

    if not job_listings:
        return []

    # Process in batches of 10
    all_skills: dict[str, dict] = {}

    for batch_start in range(0, len(job_listings), 10):
        batch = job_listings[batch_start : batch_start + 10]

        jobs_text = "\n\n---\n\n".join(
            f"Job: {j['title']} at {j['company']}\n{j['description']}" for j in batch
        )

        prompt = f"""Analyze these {job_title} job postings. Extract every technical skill, tool, framework, 
language, platform, and methodology mentioned.

Think step by step:
1. Read each job description carefully
2. Identify ALL technical terms, tools, languages, frameworks, certifications
3. Normalize names (e.g. 'pytorch', 'torch', 'PyTorch' all become 'PyTorch')
4. Count how many postings mention each skill
5. Rate importance: High if mentioned in 60%+ of postings, Medium if 30-60%, Low if under 30%

Here is an example of good output for a "Data Engineer" query:
[
    {{"skill": "Python", "category": "Programming Language", "job_count": 5, "importance": "High"}},
    {{"skill": "Apache Spark", "category": "Framework", "job_count": 3, "importance": "Medium"}},
    {{"skill": "dbt", "category": "Tool", "job_count": 2, "importance": "Low"}}
]

Now analyze these postings:

{jobs_text}

Return a JSON array with at least 15 skills. Categories should be one of:
Programming Language, Framework, Tool, Platform, Database, Cloud Service,
Methodology, Soft Skill, Certification, Library, Concept.

Return ONLY the JSON array."""

        result = _call_groq(
            system_prompt=f"You are a technical recruiter who specializes in {job_title} roles. Extract skills precisely. Return only valid JSON arrays.",
            user_prompt=prompt,
            max_tokens=3000,
        )

        if result and isinstance(result, list):
            for skill in result:
                name = skill.get("skill", "").strip()
                if not name:
                    continue
                key = name.lower()
                if key in all_skills:
                    all_skills[key]["job_count"] += skill.get("job_count", 1)
                else:
                    all_skills[key] = {
                        "skill": name,
                        "category": skill.get("category", "Technical"),
                        "job_count": skill.get("job_count", 1),
                        "importance": skill.get("importance", "Medium"),
                    }

    # Recalculate importance based on total counts
    total_jobs = len(job_listings)
    skills_list = list(all_skills.values())
    for s in skills_list:
        pct = (s["job_count"] / max(total_jobs, 1)) * 100
        s["frequency_pct"] = round(pct, 1)
        if pct >= 60:
            s["importance"] = "High"
        elif pct >= 30:
            s["importance"] = "Medium"
        else:
            s["importance"] = "Low"

    skills_list.sort(key=lambda x: x["job_count"], reverse=True)
    logger.info(f"Extracted {len(skills_list)} unique skills from {total_jobs} postings")
    return skills_list


def extract_resume_skills(resume_text: str, job_title: str) -> list[dict]:
    """Extract skills from resume text."""

    if not resume_text or not resume_text.strip():
        return []

    prompt = f"""Extract ALL technical skills, tools, frameworks, languages, platforms,
methodologies, and certifications from this resume.

Context: This person is targeting {job_title} roles.

Resume:
{resume_text[:4000]}

Think step by step:
1. Read each section of the resume
2. Pull out every technical term, tool, language, framework, platform
3. Also note soft skills and certifications
4. Estimate proficiency from context (years of experience, project depth)

Return a JSON array:
[
    {{
        "skill": "Skill Name",
        "category": "Programming Language/Framework/Tool/Platform/etc",
        "proficiency": "Beginner/Intermediate/Advanced"
    }}
]

Be thorough. Include everything mentioned even once. Return ONLY the JSON array."""

    result = _call_groq(
        system_prompt="You are a resume parser specializing in technical skill extraction. Be thorough and return only valid JSON.",
        user_prompt=prompt,
        max_tokens=2000,
    )

    if result and isinstance(result, list):
        logger.info(f"Extracted {len(result)} skills from resume")
        return result

    return []
