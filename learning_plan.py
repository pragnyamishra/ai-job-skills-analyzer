"""
Learning Plan Generator
Creates a personalized week-by-week learning plan based on identified skill gaps.
"""

import json
import os
import time
import requests
from monitoring import monitor, logger

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"


def generate_learning_plan(
    skill_gaps: list[dict],
    job_title: str,
    timeline_weeks: int = 8,
) -> dict | None:
    """Generate a week-by-week learning plan addressing the top skill gaps."""

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return None

    if not skill_gaps:
        return None

    gap_list = "\n".join(
        f"- {g['skill']} (Importance: {g.get('importance', 'Medium')}, "
        f"Found in {g.get('job_count', 0)} postings, "
        f"Category: {g.get('category', 'Technical')})"
        for g in skill_gaps[:12]
    )

    prompt = f"""Create a detailed {timeline_weeks}-week learning plan for someone who wants to
become a {job_title}. They need to learn these skills, ordered by priority:

{gap_list}

Requirements:
- Group related skills into the same week where it makes sense
- Each week should have a clear focus theme
- Include ONLY free resources (YouTube, official docs, freeCodeCamp, Kaggle, etc.)
- Each week must have a hands-on mini project
- The plan should build progressively, earlier weeks support later ones
- Include a capstone project at the end that combines multiple skills

Return as JSON:
{{
    "total_weeks": {timeline_weeks},
    "weeks": [
        {{
            "week": 1,
            "theme": "Short theme title",
            "focus_skills": ["skill1", "skill2"],
            "resources": [
                {{"title": "Resource Name", "url": "https://...", "type": "course/tutorial/docs/video"}}
            ],
            "project": "Description of what to build this week",
            "outcome": "What you can demonstrate after this week"
        }}
    ],
    "capstone_project": {{
        "title": "Project title",
        "description": "What to build",
        "skills_covered": ["skill1", "skill2", "skill3"]
    }}
}}

Return ONLY valid JSON."""

    start = time.time()
    try:
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a career learning path expert. "
                            "Create practical, actionable plans with real free resources and URLs. "
                            "Return only valid JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.5,
                "max_tokens": 4000,
            },
            timeout=90,
        )

        latency = (time.time() - start) * 1000

        if resp.status_code != 200:
            logger.error(f"Learning plan Groq error {resp.status_code}")
            return None

        result = resp.json()
        usage = result.get("usage", {})

        monitor.track_llm_call(
            func_name="generate_learning_plan",
            model=MODEL,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=latency,
        )

        text = result["choices"][0]["message"]["content"].strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        return json.loads(text)

    except Exception as e:
        logger.error(f"Learning plan error: {e}")
        return None
