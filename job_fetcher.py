"""
Job Fetcher
Fetches live job postings from RapidAPI JSearch.
"""

import os
import json
import requests
from monitoring import monitor, logger


def fetch_jobs(
    job_title: str,
    location: str = "United States",
    num_pages: int = 5,
    date_posted: str = "week",
) -> list[dict]:
    """Fetch live job postings. Returns list of dicts with title, company, description."""

    api_key = os.getenv("RAPID_API_KEY", "")
    if not api_key:
        logger.warning("RAPID_API_KEY not set")
        return []

    all_jobs = []

    for page in range(1, num_pages + 1):
        try:
            resp = requests.get(
                "https://jsearch.p.rapidapi.com/search-v2",
                headers={
                    "X-RapidAPI-Key": api_key,
                    "X-RapidAPI-Host": "jsearch.p.rapidapi.com",
                },
                params={
                    "query": f"{job_title} in {location}",
                    "page": str(page),
                    "num_pages": "1",
                    "date_posted": date_posted,
                    "country": "us",
                },
                timeout=30,
            )

            # ── DEBUG: Log the raw response so we can see exactly what the API returns ──
            raw_text = resp.text[:1000]
            logger.info(
                f"JSearch page {page}: status={resp.status_code} raw={raw_text!r}"
            )

            if resp.status_code != 200:
                logger.warning(f"JSearch API page {page}: HTTP {resp.status_code}")
                break

            # Parse JSON
            try:
                data = resp.json()
            except json.JSONDecodeError as je:
                logger.error(f"JSearch page {page}: invalid JSON — {je}")
                break

            # If the response is a plain string, the API returned an error message
            if isinstance(data, str):
                logger.error(f"JSearch page {page}: API returned string: {data!r}")
                break

            # If the response is a list, handle that too
            if isinstance(data, list):
                logger.error(f"JSearch page {page}: API returned list (len={len(data)})")
                break

            # At this point data is a dict — check for API-level errors
            status_field = data.get("status")
            if status_field is not None and status_field != "OK":
                err = data.get("error", data.get("message", "unknown"))
                logger.error(f"JSearch page {page}: status={status_field!r} error={err!r}")
                break

            # Extract jobs from the response
            jobs_on_page = data.get("data", {}).get("jobs", [])

            # search-v2 might nest jobs differently — log the structure
            if jobs_on_page:
                first_item = jobs_on_page[0]
                logger.info(
                    f"JSearch page {page}: got {len(jobs_on_page)} items, "
                    f"first item type={type(first_item).__name__}"
                )
                if isinstance(first_item, dict):
                    logger.info(f"JSearch page {page}: first item keys={list(first_item.keys())[:10]}")
            else:
                logger.info(f"JSearch page {page}: no results in 'data' field")
                # Check if results are under a different key
                logger.info(f"JSearch page {page}: top-level keys={list(data.keys())}")
                break

            for job in jobs_on_page:
                # Skip non-dict items
                if not isinstance(job, dict):
                    logger.warning(f"JSearch page {page}: skipping non-dict job item: {type(job).__name__}")
                    continue

                all_jobs.append(
                    {
                        "title": job.get("job_title", ""),
                        "company": job.get("employer_name", ""),
                        "description": (job.get("job_description", "") or "")[:2000],
                        "location": job.get("job_city", ""),
                        "posted": job.get("job_posted_at_datetime_utc", ""),
                    }
                )

        except requests.exceptions.Timeout:
            logger.warning(f"JSearch API page {page} timed out")
            break
        except Exception as e:
            logger.error(f"JSearch API page {page} unexpected error: {type(e).__name__}: {e}")
            break

    logger.info(f"Fetched {len(all_jobs)} jobs for '{job_title}' in '{location}'")
    return all_jobs