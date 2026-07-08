"""
LLMOps Monitoring Layer
Tracks latency, token usage, and API call history using Langfuse + Python logging.
"""

import os
import time
import json
import logging
from datetime import datetime
from functools import wraps

# Setup file and console logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("data/api_calls.log", mode="a"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("skills_analyzer")

# In-memory call history for the Streamlit session
_call_history = []


def get_call_history():
    return _call_history


def clear_call_history():
    _call_history.clear()


class Monitor:
    """Unified monitoring: Langfuse (if configured) + local logging."""

    def __init__(self):
        self.langfuse = None
        try:
            pub = os.getenv("LANGFUSE_PUBLIC_KEY", "")
            sec = os.getenv("LANGFUSE_SECRET_KEY", "")
            if pub and sec:
                from langfuse import Langfuse

                self.langfuse = Langfuse()
                logger.info("Langfuse connected")
            else:
                logger.info("Langfuse keys not set, using local logging only")
        except Exception as e:
            logger.info(f"Langfuse unavailable ({e}), using local logging only")

    # ---- core tracker ----
    def track_llm_call(
        self,
        func_name: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        cost_estimate: float = 0.0,
        success: bool = True,
        metadata: dict | None = None,
    ):
        record = {
            "timestamp": datetime.now().isoformat(),
            "function": func_name,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "latency_ms": round(latency_ms, 1),
            "cost_estimate": round(cost_estimate, 6),
            "success": success,
            "metadata": metadata or {},
        }

        _call_history.append(record)
        logger.info(json.dumps(record))

        if self.langfuse:
            try:
                trace = self.langfuse.trace(name=func_name)
                trace.generation(
                    name=func_name,
                    model=model,
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    },
                    metadata={
                        "latency_ms": latency_ms,
                        "cost_estimate": cost_estimate,
                        **(metadata or {}),
                    },
                )
            except Exception as e:
                logger.warning(f"Langfuse logging failed: {e}")

    # ---- decorator for easy wrapping ----
    def track(self, func_name: str | None = None):
        """Decorator that auto-tracks latency. Expects the wrapped function
        to return a dict with keys: result, model, prompt_tokens, completion_tokens."""

        def decorator(fn):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                start = time.time()
                try:
                    output = fn(*args, **kwargs)
                    latency = (time.time() - start) * 1000
                    if isinstance(output, dict) and "model" in output:
                        self.track_llm_call(
                            func_name=func_name or fn.__name__,
                            model=output.get("model", "unknown"),
                            prompt_tokens=output.get("prompt_tokens", 0),
                            completion_tokens=output.get("completion_tokens", 0),
                            latency_ms=latency,
                        )
                    return output
                except Exception as e:
                    latency = (time.time() - start) * 1000
                    self.track_llm_call(
                        func_name=func_name or fn.__name__,
                        model="unknown",
                        prompt_tokens=0,
                        completion_tokens=0,
                        latency_ms=latency,
                        success=False,
                        metadata={"error": str(e)},
                    )
                    raise

            return wrapper

        return decorator


# Singleton
monitor = Monitor()
