from fastapi import FastAPI
from fastapi.responses import JSONResponse
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, Annotated, Optional
from pydantic import Field
import ast

grok_llm = ChatOpenAI(
    model="openai/gpt-oss-20b",
    api_key="apikey",
    base_url="https://api.groq.com/openai/v1"
)

class theState(TypedDict):
    extracted_tasks:            Optional[Annotated[list[str],       "List containing Extracted Tasks",                                          Field(default=None)]]
    extracted_tasks_w_asignees: Optional[Annotated[dict[str,list[str]], "Dict: assignee → tasks",                                               Field(default=None)]]
    uns_info:                   Annotated[str,                      "Unstructured Information"]
    priority_wise_tasks:        Optional[Annotated[dict[str,list[str]],                                                                         Field(default=None)]]
    validation_score:           Optional[Annotated[str | int,       "Validation score",                                                         Field(default=None)]]
    validation_reson:           Optional[Annotated[str,             "Validation reason",                                                        Field(default=None)]]
    retry_count:                Optional[int]
    status_message:             Optional[Annotated[str,             "User-facing error or warning message",                                     Field(default=None)]]



def string_to_list(task_string: str) -> list[str]:
    return ast.literal_eval(task_string)


async def extract_tasks(state: theState):
    """Extracts tasks from unstructured input."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a task extraction assistant.

Your job is to analyze the user's input and extract every actionable task mentioned.

Rules:
- Extract all tasks from the input.
- If a task clearly belongs to a specific person, prepend the assignee's name:
  "Rahul: Complete the API integration"
- If the assignee is unclear, use:
  "Open: Complete the API integration"
- Preserve deadlines, timelines, priorities, and dependencies when explicitly mentioned.
- Each task must be a separate string.

Output Requirements:
- Return ONLY a valid Python list of strings.
- No explanations, notes, markdown, code fences, or any text outside the list.
- Must be directly parseable with Python's ast.literal_eval().

Example: ["Rahul: Complete the API integration by Friday", "Open: Prepare client presentation"]
         

Return only the Python list."""),
        ("human", "{query}")
    ])

    chain = prompt | grok_llm
    response = await chain.ainvoke({"query": state["uns_info"]})
    my_extracted_tasks = string_to_list(response.content)

    # ── CHANGE 1: empty list → set status_message and stop (no retry) ──
    if not my_extracted_tasks:
        return {
            "extracted_tasks": [],
            "status_message": "No tasks could be extracted from your input. Please provide a more detailed query."
        }

    return {"extracted_tasks": my_extracted_tasks}


async def task_wise_ass(state: theState):
    """Groups tasks by assignee."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a task assignment extraction assistant.

Input: A list of task strings.

Your job:
- Extract the assignee from each task and group all tasks by assignee.

Output:
Return ONLY a valid Python dictionary:
{{
    "eeshan": ["task 1"],
    "neel": ["task 1", "task 2"]
}}

Rules:
- Tasks marked Open will have "Open" as the assignee key.
- Keys = assignee names. Values = lists of their tasks.
- No explanations, markdown, code fences, or any text outside the dictionary.
- Return only the dictionary."""),
        ("human", "{query}")
    ])

    chain = prompt | grok_llm
    response = await chain.ainvoke({"query": str(state["extracted_tasks"])})

    # Parse the raw string dict the LLM returns
    import ast as _ast
    result = _ast.literal_eval(response.content)

    return {"extracted_tasks_w_asignees": result}


def calc_validation(state: theState):
    """Scores the extracted output using LLM-as-judge via Groq."""
    import json
    import ast as _ast

    res = json.dumps(state["extracted_tasks_w_asignees"])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a strict task extraction validator.

You will be given:
1. ORIGINAL INPUT: the raw unstructured text
2. EXTRACTED OUTPUT: a dictionary mapping assignees to their tasks

Score the extraction from 0.0 to 1.0 based on:
- Are all assignees correctly identified?
- Are all tasks captured without being missed or hallucinated?
- Are deadlines/timelines preserved if mentioned in the input?

Return ONLY a valid Python dictionary with exactly these two keys:
{{"score": <float between 0.0 and 1.0>, "reason": "<one sentence>"}}

No markdown, no explanation, no text outside the dictionary."""),
        ("human", "ORIGINAL INPUT:\n{original}\n\nEXTRACTED OUTPUT:\n{extracted}")
    ])

    chain = prompt | grok_llm
    response = chain.invoke({
        "original":  state["uns_info"],
        "extracted": res
    })

    result = _ast.literal_eval(response.content)
    score  = float(result.get("score", 0.0))
    reason = result.get("reason", "")

    print(f"Score: {score} | Reason: {reason}")

    return {
        "validation_score": score,
        "validation_reson": reason
    }


# ─────────────────────────────────────────────
# Conditional edge functions
# ─────────────────────────────────────────────

def route_after_extraction(state: theState) -> str:
    """
    CHANGE 1 (routing): if extraction produced an empty list,
    skip everything and go straight to END.
    """
    if not state.get("extracted_tasks"):
        return "empty"
    return "continue"


def validation(state: theState) -> str:
    """
    CHANGE 2: enforce max 3 retries.
    If retries exhausted → set error message and end.
    If score good enough  → end normally.
    Otherwise            → retry.
    """
    score = float(state.get("validation_score") or 0.0)
    retry_count = state.get("retry_count", 0)

    print(f"Score: {score} | Retry count: {retry_count}")

    if score > 0.5:
        return "valid"

    if retry_count >= 3:
        return "max_retries"

    return "again"


# ─────────────────────────────────────────────
# Node that bumps the retry counter before looping back
# ─────────────────────────────────────────────

def increment_retry(state: theState):
    """Increments retry_count so validation() can enforce the cap."""
    return {"retry_count": (state.get("retry_count") or 0) + 1}


# Node that writes the max-retry error into status_message
def set_max_retry_error(state: theState):
    return {
        "status_message": (
            "Unable to process your request after 3 attempts. "
            "Please modify your query and try again."
        )
    }

from langgraph.graph import StateGraph, START, END

graph = StateGraph(theState)

graph.add_node("extract_tasks",       extract_tasks)
graph.add_node("task_wise_ass",       task_wise_ass)
graph.add_node("calc_validation",     calc_validation)
graph.add_node("increment_retry",     increment_retry)
graph.add_node("set_max_retry_error", set_max_retry_error)

# START → extract_tasks
graph.add_edge(START, "extract_tasks")

# After extraction: empty list → END immediately, else → task_wise_ass
graph.add_conditional_edges(
    "extract_tasks",
    route_after_extraction,
    {
        "empty":    END,          # status_message already set inside extract_tasks
        "continue": "task_wise_ass"
    }
)

graph.add_edge("task_wise_ass", "calc_validation")

# After validation scoring
graph.add_conditional_edges(
    "calc_validation",
    validation,
    {
        "valid":       END,
        "max_retries": "set_max_retry_error",  
        "again":       "increment_retry"        
    }
)

graph.add_edge("set_max_retry_error", END)
graph.add_edge("increment_retry",     "extract_tasks")  # loop back

workflow_app = graph.compile()

# ─────────────────────────────────────────────
# FastAPI
# ─────────────────────────────────────────────

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

fastapi_app = FastAPI(title="Task Extraction API")
templates   = Jinja2Templates(directory="templates")


@fastapi_app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})


@fastapi_app.get("/extract-tasks")
async def extract_tasks_endpoint(query: str):
    """
    Runs the multi-agent pipeline and returns structured task output.

    Always returns HTTP 200 so the frontend can display the message properly.
    Check 'status_message' in the response for any errors.
    """
    result = await workflow_app.ainvoke({
        "uns_info":    query,
        "retry_count": 0
    })

    # If a status_message was set, something went wrong — surface it cleanly
    if result.get("status_message"):
        return JSONResponse({
            "success":       False,
            "status_message": result["status_message"],
            "extracted_tasks_w_asignees": None
        })

    return JSONResponse({
        "success":                    True,
        "status_message":             None,
        "extracted_tasks_w_asignees": result.get("extracted_tasks_w_asignees")
    })


if __name__ == "__main__":
    from pyngrok import ngrok
    public_url = ngrok.connect(8000)
    print(f"\n🔗 Public URL: {public_url}\n")
    import uvicorn
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)

    