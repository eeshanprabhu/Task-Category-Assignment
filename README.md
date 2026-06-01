# Task Extraction API

## Overview

Task Extraction API is a multi-agent workflow system built using FastAPI, LangGraph, LangChain, and Groq LLMs. It processes unstructured text input and converts it into structured task assignments.

The system:

* Extracts actionable tasks from free-form text.
* Identifies assignees.
* Groups tasks by assignee.
* Validates extraction quality using an LLM-as-a-judge approach.
* Automatically retries failed extractions up to 3 times.
* Returns structured JSON output.

---

## Features

### Task Extraction

Extracts all actionable tasks from unstructured text.

Example Input:

```text
Rahul will complete the API integration by Friday.
Priya should prepare the client presentation.
The testing checklist needs to be reviewed.
```

Example Output:

```python
[
    "Rahul: Complete the API integration by Friday",
    "Priya: Prepare the client presentation",
    "Open: Review the testing checklist"
]
```

---

### Assignee Grouping

Converts extracted tasks into an assignee-wise structure.

Example:

```python
{
    "Rahul": [
        "Complete the API integration by Friday"
    ],
    "Priya": [
        "Prepare the client presentation"
    ],
    "Open": [
        "Review the testing checklist"
    ]
}
```

---

### Validation Layer

The extracted output is evaluated against the original input using an LLM validator.

Validation checks:

* Task completeness
* Assignee correctness
* Deadline preservation
* Hallucination detection

A score between 0.0 and 1.0 is generated.

---

### Automatic Retry Mechanism

If validation score is below the threshold:

```python
score <= 0.5
```

the workflow automatically retries extraction.

Maximum retries:

```python
3
```

After 3 failed attempts, the workflow terminates and returns an error message.

---

## Architecture

```text
START
  │
  ▼
extract_tasks
  │
  ├── Empty Result ──► END
  │
  ▼
task_wise_ass
  │
  ▼
calc_validation
  │
  ├── score > 0.5 ──► END
  │
  ├── retry_count >= 3 ──► set_max_retry_error ──► END
  │
  ▼
increment_retry
  │
  └──────────────► extract_tasks
```

---

## Tech Stack

* FastAPI
* LangGraph
* LangChain
* Groq API
* OpenAI-compatible SDK
* Python 3.10+

---

## Installation

### Clone Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### Create Virtual Environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Replace the API key in the code:

```python
api_key="YOUR_GROQ_API_KEY"
```

Or preferably use environment variables:

```bash
GROQ_API_KEY=your_api_key
```

---

## Running the Application

### Local Run

```bash
python app.py
```

The application starts on:

```text
http://localhost:8000
```

---

### Public Access Using Ngrok

The application automatically creates a public URL:

```python
public_url = ngrok.connect(8000)
```

Example:

```text
https://abc123.ngrok-free.app
```

---

## API Endpoints

### Home Page

```http
GET /
```

Returns the frontend page.

---

### Extract Tasks

```http
GET /extract-tasks
```

#### Query Parameters

| Parameter | Type   | Required |
| --------- | ------ | -------- |
| query     | string | Yes      |

Example:

```http
GET /extract-tasks?query=Rahul should finish API integration by Friday and Priya should prepare the demo.
```

---

## Successful Response

```json
{
  "success": true,
  "status_message": null,
  "extracted_tasks_w_asignees": {
    "Rahul": [
      "Finish API integration by Friday"
    ],
    "Priya": [
      "Prepare the demo"
    ]
  }
}
```

---

## Error Response

### No Tasks Found

```json
{
  "success": false,
  "status_message": "No tasks could be extracted from your input. Please provide a more detailed query.",
  "extracted_tasks_w_asignees": null
}
```

### Maximum Retries Reached

```json
{
  "success": false,
  "status_message": "Unable to process your request after 3 attempts. Please modify your query and try again.",
  "extracted_tasks_w_asignees": null
}
```

---

## Workflow State

```python
theState
```

Stores:

* extracted_tasks
* extracted_tasks_w_asignees
* uns_info
* priority_wise_tasks
* validation_score
* validation_reson
* retry_count
* status_message

---

## Future Improvements

* Deadline extraction as structured dates
* Priority classification
* Task dependency detection
* Task status tracking
* Database integration
* Authentication and rate limiting
* Async validation pipeline

---

## License

This project is provided for educational and demonstration purposes.
