# IBM MAILON — Recruitment Safety Assistant

IBM MAILON helps students assess suspicious recruitment messages, recruiter emails, job links, screenshots, and PDF offer letters.

## Features

- Rule-based recruitment risk scoring
- Evidence-based findings and safety recommendations
- Optional Groq AI explanations
- Optional Sarvam multilingual analysis and translation
- Optional Tavily web research
- PDF text extraction
- Screenshot OCR using Tesseract
- Web dashboard and health endpoint

## Technology Stack

- Python 3.12+
- FastAPI, Uvicorn, and Pydantic
- HTML, CSS, and JavaScript
- Groq, Sarvam, and Tavily APIs
- pypdf and Tesseract OCR

## Setup

Clone the repository:

    git clone https://github.com/Ayvak16122005/ibm-mailon.git
    cd ibm-mailon

Create a virtual environment and install dependencies:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

Install the system Tesseract OCR package if screenshot analysis is required.

## Configuration

Configure these environment variables if you use the optional services:

- GROQ_API_KEY
- SARVAM_API_KEY
- TAVILY_API_KEY

Never commit API keys or .env files.

## Run

    uvicorn main:app --host 127.0.0.1 --port 8000

Dashboard: http://127.0.0.1:8000/dashboard

Health check: http://127.0.0.1:8000/health

## Model Files

Large model binaries are excluded from Git. Obtain them separately if required by your deployment and follow their applicable licences.

## Disclaimer

MAILON provides preliminary recruitment risk indicators, not proof that a job is fraudulent or legitimate. Verify opportunities through official employer channels.

## Security

Do not publish API keys, SSH private keys, personal information, secret-bearing logs, or local virtual environments.
