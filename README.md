\# Race Strategy Generator



An AI-powered race strategy engine for trail ultras (starting with Chianti Ultra Trail 75 km).



\## Features



\- GPX loading and elevation cleaning

\- Course modeling (distance, gain, gradient, segments, key climbs)

\- LLM-powered race strategy generator (pacing, fueling, mental cues)

\- Structured JSON output for UI / exports



\## Tech Stack



\- Python

\- Jupyter Notebook

\- OpenAI API

\- pandas, numpy, scipy



\## Getting Started (locally)



```bash

\# 1. Clone the repo

git clone https://github.com/<your-username>/race-strategy-generator.git

cd race-strategy-generator



\# 2. Create and activate a virtualenv

python -m venv .venv

\# On Windows:

.venv\\Scripts\\activate

\# On macOS / Linux:

source .venv/bin/activate



\# 3. Install dependencies

pip install -r requirements.txt



\# 4. Create a .env file

OPENAI\_API\_KEY=sk-...



\# 5. Run notebooks or scripts



