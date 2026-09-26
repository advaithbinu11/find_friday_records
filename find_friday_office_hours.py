#!/usr/bin/env python3
"""
find_friday_office_hours.py

Given a list of UT Austin CS professors, this script uses the OpenAI
("ChatGPT") API with its built-in web-search tool to look up each
professor's office hours, then filters down to the ones who hold
office hours on Friday afternoon.

SETUP
-----
1. pip install openai
2. Set your API key:
     export OPENAI_API_KEY="sk-..."
3. Put professor names in professors.txt (one per line), or edit the
   DEFAULT_PROFESSORS list below.

USAGE
-----
    python find_friday_office_hours.py
    python find_friday_office_hours.py --input professors.txt --output results.csv

NOTES
-----
- This uses OpenAI's Responses API with the "web_search" tool so the
  model can actually look up current office-hours pages instead of
  guessing from training data. Model/tool names occasionally change on
  OpenAI's side -- if you get an error about an unknown tool or model,
  check https://platform.openai.com/docs for the current web-search
  tool name (it has been called "web_search" / "web_search_preview"
  at different times) and update WEB_SEARCH_TOOL / MODEL below.
- Office hours listed as e.g. "1:00pm-2:30pm" on Friday count as
  "Friday afternoon" (roughly 12:00pm-5:59pm). Adjust
  is_friday_afternoon() if you want a different window.
- Always spot-check the results -- office-hours pages change every
  semester and the model can misread a page or find a stale one.
"""

import argparse
import csv
import json
import os
import re
import sys
import time

try:
    from openai import OpenAI
except ImportError:
    sys.exit("Missing dependency. Run: pip install openai")

MODEL = "gpt-4.1"                # change if you want a different model
WEB_SEARCH_TOOL = "web_search"   # change if OpenAI renames the tool

AI_ML_PROFESSORS = [
    "Peter Stone",
    "Raymond Mooney",
    "Kristen Grauman",
    "Swarat Chaudhuri",
    "Joydeep Biswas",
    "Qiang Liu",
    "Philipp Krahenbuhl",
    "Alex Dimakis",
    "Inderjit Dhillon",
    "Sujay Sanghavi",
    "Risto Miikkulainen",
    "Isil Dillig",
]
 
THEORY_ALGORITHMS_PROFESSORS = [
    "Scott Aaronson",
    "David Zuckerman",
    "Adam Klivans",
    "Vijaya Ramachandran",
    "Greg Plaxton",
    "Kevin Tian",
    "Brent Waters",
    "David Wu",
    "Eric Price",
]
 
DISTRIBUTED_SYSTEMS_PROFESSORS = [
    "Vijay Chidambaram",
    "Aditya Akella",
    "Emmett Witchel",
    "Michael Dahlin",
    "Keshav Pingali",
    "Simon Peter",
]
 
# Specifically requested professors (names corrected to their actual
# UT Austin CS listings):
#   "Alisa Norman"   -> Alison N. Norman
#   "Paarikh"        -> Devangi N. Parikh
#   "Ravishanker"    -> Veena Ravishankar
#   "Scott"          -> Scott Aaronson (already listed under theory)
REQUESTED_PROFESSORS = [
    "Alison N. Norman",
    "Devangi N. Parikh",
    "Veena Ravishankar",
]
 
DEFAULT_PROFESSORS = sorted(set(
    AI_ML_PROFESSORS
    + THEORY_ALGORITHMS_PROFESSORS
    + DISTRIBUTED_SYSTEMS_PROFESSORS
    + REQUESTED_PROFESSORS
))

SYSTEM_PROMPT = """You are a research assistant. Given a professor's name and
their university/department, use web search to find their CURRENT posted
office hours (day(s) of week and time range), ideally from their official
university or department page.

Respond with ONLY a JSON object, no other text, in this exact shape:
{
  "professor": "<name>",
  "found": true/false,
  "office_hours": [
    {"day": "Monday", "start": "2:00pm", "end": "3:00pm"}
  ],
  "source_url": "<url or empty string>"
}

If you cannot find office hours, set "found": false and use an empty list
and empty string for the other fields. Only include entries you are
reasonably confident about."""


def ask_model(client: OpenAI, professor: str) -> dict:
    """Query the model (with web search) for one professor's office hours."""
    user_prompt = (
        f"Professor: {professor}\n"
        f"Department: Computer Science, University of Texas at Austin\n"
        "Find their current office hours."
    )

    response = client.responses.create(
        model=MODEL,
        tools=[{"type": WEB_SEARCH_TOOL}],
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    text = response.output_text.strip()

    # Model sometimes wraps JSON in ```json fences despite instructions.
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "professor": professor,
            "found": False,
            "office_hours": [],
            "source_url": "",
            "_raw_response": text,
        }


def parse_time_to_hour(t: str):
    """Parse a time string like '2:30pm' -> 14.5 (float hour, 24h clock)."""
    t = t.strip().lower().replace(" ", "")
    m = re.match(r"(\d{1,2})(?::(\d{2}))?(am|pm)?", t)
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2)) if m.group(2) else 0
    meridiem = m.group(3)
    if meridiem == "pm" and hour != 12:
        hour += 12
    if meridiem == "am" and hour == 12:
        hour = 0
    return hour + minute / 60


def is_friday_afternoon(entry: dict, afternoon_start=12.0, afternoon_end=18.0) -> bool:
    if entry.get("day", "").strip().lower() != "friday":
        return False
    start = parse_time_to_hour(entry.get("start", ""))
    end = parse_time_to_hour(entry.get("end", ""))
    if start is None or end is None:
        return False
    # Overlaps the afternoon window at all
    return start < afternoon_end and end > afternoon_start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Text file with one professor name per line")
    parser.add_argument("--output", default="friday_afternoon_office_hours.csv",
                         help="CSV file to write results to")
    parser.add_argument("--delay", type=float, default=1.0,
                         help="Seconds to sleep between API calls (rate-limit friendliness)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Set OPENAI_API_KEY in your environment first.")

    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            professors = [line.strip() for line in f if line.strip()]
    else:
        professors = DEFAULT_PROFESSORS

    client = OpenAI(api_key=api_key)

    all_results = []
    friday_afternoon_matches = []

    for i, professor in enumerate(professors, 1):
        print(f"[{i}/{len(professors)}] Looking up {professor} ...")
        try:
            data = ask_model(client, professor)
        except Exception as e:
            print(f"  ERROR: {e}")
            data = {"professor": professor, "found": False, "office_hours": [], "source_url": ""}

        all_results.append(data)

        for entry in data.get("office_hours", []):
            if is_friday_afternoon(entry):
                friday_afternoon_matches.append({
                    "professor": data.get("professor", professor),
                    "day": entry.get("day", ""),
                    "start": entry.get("start", ""),
                    "end": entry.get("end", ""),
                    "source_url": data.get("source_url", ""),
                })

        time.sleep(args.delay)

    # Write full results (for auditing) and the Friday-afternoon subset.
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["professor", "day", "start", "end", "source_url"])
        writer.writeheader()
        writer.writerows(friday_afternoon_matches)

    print("\n=== Professors with Friday afternoon office hours ===")
    if friday_afternoon_matches:
        for m in friday_afternoon_matches:
            print(f"- {m['professor']}: {m['day']} {m['start']}-{m['end']} ({m['source_url']})")
    else:
        print("None found.")

    print(f"\nFull results saved to: {args.output}")


if __name__ == "__main__":
    main()
