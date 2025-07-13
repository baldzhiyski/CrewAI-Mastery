#!/usr/bin/env python
import sys
import warnings
from dotenv import load_dotenv
from crew import PdfRag

load_dotenv()
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """

    user_input = input("Enter your question: ")

    inputs = {
        "input": str(user_input).strip()
    }

    result = PdfRag().crew().kickoff(inputs=inputs)
    print(result)

if __name__ == "__main__":
    run()