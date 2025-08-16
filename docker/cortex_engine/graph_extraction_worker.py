# ## File: graph_extraction_worker.py
# Version: 2.1.1 ("Verbose Debugging")
# Date: 2025-07-08
# Purpose: A robust worker script using the outlines library.
#          - FIX (v2.1.1): Added verbose print statements to stderr for debugging
#            the inputs and outputs of the extraction process.

import sys
import argparse
import json
import ollama
from outlines.models.ollama import Ollama
from outlines import Generator
from outlines.types import JsonSchema

def main():
    """
    Accepts model name and schema via arguments, reads the prompt from stdin,
    performs guided generation, and prints the resulting JSON to stdout.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, type=str, help="The name of the Ollama model to use.")
    parser.add_argument("--schema-str", required=True, type=str, help="The Pydantic JSON schema as a string.")
    args = parser.parse_args()

    try:
        prompt = sys.stdin.read()
        client = ollama.Client()

        # --- START: v2.1.1 DEBUGGING ADDITION ---
        # Print all the inputs this worker script received to the error stream for debugging
        print("--- GRAPH EXTRACTION WORKER (v2.1.1) INPUTS ---", file=sys.stderr)
        print(f"Model Name: {args.model_name}", file=sys.stderr)
        print(f"Schema: {args.schema_str}", file=sys.stderr)
        print(f"Prompt Text (first 500 chars): {prompt[:500]}...", file=sys.stderr)
        print("-------------------------------------------------", file=sys.stderr)
        # --- END: v2.1.1 DEBUGGING ADDITION ---

        model = Ollama(model_name=args.model_name, client=client)

        generator = Generator(model, JsonSchema(args.schema_str))
        generated_json_str = generator(prompt)

        # --- START: v2.1.1 DEBUGGING ADDITION ---
        # Print the raw generated output before sending it to stdout
        print("--- RAW LLM EXTRACTION OUTPUT ---", file=sys.stderr)
        print(generated_json_str, file=sys.stderr)
        print("---------------------------------", file=sys.stderr)
        # --- END: v2.1.1 DEBUGGING ADDITION ---

        print(generated_json_str) # This is the actual output for the parent script
        sys.exit(0)

    except Exception as e:
        print(f"Worker script failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()