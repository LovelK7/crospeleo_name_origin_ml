"""
Quick validation script — runs 10 hardcoded test cases through the saved model
and prints predictions with explanations and similar examples.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.predict import OriginPredictor

TEST_CASES = [
    {"name": "Jama pod Vršićem",                  "lokalitet": "Vršić",                "mjesto": ""},
    {"name": "Acronium",                           "lokalitet": "",                      "mjesto": ""},
    {"name": "Špilja kod Marijanovića kuća",       "lokalitet": "",                      "mjesto": ""},
    {"name": "Židovske jame",                      "lokalitet": "Rupe",                  "mjesto": "Pasanska Gorica"},
    {"name": "Plodni dan",                         "lokalitet": "Bijele Stijene",         "mjesto": "Delnice"},
    {"name": "Maklenska",                          "lokalitet": "Maklenske njive",        "mjesto": "Brod Moravice"},
    {"name": "Borušnjak 2",                        "lokalitet": "",                      "mjesto": ""},
    {"name": "Grbina peć",                         "lokalitet": "Lesina, Ćićarija",      "mjesto": "Buzet"},
    {"name": "Konzerva",                           "lokalitet": "Srednji Velebit",        "mjesto": ""},
    {"name": "Spilja u Japagama",                  "lokalitet": "",                      "mjesto": "Krašić"},
]

SEP = "-" * 80


def run():
    print("Loading model ...")
    predictor = OriginPredictor()
    print(f"Model loaded. Running {len(TEST_CASES)} test cases.\n")

    for i, tc in enumerate(TEST_CASES, 1):
        result = predictor.predict(
            ime_objekta    = tc["name"],
            lokalitet      = tc.get("lokalitet", ""),
            najblize_mjesto= tc.get("mjesto", ""),
        )

        conf_bar = "#" * int(result["confidence"] * 20)
        flag = " [NEEDS REVIEW]" if result["needs_user_confirmation"] else ""

        print(SEP)
        print(f"[{i:02d}] {tc['name']}")
        if tc.get("lokalitet"):
            print(f"     Lokalitet: {tc['lokalitet']}")
        if tc.get("mjesto"):
            print(f"     Mjesto:    {tc['mjesto']}")
        print()
        print(f"     Prediction : {result['predicted_value']}{flag}")
        print(f"     Confidence : {result['confidence']:.3f}  {conf_bar}")
        print()

        print("     Top-3:")
        for rank, entry in enumerate(result["top_k"][:3], 1):
            bar = "#" * int(entry["score"] * 20)
            print(f"       {rank}. {entry['value']:<35} {entry['score']:.4f}  {bar}")
        print()

        print("     Explanation:")
        for reason in result["explanation"]:
            print(f"       - {reason}")
        print()

        print("     Similar training examples:")
        for ex in result["similar_examples"]:
            print(f"       {ex['similarity']:.3f}  [{ex['origin'][:25]:<25}]  {ex['name']}")
        print()

    print(SEP)
    print("Validation complete.")


if __name__ == "__main__":
    run()
