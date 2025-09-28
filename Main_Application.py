from src.engine import InsightEngine
from src.visualization import visualize_simulation
import argparse

def main():
    """
    Main function to run the LLM Insight Engine from the command line.
    """
    parser = argparse.ArgumentParser(description="Run the LLM Insight Engine on a sentence.")
    parser.add_argument(
        "text",
        type=str,
        help="The sentence or text to analyze."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bert-base-uncased",
        help="The Hugging Face model name to use."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reasoning_map.png",
        help="The file path to save the output visualization."
    )
    args = parser.parse_args()

    print("--- LLM Insight Engine ---")
    
    try:
        # 1. Initialize the engine
        engine = InsightEngine(model_name=args.model)

        # 2. Perform the analysis
        print(f"\nAnalyzing text: '{args.text}'")
        analysis_result = engine.analyze_sentence(args.text)

        # 3. Visualize the results
        print("\nGenerating visualization...")
        visualize_simulation(
            positions=analysis_result['final_positions'],
            tokens=analysis_result['tokens'],
            attention_matrix=analysis_result['attention_matrix'],
            save_path=args.output
        )
        print("\n--- Analysis Complete ---")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure you have a stable internet connection and the model name is correct.")


if __name__ == "__main__":
    main()
