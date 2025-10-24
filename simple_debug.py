"""
Simple debug script to find strategy_text error
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("Debugging EnhancedMichael for strategy_text error...")

    try:
        # Test import
        from src.enhanced_agent import EnhancedMichael
        print("Import successful")

        # Initialize agent
        agent = EnhancedMichael("qwen3-8b")
        print("Agent initialized")

        # Test simple call
        obs = "You are Player 0. Your role: Doctor. Team: Villager."
        result = agent(obs)
        print(f"Agent call successful: {result}")

    except NameError as e:
        if "strategy_text" in str(e):
            print(f"FOUND strategy_text error: {e}")
            import traceback
            traceback.print_exc()
        else:
            print(f"Different NameError: {e}")
    except Exception as e:
        print(f"Other error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()