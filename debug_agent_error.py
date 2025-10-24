"""
Debug script to identify the 'strategy_text' error in EnhancedMichael
"""

import sys
import os
import traceback
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_enhanced_agent_comprehensive():
    """Comprehensive test to reproduce the error"""
    print("Debugging EnhancedMichael error...")

    try:
        # Test 1: Import and initialize
        print("\n1. Testing import...")
        from src.enhanced_agent import EnhancedMichael
        agent = EnhancedMichael("qwen3-8b", enable_learning=True)
        print("✓ Import and initialization successful")

        # Test 2: Check all methods exist
        print("\n2. Testing method existence...")
        methods_to_check = [
            '_get_strategy_guidance',
            '_enhanced_analyze',
            '_enhanced_strategy_update',
            '_evaluate_action_for_brilliance',
            '_complete_pending_evaluations'
        ]

        for method in methods_to_check:
            if hasattr(agent, method):
                print(f"✓ {method} exists")
            else:
                print(f"✗ {method} missing")

        # Test 3: Initialize with different observations
        print("\n3. Testing with various observations...")

        test_observations = [
            "You are Player 0. Your role: Doctor. Team: Villager.",
            "You are Player 1. Your role: Mafia. Team: Mafia.",
            "[SYSTEM: Night 1 begins, [0]: I'm the detective, [1]: I think [2] is mafia]"
        ]

        for i, obs in enumerate(test_observations, 1):
            try:
                print(f"\nTest {i}: {obs[:50]}...")
                result = agent(obs)
                print(f"✓ Test {i} successful: {result}")
            except Exception as e:
                print(f"✗ Test {i} failed: {e}")
                traceback.print_exc()

                # Check if this is the target error
                if "strategy_text" in str(e):
                    print(f"*** FOUND TARGET ERROR: {e} ***")
                    return False

        # Test 4: Test internal methods directly
        print("\n4. Testing internal methods...")

        try:
            print("Testing _get_strategy_guidance...")
            guidance = agent._get_strategy_guidance("test obs", "night")
            print(f"✓ _get_strategy_guidance works: {len(guidance)} chars")
        except Exception as e:
            print(f"✗ _get_strategy_guidance failed: {e}")
            if "strategy_text" in str(e):
                print(f"*** FOUND TARGET ERROR in _get_strategy_guidance ***")
                traceback.print_exc()
                return False

        try:
            print("Testing _enhanced_analyze...")
            analysis = agent._enhanced_analyze("test obs", "")
            print(f"✓ _enhanced_analyze works: {len(analysis)} chars")
        except Exception as e:
            print(f"✗ _enhanced_analyze failed: {e}")
            if "strategy_text" in str(e):
                print(f"*** FOUND TARGET ERROR in _enhanced_analyze ***")
                traceback.print_exc()
                return False

        try:
            print("Testing _enhanced_strategy_update...")
            strategy = agent._enhanced_strategy_update("test analysis", "test belief", "")
            print(f"✓ _enhanced_strategy_update works: {len(strategy)} chars")
        except Exception as e:
            print(f"✗ _enhanced_strategy_update failed: {e}")
            if "strategy_text" in str(e):
                print(f"*** FOUND TARGET ERROR in _enhanced_strategy_update ***")
                traceback.print_exc()
                return False

        print("\n✓ All tests completed without 'strategy_text' error")
        return True

    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        traceback.print_exc()
        return False

def check_for_dynamic_errors():
    """Check for errors that might occur during dynamic execution"""
    print("\n" + "="*60)
    print("CHECKING FOR DYNAMIC ERRORS")
    print("="*60)

    try:
        # This simulates a more complex scenario
        from src.enhanced_agent import EnhancedMichael
        import textarena as ta

        # Initialize agents for a game
        agents = {}
        for i in range(3):
            agents[i] = EnhancedMichael("qwen3-8b", enable_learning=True)

        # Initialize environment
        env = ta.make(env_id="SecretMafia-v0")
        env.reset(num_players=len(agents))

        print("Running a few turns to catch dynamic errors...")

        for turn in range(5):  # Run 5 turns
            try:
                player_id, observation = env.get_observation()
                action = agents[player_id](observation)
                done, step_info = env.step(action=action)

                print(f"Turn {turn+1}: Player {player_id} -> {action}")

                if done:
                    break

            except Exception as e:
                print(f"✗ Error in turn {turn+1}: {e}")
                if "strategy_text" in str(e):
                    print(f"*** FOUND TARGET ERROR in game execution ***")
                    traceback.print_exc()
                    return False

        env.close()
        print("✓ Dynamic test completed without error")
        return True

    except Exception as e:
        print(f"✗ Dynamic test setup failed: {e}")
        if "strategy_text" in str(e):
            print(f"*** FOUND TARGET ERROR in setup ***")
            traceback.print_exc()
        return False

def main():
    """Main debug function"""
    print("ENHANCED MICHAEL ERROR DEBUGGER")
    print("Looking for 'strategy_text' NameError...")

    # Test 1: Comprehensive method testing
    success1 = test_enhanced_agent_comprehensive()

    # Test 2: Dynamic execution testing
    success2 = check_for_dynamic_errors()

    print("\n" + "="*60)
    print("DEBUG SUMMARY")
    print("="*60)

    if success1 and success2:
        print("✓ No 'strategy_text' error found")
        print("The error might be:")
        print("1. Already fixed")
        print("2. Occurs in specific edge cases not covered")
        print("3. Related to a different component")
        print("4. Context-dependent (specific game state)")
    else:
        print("✗ 'strategy_text' error found and details above")
        print("Check the traceback for exact location")

if __name__ == "__main__":
    main()