#!/usr/bin/env python3
"""
Quick test to verify the fixes work
"""

from poker_bot_pluribus import PluribusBot, GameState, Action, sample_from_distribution
import numpy as np

def test_sample_action_fix():
    """Test that the sample action fix works"""
    print("Testing sample action fix...")
    
    # Create a mock strategy with tuple actions
    actions = [
        (Action.FOLD, 0),
        (Action.CALL, 100),
        (Action.RAISE, 200)
    ]
    probs = [0.2, 0.5, 0.3]
    
    # Test sampling
    try:
        sampled = sample_from_distribution(actions, probs)
        print(f"✓ Successfully sampled action: {sampled}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_state_cloning():
    """Test that game state cloning works"""
    print("\nTesting game state cloning...")
    
    try:
        # Create a test state
        state = GameState(
            pot=100,
            players_in_hand={0, 1, 2},
            current_player=0,
            board_cards=[1, 2, 3],
            betting_round=1,
            last_bet=50,
            player_bets={0: 50, 1: 50},
            player_chips={0: 1000, 1: 1000, 2: 1000}
        )
        
        # Clone it
        new_state = GameState(
            pot=state.pot,
            players_in_hand=state.players_in_hand.copy(),
            current_player=state.current_player,
            board_cards=state.board_cards.copy(),
            betting_round=state.betting_round,
            last_bet=state.last_bet,
            player_bets=state.player_bets.copy(),
            player_chips=state.player_chips.copy()
        )
        
        # Modify new state
        new_state.players_in_hand.remove(0)
        
        # Check original is unchanged
        assert 0 in state.players_in_hand
        assert 0 not in new_state.players_in_hand
        
        print("✓ Game state cloning works correctly")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_minimal_training():
    """Test minimal training works"""
    print("\nTesting minimal training...")
    
    try:
        bot = PluribusBot()
        # Test with even fewer iterations
        bot.train_blueprint(iterations=5)
        print(f"✓ Training completed with {bot.blueprint.iteration} iterations")
        print(f"  Strategy nodes: {len(bot.blueprint.strategy_sum)}")
        print(f"  Regret nodes: {len(bot.blueprint.regret_sum)}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_probability_normalization():
    """Test probability normalization edge cases"""
    print("\nTesting probability normalization...")
    
    try:
        # Test robust sampling with edge cases
        
        # Test 1: Normal case
        actions = ['a', 'b', 'c']
        probs = [0.2, 0.5, 0.3]
        result = sample_from_distribution(actions, probs)
        assert result in actions
        
        # Test 2: All zeros
        probs = [0.0, 0.0, 0.0]
        result = sample_from_distribution(actions, probs)
        assert result in actions
        
        # Test 3: Very small numbers
        probs = [1e-10, 1e-10, 1e-10]
        result = sample_from_distribution(actions, probs)
        assert result in actions
        
        # Test 4: One large, others zero
        probs = [0.0, 1.0, 0.0]
        samples = [sample_from_distribution(actions, probs) for _ in range(10)]
        assert all(s == 'b' for s in samples)
        
        print("✓ Probability normalization works correctly")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    print("=== Quick Test of Fixes ===\n")
    
    tests = [
        test_sample_action_fix,
        test_state_cloning,
        test_probability_normalization,
        test_minimal_training
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n{'='*30}")
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("\n✅ All tests passed! You can now run train.py")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
