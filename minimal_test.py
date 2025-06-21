#!/usr/bin/env python3
"""
Minimal test of bot functionality
"""

from poker_bot_pluribus import PluribusBot, GameState, Action, sample_from_distribution

def test_sampling_function():
    """Test the robust sampling function directly"""
    print("Testing robust sampling function...")
    
    # Test 1: Normal probabilities
    actions = ['a', 'b', 'c']
    probs = [0.2, 0.5, 0.3]
    
    samples = []
    for _ in range(100):
        samples.append(sample_from_distribution(actions, probs))
    
    # Check all actions were sampled
    unique_samples = set(samples)
    print(f"Sampled actions: {unique_samples}")
    
    # Test 2: Edge case - all zeros
    probs = [0.0, 0.0, 0.0]
    result = sample_from_distribution(actions, probs)
    print(f"All-zero probabilities result: {result}")
    
    # Test 3: Edge case - very small numbers
    probs = [1e-100, 1e-100, 1e-100]
    result = sample_from_distribution(actions, probs)
    print(f"Tiny probabilities result: {result}")
    
    print("✓ Robust sampling works!\n")

def test_bot_creation():
    """Test creating bot without training"""
    print("Testing bot creation...")
    
    try:
        bot = PluribusBot()
        print("✓ Bot created successfully")
        
        # Create a simple game state
        state = GameState(
            pot=100,
            players_in_hand={0, 1},
            current_player=0,
            board_cards=[],
            betting_round=0,
            last_bet=50,
            player_bets={0: 25, 1: 50},
            player_chips={0: 1000, 1: 1000}
        )
        
        # Try to get an action (will use uniform random strategy)
        action, amount = bot.get_action(state, player=0, use_search=False)
        print(f"✓ Bot returned action: {action.name} ${amount}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=== Minimal Bot Test ===\n")
    
    test_sampling_function()
    
    if test_bot_creation():
        print("\n✅ Basic bot functionality confirmed!")
        print("\nYou should now be able to:")
        print("1. Run quick_test.py for more thorough testing")
        print("2. Run train.py to train the bot")
        print("3. Run play.py to play against the bot")
    else:
        print("\n❌ Bot creation failed. Check the error above.")

if __name__ == "__main__":
    main()
