#!/usr/bin/env python3
"""
Test script to verify Pluribus poker bot installation
"""

import sys

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError:
        print("✗ NumPy not found. Run: pip install numpy")
        return False
    
    try:
        from poker_bot_pluribus import PluribusBot, GameState, Action
        print("✓ poker_bot_pluribus imported successfully")
    except ImportError as e:
        print(f"✗ Error importing poker_bot_pluribus: {e}")
        return False
    
    try:
        from poker_bot_utils import HandEvaluator, card_to_string
        print("✓ poker_bot_utils imported successfully")
    except ImportError as e:
        print(f"✗ Error importing poker_bot_utils: {e}")
        return False
    
    try:
        from poker_game_engine import PokerGameEngine, TrainingHarness
        print("✓ poker_game_engine imported successfully")
    except ImportError as e:
        print(f"✗ Error importing poker_game_engine: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    
    try:
        from poker_bot_pluribus import PluribusBot
        from poker_bot_utils import HandEvaluator, string_to_card
        
        # Test bot creation
        bot = PluribusBot()
        print("✓ Bot created successfully")
        
        # Test hand evaluator
        evaluator = HandEvaluator()
        hole = [string_to_card('As'), string_to_card('Ah')]
        board = [string_to_card('Ad'), string_to_card('Ac'), string_to_card('Kh')]
        strength = evaluator.evaluate_hand(hole, board)
        print("✓ Hand evaluator working")
        
        # Test game engine
        from poker_game_engine import PokerGameEngine
        game = PokerGameEngine(num_players=2)
        print("✓ Game engine created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during functionality test: {e}")
        return False

def main():
    print("=== Pluribus Poker Bot Installation Test ===\n")
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("⚠️  Warning: Python 3.8+ recommended")
    else:
        print("✓ Python version OK")
    
    print()
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Please check file placement and dependencies.")
        return 1
    
    # Test functionality
    if not test_basic_functionality():
        print("\n❌ Functionality test failed.")
        return 1
    
    print("\n✅ All tests passed! The poker bot is ready to use.")
    print("\nNext steps:")
    print("- Run 'python train.py' to train the bot")
    print("- Run 'python play.py' to play against the bot")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
