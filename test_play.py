#!/usr/bin/env python3
"""
Test playing mechanics without full training
"""

from poker_game_engine import PokerGameEngine

def main():
    print("=== Testing Poker Game Mechanics ===")
    print("This will play one hand with an untrained bot to test mechanics.\n")
    
    # Create game
    game = PokerGameEngine(num_players=2)  # Just 2 players for simplicity
    
    # Override to make both players bots for automated test
    game.players[0].is_bot = True
    game.players[1].is_bot = True
    
    print("Playing one automated hand between two untrained bots...")
    
    try:
        game.play_hand(verbose=True)
        print("\n✅ Game mechanics working correctly!")
        
        # Show final state
        print("\nFinal chip counts:")
        for player in game.players:
            print(f"Player {player.id}: {player.chips} chips")
            
    except Exception as e:
        print(f"\n❌ Error during game: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
