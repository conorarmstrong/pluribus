#!/usr/bin/env python3
"""
Interactive script to play poker against Pluribus bot
"""

import os
from poker_game_engine import PokerGameEngine

def main():
    print("=== Pluribus Poker - Interactive Game ===")
    print("You are Player 0, all others are Pluribus bots")
    print("Starting chips: 10,000")
    print("Blinds: 50/100")
    print()
    
    # Initialize game with 1 human and 5 bots
    game = PokerGameEngine(num_players=6)
    
    # Check if pre-trained bot exists
    bot_file = "my_trained_bot.pkl"
    if os.path.exists(bot_file):
        print(f"Loading trained bot from {bot_file}...")
        game.bot.load_blueprint(bot_file)
        print("Trained bot loaded!")
    else:
        print("No trained bot found. Bot will play with random strategy.")
        print("Run 'python train.py' first for better bot performance.")
    
    print("\nLet's play poker!\n")
    
    # Game loop
    hand_count = 0
    while True:
        hand_count += 1
        print(f"\n{'='*50}")
        print(f"HAND #{hand_count}")
        print(f"{'='*50}")
        
        game.play_hand(verbose=True)
        
        # Check if human player is broke
        if game.players[0].chips <= 0:
            print("\n*** You're out of chips! Game over. ***")
            break
        
        # Show current standings
        print(f"\nCurrent chip counts:")
        for i, player in enumerate(game.players):
            if player.chips > 0:
                profit = player.chips - 10000
                sign = "+" if profit >= 0 else ""
                print(f"Player {i}: {player.chips:,} chips ({sign}{profit:,})")
        
        # Ask if want to continue
        print()
        if input("Play another hand? (y/n): ").lower() != 'y':
            break
    
    # Final statistics
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    
    total_hands = hand_count
    human_player = game.players[0]
    final_chips = human_player.chips
    profit = final_chips - 10000
    
    print(f"Hands played: {total_hands}")
    print(f"Your final chips: {final_chips:,}")
    print(f"Net profit/loss: {'+' if profit >= 0 else ''}{profit:,}")
    
    if profit > 0:
        print("\nCongratulations! You beat Pluribus!")
    elif profit < 0:
        print("\nPluribUS wins this session. Try again!")
    else:
        print("\nA perfect tie!")

if __name__ == "__main__":
    main()
