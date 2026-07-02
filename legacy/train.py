#!/usr/bin/env python3
"""
Training script for Pluribus poker bot
"""

import sys
import time
from poker_bot_pluribus import PluribusBot
from poker_game_engine import TrainingHarness

def main():
    print("=== Pluribus Poker Bot Training ===")
    print("Starting training process...")
    
    # Create bot
    bot = PluribusBot()
    
    # First train blueprint with fewer iterations for testing
    print("\nPhase 1: Training blueprint strategy...")
    print("(This is a simplified training for testing - use more iterations for competitive play)")
    start_time = time.time()
    
    # Train with very few iterations for initial testing
    try:
        bot.train_blueprint(iterations=100)  # Very quick training for testing
    except Exception as e:
        print(f"Error during blueprint training: {e}")
        print("Continuing with untrained blueprint...")
    
    blueprint_time = time.time() - start_time
    print(f"Blueprint training completed in {blueprint_time:.2f} seconds")
    
    # Save checkpoint
    bot.save_blueprint("blueprint_checkpoint.pkl")
    
    # Create training harness for self-play
    harness = TrainingHarness(bot)
    
    # Training parameters
    num_hands = 10  # Very few hands for initial test
    
    print(f"\nPhase 2: Self-play training with {num_hands} hands...")
    start_time = time.time()
    
    # Override the harness train method to not re-train blueprint
    harness.bot = bot  # Use already trained bot
    
    # Play hands for experience
    try:
        for hand_num in range(num_hands):
            print(f"Playing hand {hand_num + 1}/{num_hands}")
            harness.game.play_hand(verbose=False)
    except Exception as e:
        print(f"Error during self-play: {e}")
        print("Saving current progress...")
    
    # Calculate training time
    elapsed_time = time.time() - start_time
    print(f"\nSelf-play completed in {elapsed_time:.2f} seconds")
    if num_hands > 0:
        print(f"Average time per hand: {elapsed_time/num_hands:.3f} seconds")
    
    # Save trained bot
    filename = "my_trained_bot.pkl"
    bot.save_blueprint(filename)
    print(f"\nTrained bot saved to: {filename}")
    
    # Display some statistics
    print(f"\nTraining Statistics:")
    print(f"Blueprint iterations: {bot.blueprint.iteration}")
    print(f"Strategy nodes: {len(bot.blueprint.strategy_sum)}")
    print(f"Regret nodes: {len(bot.blueprint.regret_sum)}")
    
    print("\n✅ Training complete!")
    print("\nFor stronger play, run again with more iterations:")
    print("- Testing: bot.train_blueprint(iterations=1000)")
    print("- Competitive: bot.train_blueprint(iterations=1000000)")
    print("- Professional: bot.train_blueprint(iterations=10000000)")

if __name__ == "__main__":
    main()
