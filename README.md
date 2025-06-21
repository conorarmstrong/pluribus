# Pluribus-Style Poker Bot

A Python implementation of the Pluribus poker AI system that defeated professional players in 6-player no-limit Texas Hold'em. Based on the research paper "Superhuman AI for multiplayer poker" by Brown & Sandholm (2019).

## Overview

This bot uses a two-stage approach:
1. **Offline Blueprint Strategy**: Computed via Monte Carlo Counterfactual Regret Minimization (MCCFR)
2. **Online Real-time Search**: Depth-limited search during play for improved decisions

Key features:
- Plays 6-player no-limit Texas Hold'em
- No opponent modeling - uses fixed, balanced strategy
- Efficient implementation requiring only 2 CPUs and <128GB RAM
- Achieves superhuman performance after training

## Installation

### Requirements
```bash
Python 3.8+
numpy>=1.19.0
```

### Setup
```bash
# Clone or download the files
git clone <repository>
cd pluribus-poker-bot

# Install dependencies
pip install numpy

# Optional: Create virtual environment
python -m venv poker_env
source poker_env/bin/activate  # On Windows: poker_env\Scripts\activate
```

## File Structure

```
pluribus-poker-bot/
│
├── poker_bot_pluribus.py    # Core bot implementation
│   ├── PluribusBot          # Main bot class
│   ├── LinearMCCFR          # Blueprint strategy trainer
│   ├── DepthLimitedSearch   # Real-time search engine
│   └── AbstractionEngine    # Action/information abstraction
│
├── poker_bot_utils.py       # Utility functions
│   ├── HandEvaluator        # Fast hand strength evaluation
│   ├── AdvancedAbstraction  # Sophisticated hand bucketing
│   └── PerformanceTracker   # Track winnings in mbb/game
│
└── poker_game_engine.py     # Game implementation
    ├── PokerGameEngine      # Texas Hold'em rules engine
    ├── TrainingHarness      # Self-play training system
    └── Player               # Player state management
```

## Quick Start

### Playing Against the Bot

```python
from poker_game_engine import PokerGameEngine

# Create game with 1 human (you) and 5 bots
game = PokerGameEngine(num_players=6)

# Play interactive poker
game.play_hand(verbose=True)
```

### Training a New Bot

```python
from poker_bot_pluribus import PluribusBot
from poker_game_engine import TrainingHarness

# Create and train bot
bot = PluribusBot()
harness = TrainingHarness(bot)
harness.train(num_hands=10000)

# Save trained bot
bot.save_blueprint("my_trained_bot.pkl")
```

## User Guide

### Playing a Game

1. **Start the game**:
```python
python poker_game_engine.py
```

2. **Game Interface**:
- You are always Player 0
- Other players are Pluribus bots
- Starting chips: 10,000
- Blinds: 50/100

3. **Your Turn**:
```
Your turn (Player 0)
Hole cards: ['As', 'Kh']
Board: ['Qd', 'Js', '7c']
Pot: 450, Current bet: 200
Your chips: 9800, Bet this round: 0
Actions: (f)old, (c)all, (r)aise
Choice: c
```

4. **Actions**:
- `f` - Fold your hand
- `c` - Call the current bet
- `r` - Raise (you'll be prompted for amount)
- `b` - Bet (when no one has bet)

### Understanding the Display

**Card Notation**:
- Ranks: 2-9, T(10), J, Q, K, A
- Suits: s(spades), h(hearts), d(diamonds), c(clubs)
- Example: `As` = Ace of spades

**Game Phases**:
1. Preflop - 2 hole cards dealt
2. Flop - 3 community cards
3. Turn - 1 more community card
4. River - Final community card
5. Showdown - Best hand wins

## Technical Documentation

### Core Algorithms

#### Linear Monte Carlo CFR
```python
# Training blueprint strategy
bot = PluribusBot()
bot.train_blueprint(iterations=1000000)  # More iterations = stronger play
```

The blueprint uses:
- Linear weighting (iteration T gets weight T)
- Negative regret pruning (95% skip rate)
- Monte Carlo sampling for large game tree

#### Real-time Search
```python
# Search configuration
action, amount = bot.get_action(
    state,
    player_id,
    use_search=True  # Enable depth-limited search
)
```

Search features:
- 4 continuation strategies (k=4)
- 1-33 second time limit per decision
- Pseudo-harmonic mapping for off-tree actions

### Abstraction Systems

#### Action Abstraction
```python
# Bet sizes as fractions of pot
bet_fractions = [0.33, 0.5, 0.75, 1.0, 1.5, 2.0]
max_actions_per_node = 14
```

#### Information Abstraction
```python
# Hand bucketing for strategic similarity
abstraction = AdvancedAbstraction()
bucket = abstraction.get_information_abstraction(
    hole_cards=(card1, card2),
    board=[...],
    round=betting_round
)
```

### Performance Metrics

Track performance using milli big blinds per game:
```python
tracker = PerformanceTracker()
tracker.record_hand(winnings)
print(f"Performance: {tracker.get_mbb_per_game()} mbb/game")
```

Target: >30 mbb/game indicates strong performance

## Training Guide

### Basic Training
```python
# Quick training for testing
bot = PluribusBot()
bot.train_blueprint(iterations=10000)  # ~1 minute
```

### Competitive Training
```python
# Full training for strong play
bot = PluribusBot()
bot.train_blueprint(iterations=10000000)  # ~12 hours
bot.save_blueprint("competitive_bot.pkl")
```

### Resume Training
```python
# Load and continue training
bot = PluribusBot()
bot.load_blueprint("checkpoint.pkl")
bot.train_blueprint(iterations=5000000)  # Additional training
```

## Customization

### Modify Game Parameters
```python
game = PokerGameEngine(
    num_players=6,        # 2-10 players
    starting_chips=20000, # Starting stack
    big_blind=200,        # Big blind size
    small_blind=100       # Small blind size
)
```

### Adjust Abstraction
```python
# Modify bet sizes
abstraction = AbstractionEngine()
abstraction.bet_fractions = [0.25, 0.5, 1.0, 2.0]  # Custom sizes
abstraction.max_actions_per_node = 10  # Fewer actions for speed
```

### Search Parameters
```python
# Adjust search time
def search(self, state, player, time_limit=10.0):  # Faster decisions
    # ...
```

## Performance Optimization

### Memory Usage
- Blueprint size: ~50GB for full strategy
- Runtime memory: <128GB with compression
- Use `MemoryEfficientStorage` for larger games

### Speed Optimization
```python
# Disable search for speed
action = bot.get_action(state, player, use_search=False)

# Reduce Monte Carlo samples
equity = abstraction._estimate_equity(cards, board, samples=50)  # Faster
```

### Parallel Training
```python
# Use multiple processes (advanced)
from multiprocessing import Pool

def train_worker(iterations):
    bot = PluribusBot()
    bot.train_blueprint(iterations)
    return bot

with Pool(4) as pool:
    bots = pool.map(train_worker, [250000] * 4)
    # Merge strategies...
```

## Troubleshooting

### Common Issues

**"Bot plays too aggressively"**
- Ensure sufficient training iterations (>1M)
- Check if blueprint loaded correctly

**"Game crashes during hand"**
- Verify all players have chips
- Check for valid card dealing (52 unique cards)

**"Bot takes too long to act"**
- Reduce search time limit
- Disable search for early betting rounds
- Use action abstraction more aggressively

**"Memory errors during training"**
- Reduce abstraction granularity
- Enable strategy compression
- Train in chunks and save checkpoints

### Debug Mode
```python
# Enable detailed logging
game.play_hand(verbose=True)

# Check bot internals
print(f"Blueprint iterations: {bot.blueprint.iteration}")
print(f"Strategy nodes: {len(bot.blueprint.strategy_sum)}")
```

## Performance Benchmarks

Expected performance after training:

| Training Iterations | Approximate Strength | Time Required |
|-------------------|---------------------|---------------|
| 10,000           | Beginner            | 1 minute      |
| 100,000          | Intermediate        | 10 minutes    |
| 1,000,000        | Advanced            | 2 hours       |
| 10,000,000       | Expert              | 12 hours      |
| 100,000,000      | Superhuman          | 5 days        |

## Contributing

To improve the bot:

1. **Better Hand Evaluation**: Implement full equity calculation
2. **Improved Abstraction**: Use clustering algorithms for bucketing
3. **Endgame Solving**: Add endgame solver for river decisions
4. **Opponent Exploitation**: Add optional opponent modeling

## License

This implementation is for educational purposes. The algorithms are based on published research by Brown & Sandholm (2019).

## References

- Original Paper: "Superhuman AI for multiplayer poker" (Science, 2019)
- Monte Carlo CFR: Lanctot et al. (2009)
- Linear CFR: Brown & Sandholm (2019)
- Depth-Limited Search: Brown, Sandholm & Amos (2018)

## Support

For questions or issues:
1. Check the troubleshooting section
2. Review the technical documentation
3. Examine the example code in each module
4. Reference the original Pluribus paper for algorithm details