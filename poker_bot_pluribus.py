import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import random
from enum import Enum

# Utility function for robust action sampling
def sample_from_distribution(actions: list, probs: list):
    """Robust sampling that handles floating point issues"""
    if not actions:
        return None
    
    # Convert to numpy array
    probs = np.array(probs, dtype=np.float64)
    
    # Handle all zeros
    if probs.sum() == 0:
        return actions[0] if actions else None
    
    # Normalize
    probs = probs / probs.sum()
    
    # Use cumulative distribution for robust sampling
    cumsum = np.cumsum(probs)
    cumsum[-1] = 1.0  # Force last element to be exactly 1
    
    r = np.random.random()
    for i, cs in enumerate(cumsum):
        if r <= cs:
            return actions[i]
    
    # Fallback (should never reach here)
    return actions[-1]

# Core Game Types
class Action(Enum):
    FOLD = 0
    CALL = 1
    RAISE = 2

@dataclass
class GameState:
    """Represents current poker game state"""
    pot: int
    players_in_hand: Set[int]
    current_player: int
    board_cards: List[int]
    betting_round: int
    last_bet: int
    player_bets: Dict[int, int]
    player_chips: Dict[int, int]
    
@dataclass
class InfoSet:
    """Information set - what a player knows"""
    hole_cards: Tuple[int, int]
    board_cards: Tuple[int, ...]
    betting_history: str
    
    def __hash__(self):
        return hash((self.hole_cards, self.board_cards, self.betting_history))

# Abstraction Module
class AbstractionEngine:
    """Handles action and information abstraction"""
    
    def __init__(self):
        # Action abstraction: limited bet sizes
        self.bet_fractions = [0.33, 0.5, 0.75, 1.0, 1.5, 2.0]
        self.max_actions_per_node = 14
        
    def get_abstract_actions(self, state: GameState) -> List[Tuple[Action, int]]:
        """Returns available actions with abstracted bet sizes"""
        actions = [(Action.FOLD, 0)]
        
        if state.last_bet > state.player_bets.get(state.current_player, 0):
            actions.append((Action.CALL, state.last_bet))
        
        # Add limited raise sizes based on pot fractions
        min_raise = state.last_bet * 2
        pot_size = state.pot
        
        raise_sizes = []
        for fraction in self.bet_fractions:
            size = int(pot_size * fraction) + state.last_bet
            if size >= min_raise:
                raise_sizes.append(size)
        
        # Limit total actions
        raise_sizes = raise_sizes[:self.max_actions_per_node - len(actions)]
        actions.extend([(Action.RAISE, size) for size in raise_sizes])
        
        return actions
    
    def bucket_hand(self, hole_cards: Tuple[int, int], board: List[int]) -> int:
        """Information abstraction - bucket similar hands together"""
        # Simplified bucketing based on hand strength percentile
        # In practice, use more sophisticated clustering
        strength = self._estimate_hand_strength(hole_cards, board)
        return int(strength * 10)  # 10 buckets
    
    def _estimate_hand_strength(self, hole_cards: Tuple[int, int], board: List[int]) -> float:
        """Estimate hand strength (0-1) - simplified version"""
        # Real implementation would use Monte Carlo rollouts
        # This is a placeholder that considers high cards
        high_card = max(hole_cards)
        return high_card / 52.0

# Monte Carlo CFR Implementation
class LinearMCCFR:
    """Linear Monte Carlo Counterfactual Regret Minimization"""
    
    def __init__(self, abstraction: AbstractionEngine):
        self.abstraction = abstraction
        self.regret_sum = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))
        self.iteration = 0
        
    def train(self, num_iterations: int, num_players: int = 6):
        """Train blueprint strategy via self-play"""
        for i in range(num_iterations):
            self.iteration = i + 1
            
            # Linear CFR weighting
            weight = self.iteration
            
            # Sample a traverser
            traverser = i % num_players
            
            # Initialize game state
            state = self._initialize_game(num_players)
            
            # Run MCCFR iteration
            self._mccfr(state, traverser, 1.0, weight)
            
            # Skip negative regret actions 95% of the time (optimization)
            if random.random() < 0.05:
                self._prune_negative_regrets()
    
    def _mccfr(self, state: GameState, traverser: int, reach_prob: float, weight: float) -> float:
        """Monte Carlo CFR traversal"""
        if self._is_terminal(state):
            return self._evaluate(state, traverser)
        
        # Skip if current player is not in the hand
        if state.current_player not in state.players_in_hand:
            # Move to next active player
            state.current_player = self._next_active_player(state)
            if state.current_player == -1:  # No active players
                return self._evaluate(state, traverser)
        
        if state.current_player != traverser:
            # Sample opponent action
            strategy = self._get_strategy(state)
            action = self._sample_action(strategy)
            next_state = self._apply_action(state, action)
            return self._mccfr(next_state, traverser, reach_prob, weight)
        
        # Traverser's turn - explore all actions
        info_set = self._get_info_set(state, traverser)
        info_set_str = str(info_set)
        
        strategy = self._get_strategy(state)
        utilities = {}
        
        for action in self.abstraction.get_abstract_actions(state):
            next_state = self._apply_action(state, action)
            utilities[action] = self._mccfr(next_state, traverser, 
                                          reach_prob * strategy[action], weight)
        
        # Compute counterfactual values
        node_utility = sum(strategy[a] * utilities[a] for a in utilities)
        
        # Update regrets with linear weighting
        for action in utilities:
            regret = utilities[action] - node_utility
            self.regret_sum[info_set_str][action] += weight * regret
        
        # Update strategy sum
        for action in strategy:
            self.strategy_sum[info_set_str][action] += weight * reach_prob * strategy[action]
        
        return node_utility
    
    def _get_strategy(self, state: GameState) -> Dict[Tuple[Action, int], float]:
        """Get current strategy using regret matching"""
        info_set = self._get_info_set(state, state.current_player)
        info_set_str = str(info_set)
        
        actions = self.abstraction.get_abstract_actions(state)
        strategy = {}
        
        # Regret matching
        normalizing_sum = 0
        for action in actions:
            positive_regret = max(0, self.regret_sum[info_set_str][action])
            strategy[action] = positive_regret
            normalizing_sum += positive_regret
        
        # Normalize or use uniform strategy
        if normalizing_sum > 0:
            for action in actions:
                strategy[action] = strategy[action] / normalizing_sum
        else:
            # Uniform strategy when no positive regrets
            uniform_prob = 1.0 / len(actions)
            for action in actions:
                strategy[action] = uniform_prob
        
        return strategy
    
    def get_average_strategy(self, info_set: InfoSet) -> Dict[Tuple[Action, int], float]:
        """Get average strategy (Nash approximation)"""
        info_set_str = str(info_set)
        
        if info_set_str not in self.strategy_sum:
            return {}
        
        total = sum(self.strategy_sum[info_set_str].values())
        if total == 0:
            return {}
        
        return {a: s/total for a, s in self.strategy_sum[info_set_str].items()}
    
    def _initialize_game(self, num_players: int) -> GameState:
        """Initialize a new poker hand"""
        return GameState(
            pot=0,
            players_in_hand=set(range(num_players)),
            current_player=0,
            board_cards=[],
            betting_round=0,
            last_bet=0,
            player_bets={},
            player_chips={i: 10000 for i in range(num_players)}
        )
    
    def _get_info_set(self, state: GameState, player: int) -> InfoSet:
        """Get information set for a player"""
        # Simplified - real implementation tracks hole cards
        return InfoSet(
            hole_cards=(0, 0),  # Would be actual hole cards
            board_cards=tuple(state.board_cards),
            betting_history=self._encode_betting_history(state)
        )
    
    def _encode_betting_history(self, state: GameState) -> str:
        """Encode betting history as string"""
        # Simplified encoding
        return f"R{state.betting_round}B{state.last_bet}"
    
    def _is_terminal(self, state: GameState) -> bool:
        """Check if hand is complete"""
        return len(state.players_in_hand) == 1 or state.betting_round >= 4
    
    def _evaluate(self, state: GameState, player: int) -> float:
        """Evaluate terminal node utility for player"""
        # Simplified - would need full hand evaluation
        if player in state.players_in_hand:
            return state.pot / len(state.players_in_hand)
        return 0
    
    def _apply_action(self, state: GameState, action: Tuple[Action, int]) -> GameState:
        """Apply action to game state"""
        # Clone state and apply action
        # Simplified implementation
        new_state = GameState(
            pot=state.pot,
            players_in_hand=state.players_in_hand.copy(),  # Copy the set
            current_player=state.current_player,
            board_cards=state.board_cards.copy(),
            betting_round=state.betting_round,
            last_bet=state.last_bet,
            player_bets=state.player_bets.copy(),
            player_chips=state.player_chips.copy()
        )
        act_type, amount = action
        
        if act_type == Action.FOLD:
            new_state.players_in_hand.discard(state.current_player)  # Use discard to avoid KeyError
        elif act_type == Action.CALL:
            new_state.player_bets[state.current_player] = state.last_bet
            new_state.pot += state.last_bet
        elif act_type == Action.RAISE:
            new_state.player_bets[state.current_player] = amount
            new_state.last_bet = amount
            new_state.pot += amount
        
        # Move to next active player
        new_state.current_player = self._next_active_player(new_state)
        
        return new_state
    
    def _next_active_player(self, state: GameState) -> int:
        """Find next active player in hand"""
        if not state.players_in_hand:
            return -1
        
        # Start from next position
        pos = (state.current_player + 1) % 6
        
        # Find next player in hand
        for _ in range(6):
            if pos in state.players_in_hand:
                return pos
            pos = (pos + 1) % 6
        
        return -1  # No active players
    
    def _sample_action(self, strategy: Dict) -> Tuple[Action, int]:
        """Sample action from strategy"""
        actions = list(strategy.keys())
        probs = list(strategy.values())
        
        # Handle empty or all-zero strategies
        if not actions or sum(probs) == 0:
            # Return a default action
            return (Action.CALL, 0)
        
        # Normalize probabilities to ensure they sum to 1
        probs = np.array(probs, dtype=np.float64)
        probs = probs / probs.sum()
        
        # Fix any numerical issues
        probs = probs / probs.sum()  # Double normalization for precision
        
        idx = np.random.choice(len(actions), p=probs)
        return actions[idx]
    
    def _prune_negative_regrets(self):
        """Remove actions with very negative regret (optimization)"""
        for info_set in list(self.regret_sum.keys()):
            for action in list(self.regret_sum[info_set].keys()):
                if self.regret_sum[info_set][action] < -10000:
                    del self.regret_sum[info_set][action]

# Real-Time Search Module
class DepthLimitedSearch:
    """Real-time search with depth limits and opponent modeling"""
    
    def __init__(self, blueprint: LinearMCCFR, abstraction: AbstractionEngine):
        self.blueprint = blueprint
        self.abstraction = abstraction
        self.continuation_strategies = self._generate_continuation_strategies()
    
    def search(self, state: GameState, player: int, time_limit: float = 20.0) -> Tuple[Action, int]:
        """Perform real-time search to find best action"""
        # Build subgame
        subgame_root = self._build_subgame(state, player)
        
        # Solve subgame using Linear CFR
        subgame_solver = LinearMCCFR(self.abstraction)
        
        # Run iterations until time limit
        iterations = 0
        start_time = self._get_time()
        
        while self._get_time() - start_time < time_limit:
            subgame_solver.train(100, len(state.players_in_hand))
            iterations += 100
        
        # Extract action from solved subgame
        info_set = self._get_info_set(state, player)
        strategy = subgame_solver.get_average_strategy(info_set)
        
        # Sample from strategy
        return self._sample_action(strategy)
    
    def _build_subgame(self, state: GameState, player: int) -> GameState:
        """Build subgame rooted at current state"""
        # Include reach probabilities for all hands player could have
        # Simplified - full implementation tracks hand ranges
        return state
    
    def _generate_continuation_strategies(self) -> List[Dict]:
        """Generate k=4 continuation strategies"""
        return [
            self.blueprint,  # Original blueprint
            self._bias_strategy(self.blueprint, "fold"),
            self._bias_strategy(self.blueprint, "call"),
            self._bias_strategy(self.blueprint, "raise")
        ]
    
    def _bias_strategy(self, base_strategy: LinearMCCFR, bias: str) -> LinearMCCFR:
        """Create biased version of strategy"""
        # Modify base strategy to favor certain actions
        # Simplified implementation
        return base_strategy
    
    def _get_time(self) -> float:
        """Get current time in seconds"""
        import time
        return time.time()
    
    def _get_info_set(self, state: GameState, player: int) -> InfoSet:
        """Get information set for player"""
        return InfoSet(
            hole_cards=(0, 0),  # Would be actual hole cards
            board_cards=tuple(state.board_cards),
            betting_history=self._encode_betting_history(state)
        )
    
    def _encode_betting_history(self, state: GameState) -> str:
        """Encode betting history"""
        return f"R{state.betting_round}B{state.last_bet}"
    
    def _sample_action(self, strategy: Dict) -> Tuple[Action, int]:
        """Sample action from strategy"""
        if not strategy:
            # Default to check/call if no strategy
            return (Action.CALL, 0)
        
        actions = list(strategy.keys())
        probs = list(strategy.values())
        
        # Handle empty or all-zero strategies
        if sum(probs) == 0:
            return (Action.CALL, 0)
        
        # Normalize probabilities
        probs = np.array(probs, dtype=np.float64)
        probs = probs / probs.sum()
        
        # Fix any numerical issues
        probs = probs / probs.sum()  # Double normalization for precision
        
        idx = np.random.choice(len(actions), p=probs)
        return actions[idx]

# Main Poker Bot
class PluribusBot:
    """Main poker bot combining blueprint and real-time search"""
    
    def __init__(self):
        self.abstraction = AbstractionEngine()
        self.blueprint = LinearMCCFR(self.abstraction)
        self.search_engine = DepthLimitedSearch(self.blueprint, self.abstraction)
        self.hand_history = []
    
    def train_blueprint(self, iterations: int = 1000000):
        """Train offline blueprint strategy"""
        print(f"Training blueprint strategy with {iterations} iterations...")
        self.blueprint.train(iterations)
        print("Blueprint training complete")
    
    def get_action(self, state: GameState, player: int, 
                   use_search: bool = True) -> Tuple[Action, int]:
        """Get action for current game state"""
        
        # First betting round or small pots - use blueprint
        if state.betting_round == 0 or state.pot < 100:
            info_set = self._get_info_set(state, player)
            strategy = self.blueprint.get_average_strategy(info_set)
            
            if strategy:
                return self._sample_from_strategy(strategy)
            else:
                # Default to check/call
                return (Action.CALL, state.last_bet)
        
        # Later rounds or large pots - use real-time search
        if use_search:
            return self.search_engine.search(state, player)
        else:
            # Fall back to blueprint
            info_set = self._get_info_set(state, player)
            strategy = self.blueprint.get_average_strategy(info_set)
            return self._sample_from_strategy(strategy)
    
    def _get_info_set(self, state: GameState, player: int) -> InfoSet:
        """Get information set for player"""
        return InfoSet(
            hole_cards=(0, 0),  # Would track actual hole cards
            board_cards=tuple(state.board_cards),
            betting_history=self._encode_betting_history(state)
        )
    
    def _encode_betting_history(self, state: GameState) -> str:
        """Encode betting history"""
        return f"R{state.betting_round}B{state.last_bet}"
    
    def _sample_from_strategy(self, strategy: Dict) -> Tuple[Action, int]:
        """Sample action from strategy distribution"""
        if not strategy:
            return (Action.CALL, 0)
        
        actions = list(strategy.keys())
        probs = list(strategy.values())
        
        # Handle empty or all-zero strategies
        if sum(probs) == 0:
            return (Action.CALL, 0)
        
        # Normalize probabilities
        probs = np.array(probs, dtype=np.float64)
        probs = probs / probs.sum()
        
        # Fix any numerical issues
        probs = probs / probs.sum()  # Double normalization for precision
        
        idx = np.random.choice(len(actions), p=probs)
        return actions[idx]
    
    def save_blueprint(self, filename: str):
        """Save trained blueprint to file"""
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump({
                'regret_sum': dict(self.blueprint.regret_sum),
                'strategy_sum': dict(self.blueprint.strategy_sum),
                'iteration': self.blueprint.iteration
            }, f)
    
    def load_blueprint(self, filename: str):
        """Load trained blueprint from file"""
        import pickle
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.blueprint.regret_sum = defaultdict(lambda: defaultdict(float), data['regret_sum'])
            self.blueprint.strategy_sum = defaultdict(lambda: defaultdict(float), data['strategy_sum'])
            self.blueprint.iteration = data['iteration']

# Usage Example
if __name__ == "__main__":
    # Initialize bot
    bot = PluribusBot()
    
    # Train blueprint (in practice, use many more iterations)
    bot.train_blueprint(iterations=10000)
    
    # Example game state
    example_state = GameState(
        pot=150,
        players_in_hand={0, 1, 2, 3},
        current_player=0,
        board_cards=[10, 22, 35],  # Flop
        betting_round=1,
        last_bet=50,
        player_bets={1: 50, 2: 50},
        player_chips={i: 10000 for i in range(6)}
    )
    
    # Get bot's action
    action, amount = bot.get_action(example_state, player=0, use_search=True)
    print(f"Bot action: {action.name} ${amount}")
