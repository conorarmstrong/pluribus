import numpy as np
from typing import List, Tuple, Dict, Set
from itertools import combinations
from collections import Counter

# Card representation: 0-51 (0-12: 2-A of spades, 13-25: hearts, etc.)
RANKS = '23456789TJQKA'
SUITS = 'shdc'

def card_to_string(card: int) -> str:
    """Convert card number to string representation"""
    rank = RANKS[card % 13]
    suit = SUITS[card // 13]
    return f"{rank}{suit}"

def string_to_card(s: str) -> int:
    """Convert string to card number"""
    rank = RANKS.index(s[0])
    suit = SUITS.index(s[1])
    return suit * 13 + rank

class HandEvaluator:
    """Fast poker hand evaluation using lookup tables"""
    
    def __init__(self):
        self._init_lookup_tables()
    
    def _init_lookup_tables(self):
        """Initialize lookup tables for fast hand evaluation"""
        # Precompute hand rankings
        self.rank_values = {r: i for i, r in enumerate(RANKS)}
        self.hand_rankings = {
            'high_card': 0,
            'pair': 1,
            'two_pair': 2,
            'three_kind': 3,
            'straight': 4,
            'flush': 5,
            'full_house': 6,
            'four_kind': 7,
            'straight_flush': 8
        }
    
    def evaluate_hand(self, hole_cards: List[int], board: List[int]) -> int:
        """Evaluate poker hand strength (higher is better)"""
        all_cards = hole_cards + board
        
        if len(all_cards) < 5:
            return 0
        
        # Find best 5-card combination
        best_rank = 0
        
        for combo in combinations(all_cards, 5):
            rank = self._evaluate_5_cards(list(combo))
            best_rank = max(best_rank, rank)
        
        return best_rank
    
    def _evaluate_5_cards(self, cards: List[int]) -> int:
        """Evaluate exactly 5 cards"""
        ranks = [c % 13 for c in cards]
        suits = [c // 13 for c in cards]
        
        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)
        
        is_flush = max(suit_counts.values()) == 5
        is_straight = self._is_straight(sorted(ranks))
        
        # Check hand types from best to worst
        if is_flush and is_straight:
            return self._make_hand_value(8, ranks)  # Straight flush
        
        if 4 in rank_counts.values():
            return self._make_hand_value(7, ranks)  # Four of a kind
        
        if 3 in rank_counts.values() and 2 in rank_counts.values():
            return self._make_hand_value(6, ranks)  # Full house
        
        if is_flush:
            return self._make_hand_value(5, ranks)  # Flush
        
        if is_straight:
            return self._make_hand_value(4, ranks)  # Straight
        
        if 3 in rank_counts.values():
            return self._make_hand_value(3, ranks)  # Three of a kind
        
        pairs = [r for r, c in rank_counts.items() if c == 2]
        if len(pairs) == 2:
            return self._make_hand_value(2, ranks)  # Two pair
        
        if len(pairs) == 1:
            return self._make_hand_value(1, ranks)  # One pair
        
        return self._make_hand_value(0, ranks)  # High card
    
    def _is_straight(self, sorted_ranks: List[int]) -> bool:
        """Check if ranks form a straight"""
        # Check regular straight
        for i in range(1, 5):
            if sorted_ranks[i] != sorted_ranks[i-1] + 1:
                break
        else:
            return True
        
        # Check A-2-3-4-5 straight
        if sorted_ranks == [0, 1, 2, 3, 12]:
            return True
        
        return False
    
    def _make_hand_value(self, hand_type: int, ranks: List[int]) -> int:
        """Create comparable hand value"""
        # Combine hand type with kickers for tie-breaking
        value = hand_type * 10**10
        
        # Add kicker values
        rank_counts = Counter(ranks)
        sorted_ranks = sorted(rank_counts.items(), 
                            key=lambda x: (x[1], x[0]), reverse=True)
        
        for i, (rank, _) in enumerate(sorted_ranks[:5]):
            value += rank * (10 ** (8 - i*2))
        
        return value

class AdvancedAbstraction:
    """Advanced abstraction techniques from Pluribus"""
    
    def __init__(self):
        self.evaluator = HandEvaluator()
        self.preflop_buckets = self._compute_preflop_buckets()
    
    def _compute_preflop_buckets(self) -> Dict[Tuple[int, int], int]:
        """Compute preflop hand buckets using k-means clustering"""
        # Group strategically similar starting hands
        buckets = {}
        
        # Simplified bucketing - real implementation uses equity calculations
        # Premium hands
        premium = [(12, 12), (11, 11), (10, 10), (12, 11)]  # AA, KK, QQ, AK
        for hand in premium:
            buckets[hand] = 0
        
        # Strong hands
        strong = [(9, 9), (8, 8), (12, 10), (12, 9)]  # 99, 88, AQ, AJ
        for hand in strong:
            buckets[hand] = 1
        
        # Medium pairs and broadway
        medium = [(7, 7), (6, 6), (11, 10), (10, 9)]
        for hand in medium:
            buckets[hand] = 2
        
        # Default bucket for others
        return buckets
    
    def get_information_abstraction(self, hole_cards: Tuple[int, int], 
                                  board: List[int], round: int) -> int:
        """Get abstracted information set bucket"""
        if round == 0:  # Preflop
            # Use precomputed buckets
            normalized = self._normalize_hole_cards(hole_cards)
            return self.preflop_buckets.get(normalized, 3)
        else:
            # Use percentile bucketing based on equity
            equity = self._estimate_equity(hole_cards, board)
            
            # More buckets for later streets
            num_buckets = 10 if round == 1 else 20
            return int(equity * num_buckets)
    
    def _normalize_hole_cards(self, hole_cards: Tuple[int, int]) -> Tuple[int, int]:
        """Normalize hole cards by rank only"""
        r1, r2 = hole_cards[0] % 13, hole_cards[1] % 13
        return (max(r1, r2), min(r1, r2))
    
    def _estimate_equity(self, hole_cards: Tuple[int, int], 
                        board: List[int], samples: int = 100) -> float:
        """Estimate hand equity using Monte Carlo simulation"""
        wins = 0
        
        used_cards = set(hole_cards + board)
        remaining_cards = [c for c in range(52) if c not in used_cards]
        
        for _ in range(samples):
            # Sample opponent cards and remaining board
            sample = np.random.choice(remaining_cards, 
                                    size=2 + (5 - len(board)), 
                                    replace=False)
            
            opp_cards = sample[:2]
            new_board = board + list(sample[2:])
            
            my_strength = self.evaluator.evaluate_hand(list(hole_cards), new_board)
            opp_strength = self.evaluator.evaluate_hand(list(opp_cards), new_board)
            
            if my_strength > opp_strength:
                wins += 1
            elif my_strength == opp_strength:
                wins += 0.5
        
        return wins / samples

class PseudoHarmonicMapping:
    """Map off-tree opponent actions to nearest on-tree action"""
    
    def __init__(self, abstract_sizes: List[int]):
        self.abstract_sizes = abstract_sizes
    
    def map_bet(self, actual_bet: int, pot: int) -> int:
        """Map actual bet to nearest abstract bet size"""
        # Find closest abstract bet as fraction of pot
        actual_fraction = actual_bet / max(pot, 1)
        
        best_distance = float('inf')
        best_bet = actual_bet
        
        for abstract_fraction in [0.33, 0.5, 0.75, 1.0, 1.5, 2.0]:
            abstract_bet = int(pot * abstract_fraction)
            distance = abs(actual_fraction - abstract_fraction)
            
            if distance < best_distance:
                best_distance = distance
                best_bet = abstract_bet
        
        return best_bet

class ExploitabilityCalculator:
    """Calculate exploitability of a strategy profile"""
    
    def __init__(self, game_tree):
        self.game_tree = game_tree
    
    def calculate_best_response(self, strategy_profile: Dict, player: int) -> float:
        """Calculate best response value for a player"""
        # This would traverse game tree to find optimal counter-strategy
        # Returns expected value of best response
        pass
    
    def calculate_exploitability(self, strategy_profile: Dict) -> float:
        """Calculate total exploitability (Nash distance)"""
        total_exploitability = 0
        
        for player in range(6):
            br_value = self.calculate_best_response(strategy_profile, player)
            nash_value = self._get_nash_value(player)
            exploitability = br_value - nash_value
            total_exploitability += max(0, exploitability)
        
        return total_exploitability / 6

class MemoryEfficientStorage:
    """Compressed storage for blueprint strategy"""
    
    def __init__(self):
        self.compression_threshold = 0.001
    
    def compress_strategy(self, strategy_sum: Dict[str, Dict]) -> bytes:
        """Compress strategy by removing near-zero probabilities"""
        compressed = {}
        
        for info_set, action_probs in strategy_sum.items():
            # Normalize
            total = sum(action_probs.values())
            if total == 0:
                continue
            
            # Keep only significant actions
            compressed[info_set] = {
                a: p/total 
                for a, p in action_probs.items() 
                if p/total > self.compression_threshold
            }
        
        # Further compress using gzip
        import gzip
        import pickle
        return gzip.compress(pickle.dumps(compressed))
    
    def decompress_strategy(self, compressed_data: bytes) -> Dict:
        """Decompress strategy data"""
        import gzip
        import pickle
        return pickle.loads(gzip.decompress(compressed_data))

# Performance Metrics
class PerformanceTracker:
    """Track bot performance using milli big blinds per game"""
    
    def __init__(self):
        self.hands_played = 0
        self.total_winnings = 0
        self.bb_size = 100  # Big blind size
    
    def record_hand(self, winnings: int):
        """Record result of a hand"""
        self.hands_played += 1
        self.total_winnings += winnings
    
    def get_bb_per_100(self) -> float:
        """Get big blinds won per 100 hands"""
        if self.hands_played == 0:
            return 0
        
        bb_won = self.total_winnings / self.bb_size
        return (bb_won / self.hands_played) * 100
    
    def get_mbb_per_game(self) -> float:
        """Get milli big blinds per game (standard metric)"""
        return self.get_bb_per_100() * 10

# Example usage of advanced components
if __name__ == "__main__":
    # Test hand evaluator
    evaluator = HandEvaluator()
    
    # Royal flush
    hole = [string_to_card('As'), string_to_card('Ks')]
    board = [string_to_card('Qs'), string_to_card('Js'), 
             string_to_card('Ts'), string_to_card('2h'), string_to_card('3d')]
    
    strength = evaluator.evaluate_hand(hole, board)
    print(f"Royal flush strength: {strength}")
    
    # Test equity calculation
    abstraction = AdvancedAbstraction()
    equity = abstraction._estimate_equity(
        (string_to_card('Ah'), string_to_card('Ad')),
        [string_to_card('Kh'), string_to_card('Qd'), string_to_card('2c')]
    )
    print(f"AA equity on KQ2 board: {equity:.2%}")
    
    # Test performance tracking
    tracker = PerformanceTracker()
    # Simulate some results
    for _ in range(100):
        tracker.record_hand(np.random.normal(10, 200))  # Random P&L
    
    print(f"Performance: {tracker.get_mbb_per_game():.1f} mbb/game")
