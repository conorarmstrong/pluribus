import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import random
import pickle
import gzip
from enum import Enum

from poker_bot_utils import HandEvaluator


def sample_from_distribution(actions: list, probs: list):
    """Robust sampling that handles floating point issues"""
    if not actions:
        return None

    probs = np.array(probs, dtype=np.float64)

    if probs.sum() == 0:
        return actions[0]

    probs = probs / probs.sum()

    cumsum = np.cumsum(probs)
    cumsum[-1] = 1.0

    r = np.random.random()
    for i, cs in enumerate(cumsum):
        if r <= cs:
            return actions[i]

    return actions[-1]


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
    hole_cards: Dict[int, List[int]] = field(default_factory=dict)
    deck: List[int] = field(default_factory=list)
    players_acted: Set[int] = field(default_factory=set)

    def __hash__(self):
        return id(self)


@dataclass
class InfoSet:
    """Information set - what a player knows"""
    hole_cards: Tuple[int, int]
    board_cards: Tuple[int, ...]
    betting_history: str

    def __hash__(self):
        return hash((self.hole_cards, self.board_cards, self.betting_history))


class AbstractionEngine:
    """Handles action and information abstraction"""

    def __init__(self):
        self.bet_fractions = [0.33, 0.5, 0.75, 1.0, 1.5, 2.0]
        self.max_actions_per_node = 14
        self._evaluator = HandEvaluator()

    def get_abstract_actions(self, state: GameState) -> List[Tuple[Action, int]]:
        """Returns available actions with abstracted bet sizes"""
        actions = [(Action.FOLD, 0)]

        player_bet = state.player_bets.get(state.current_player, 0)
        if state.last_bet > player_bet:
            actions.append((Action.CALL, state.last_bet))

        min_raise = state.last_bet * 2 if state.last_bet > 0 else 100
        pot_size = state.pot

        raise_sizes = []
        for fraction in self.bet_fractions:
            size = int(pot_size * fraction) + state.last_bet
            if size >= min_raise:
                raise_sizes.append(size)

        raise_sizes = raise_sizes[:self.max_actions_per_node - len(actions)]
        actions.extend([(Action.RAISE, size) for size in raise_sizes])

        return actions

    def bucket_hand(self, hole_cards: Tuple[int, int], board: List[int]) -> int:
        """Information abstraction - bucket similar hands together"""
        strength = self._estimate_hand_strength(hole_cards, board)
        return int(strength * 10)

    def _estimate_hand_strength(self, hole_cards: Tuple[int, int], board: List[int]) -> float:
        """Estimate hand strength (0-1) using hand evaluator when possible"""
        if len(hole_cards) + len(board) < 5:
            # Not enough cards for a full evaluation; use high-card heuristic
            return max(hole_cards) / 52.0
        strength = self._evaluator.evaluate_hand(list(hole_cards), board)
        # hand_type occupies the top decimal digits (0-8); normalize to 0-1
        hand_type = strength // 10 ** 10
        return hand_type / 8.0


class LinearMCCFR:
    """Linear Monte Carlo Counterfactual Regret Minimization"""

    def __init__(self, abstraction: AbstractionEngine):
        self.abstraction = abstraction
        self.evaluator = HandEvaluator()
        self.regret_sum: Dict[str, Dict] = defaultdict(lambda: defaultdict(float))
        self.strategy_sum: Dict[str, Dict] = defaultdict(lambda: defaultdict(float))
        self.iteration = 0
        self.num_players = 6

    def train(self, num_iterations: int, num_players: int = 6):
        """Train blueprint strategy via self-play"""
        self.num_players = num_players
        for i in range(num_iterations):
            self.iteration = i + 1
            weight = self.iteration
            traverser = i % num_players
            state = self._initialize_game(num_players)
            self._mccfr(state, traverser, 1.0, weight)

            if random.random() < 0.05:
                self._prune_negative_regrets()

    def _mccfr(self, state: GameState, traverser: int,
               reach_prob: float, weight: float) -> float:
        """Monte Carlo CFR traversal"""
        if self._is_terminal(state):
            return self._evaluate(state, traverser)

        if state.current_player not in state.players_in_hand:
            state.current_player = self._next_active_player(state)
            if state.current_player == -1:
                return self._evaluate(state, traverser)

        if state.current_player != traverser:
            strategy = self._get_strategy(state)
            action = self._sample_action(strategy)
            next_state = self._apply_action(state, action)
            return self._mccfr(next_state, traverser, reach_prob, weight)

        info_set_str = str(self._get_info_set(state, traverser))
        strategy = self._get_strategy(state)
        utilities = {}

        for action in self.abstraction.get_abstract_actions(state):
            next_state = self._apply_action(state, action)
            utilities[action] = self._mccfr(
                next_state, traverser,
                reach_prob * strategy.get(action, 0.0), weight
            )

        node_utility = sum(strategy.get(a, 0.0) * utilities[a] for a in utilities)

        for action in utilities:
            regret = utilities[action] - node_utility
            self.regret_sum[info_set_str][action] += weight * regret

        for action in strategy:
            self.strategy_sum[info_set_str][action] += (
                weight * reach_prob * strategy[action]
            )

        return node_utility

    def _get_strategy(self, state: GameState) -> Dict[Tuple[Action, int], float]:
        """Get current strategy using regret matching"""
        info_set_str = str(self._get_info_set(state, state.current_player))
        actions = self.abstraction.get_abstract_actions(state)
        strategy = {}

        normalizing_sum = 0.0
        for action in actions:
            positive_regret = max(0.0, self.regret_sum[info_set_str][action])
            strategy[action] = positive_regret
            normalizing_sum += positive_regret

        if normalizing_sum > 0:
            for action in actions:
                strategy[action] /= normalizing_sum
        else:
            uniform = 1.0 / len(actions)
            for action in actions:
                strategy[action] = uniform

        return strategy

    def get_average_strategy(self, info_set: InfoSet) -> Dict[Tuple[Action, int], float]:
        """Get average strategy (Nash approximation)"""
        info_set_str = str(info_set)

        if info_set_str not in self.strategy_sum:
            return {}

        total = sum(self.strategy_sum[info_set_str].values())
        if total == 0:
            return {}

        return {a: s / total for a, s in self.strategy_sum[info_set_str].items()}

    def _initialize_game(self, num_players: int) -> GameState:
        """Initialize a new poker hand with dealt cards and blinds"""
        deck = list(range(52))
        random.shuffle(deck)

        # Deal 2 hole cards per player
        hole_cards: Dict[int, List[int]] = {}
        for i in range(num_players):
            hole_cards[i] = [deck.pop(), deck.pop()]

        # Post blinds: player 1 = SB, player 2 = BB (player 0 = dealer)
        player_chips = {i: 10000 for i in range(num_players)}
        player_bets = {i: 0 for i in range(num_players)}

        sb = 1 % num_players
        bb = 2 % num_players

        sb_amount = min(50, player_chips[sb])
        player_chips[sb] -= sb_amount
        player_bets[sb] = sb_amount

        bb_amount = min(100, player_chips[bb])
        player_chips[bb] -= bb_amount
        player_bets[bb] = bb_amount

        pot = sb_amount + bb_amount

        # Preflop: first to act is UTG (player 3, or player 0 in short-handed)
        first_to_act = 3 % num_players

        return GameState(
            pot=pot,
            players_in_hand=set(range(num_players)),
            current_player=first_to_act,
            board_cards=[],
            betting_round=0,
            last_bet=bb_amount,
            player_bets=player_bets,
            player_chips=player_chips,
            hole_cards=hole_cards,
            deck=deck,
            players_acted=set(),  # Nobody has voluntarily acted yet
        )

    def _get_info_set(self, state: GameState, player: int) -> InfoSet:
        """Get information set for a player using their actual hole cards"""
        raw = state.hole_cards.get(player, [0, 0])
        hole_cards = tuple(sorted(raw))
        return InfoSet(
            hole_cards=hole_cards,
            board_cards=tuple(state.board_cards),
            betting_history=self._encode_betting_history(state),
        )

    def _encode_betting_history(self, state: GameState) -> str:
        """Encode betting history as a canonical string"""
        bets = tuple(sorted(state.player_bets.items()))
        return f"R{state.betting_round}|L{state.last_bet}|B{bets}"

    def _is_terminal(self, state: GameState) -> bool:
        """Check if hand is complete"""
        return len(state.players_in_hand) <= 1 or state.betting_round >= 4

    def _evaluate(self, state: GameState, player: int) -> float:
        """Evaluate terminal node utility for player using actual hand strengths"""
        if not state.players_in_hand:
            return 0.0

        if len(state.players_in_hand) == 1:
            winner = next(iter(state.players_in_hand))
            return float(state.pot) if winner == player else 0.0

        # Showdown: find winner(s) by evaluating hands
        best_strength = -1
        winners: List[int] = []

        for p in state.players_in_hand:
            cards = state.hole_cards.get(p, [])
            if not cards:
                continue
            strength = self.evaluator.evaluate_hand(cards, state.board_cards)
            if strength > best_strength:
                best_strength = strength
                winners = [p]
            elif strength == best_strength:
                winners.append(p)

        if player in winners:
            return state.pot / len(winners)
        return 0.0

    def _is_round_complete(self, state: GameState) -> bool:
        """Check if the current betting round is over"""
        if len(state.players_in_hand) <= 1:
            return True

        # Every remaining player must have had a chance to act
        if not state.players_in_hand.issubset(state.players_acted):
            return False

        # Every remaining player must have matched the current bet (or be all-in)
        for p in state.players_in_hand:
            if (state.player_bets.get(p, 0) < state.last_bet
                    and state.player_chips.get(p, 0) > 0):
                return False

        return True

    def _advance_round(self, state: GameState) -> GameState:
        """Deal community cards and set up the next betting round"""
        new_deck = state.deck.copy()
        new_board = state.board_cards.copy()
        next_round = state.betting_round + 1

        if next_round == 1:       # Flop: deal 3 cards
            new_board.extend(new_deck[:3])
            new_deck = new_deck[3:]
        elif next_round == 2:     # Turn: deal 1 card
            new_board.append(new_deck[0])
            new_deck = new_deck[1:]
        elif next_round == 3:     # River: deal 1 card
            new_board.append(new_deck[0])
            new_deck = new_deck[1:]
        # next_round == 4: showdown, no new cards

        new_state = GameState(
            pot=state.pot,
            players_in_hand=state.players_in_hand.copy(),
            current_player=-1,
            board_cards=new_board,
            betting_round=next_round,
            last_bet=0,
            player_bets={p: 0 for p in state.player_chips},
            player_chips=state.player_chips.copy(),
            hole_cards=state.hole_cards,
            deck=new_deck,
            players_acted=set(),
        )

        # Set first active player post-flop (starts from seat 1, after dealer at 0)
        if next_round < 4:
            num_players = len(new_state.player_chips)
            for offset in range(num_players):
                pos = (1 + offset) % num_players
                if pos in new_state.players_in_hand:
                    new_state.current_player = pos
                    break

        return new_state

    def _apply_action(self, state: GameState,
                      action: Tuple[Action, int]) -> GameState:
        """Apply action and advance round if betting is complete"""
        new_state = GameState(
            pot=state.pot,
            players_in_hand=state.players_in_hand.copy(),
            current_player=state.current_player,
            board_cards=state.board_cards.copy(),
            betting_round=state.betting_round,
            last_bet=state.last_bet,
            player_bets=state.player_bets.copy(),
            player_chips=state.player_chips.copy(),
            hole_cards=state.hole_cards,
            deck=state.deck.copy(),
            players_acted=state.players_acted.copy(),
        )

        act_type, amount = action
        cp = state.current_player
        current_bet_by_player = state.player_bets.get(cp, 0)
        available_chips = state.player_chips.get(cp, 0)

        if act_type == Action.FOLD:
            new_state.players_in_hand.discard(cp)
            new_state.players_acted.add(cp)

        elif act_type == Action.CALL:
            call_amount = max(0, state.last_bet - current_bet_by_player)
            call_amount = min(call_amount, available_chips)
            new_state.player_bets[cp] = current_bet_by_player + call_amount
            new_state.player_chips[cp] = available_chips - call_amount
            new_state.pot += call_amount
            new_state.players_acted.add(cp)

        elif act_type == Action.RAISE:
            min_raise = state.last_bet * 2 if state.last_bet > 0 else 100
            total_bet = max(amount, min_raise)
            additional = total_bet - current_bet_by_player
            additional = min(additional, available_chips)
            new_state.player_bets[cp] = current_bet_by_player + additional
            new_state.player_chips[cp] = available_chips - additional
            new_state.pot += additional
            new_state.last_bet = new_state.player_bets[cp]
            new_state.players_acted = {cp}

        if self._is_round_complete(new_state):
            new_state = self._advance_round(new_state)
        else:
            new_state.current_player = self._next_active_player(new_state)

        return new_state

    def _next_active_player(self, state: GameState) -> int:
        """Find next active player; uses num_players from player_chips dict"""
        if not state.players_in_hand:
            return -1

        num_players = len(state.player_chips)
        if num_players == 0:
            return -1

        pos = (state.current_player + 1) % num_players
        for _ in range(num_players):
            if pos in state.players_in_hand:
                return pos
            pos = (pos + 1) % num_players

        return -1

    def _sample_action(self, strategy: Dict) -> Tuple[Action, int]:
        """Sample action from strategy"""
        actions = list(strategy.keys())
        probs = list(strategy.values())

        if not actions or sum(probs) == 0:
            return (Action.CALL, 0)

        probs = np.array(probs, dtype=np.float64)
        probs = probs / probs.sum()

        idx = np.random.choice(len(actions), p=probs)
        return actions[idx]

    def _prune_negative_regrets(self):
        """Remove actions with very negative regret"""
        for info_set in list(self.regret_sum.keys()):
            for action in list(self.regret_sum[info_set].keys()):
                if self.regret_sum[info_set][action] < -10000:
                    del self.regret_sum[info_set][action]


class DepthLimitedSearch:
    """Real-time search with depth limits"""

    def __init__(self, blueprint: LinearMCCFR, abstraction: AbstractionEngine):
        self.blueprint = blueprint
        self.abstraction = abstraction
        self.continuation_strategies = self._generate_continuation_strategies()

    def search(self, state: GameState, player: int,
               time_limit: float = 20.0) -> Tuple[Action, int]:
        """Perform real-time search to find best action"""
        subgame_solver = LinearMCCFR(self.abstraction)
        # Seed the subgame solver with the blueprint's accumulated knowledge
        subgame_solver.regret_sum = self.blueprint.regret_sum
        subgame_solver.strategy_sum = self.blueprint.strategy_sum
        subgame_solver.iteration = self.blueprint.iteration

        num_players = len(state.players_in_hand)
        start = time.time()
        while time.time() - start < time_limit:
            subgame_solver.train(100, num_players)

        info_set = self._get_info_set(state, player)
        strategy = subgame_solver.get_average_strategy(info_set)
        return self._sample_action(strategy)

    def _build_subgame(self, state: GameState, player: int) -> GameState:
        """Return the current state as the subgame root"""
        return state

    def _generate_continuation_strategies(self) -> List:
        """Generate k=4 continuation strategies biased toward different actions"""
        return [
            self.blueprint,
            self._bias_strategy(self.blueprint, "fold"),
            self._bias_strategy(self.blueprint, "call"),
            self._bias_strategy(self.blueprint, "raise"),
        ]

    def _bias_strategy(self, base: LinearMCCFR, bias: str) -> LinearMCCFR:
        """Create a copy of the strategy with one action type upweighted"""
        import copy
        biased = LinearMCCFR(self.abstraction)
        biased.strategy_sum = copy.deepcopy(base.strategy_sum)
        biased.regret_sum = copy.deepcopy(base.regret_sum)
        biased.iteration = base.iteration

        bias_factor = 3.0
        bias_action_map = {
            "fold": Action.FOLD,
            "call": Action.CALL,
            "raise": Action.RAISE,
        }
        target = bias_action_map.get(bias)

        for action_probs in biased.strategy_sum.values():
            for action in list(action_probs.keys()):
                act_type = action[0] if isinstance(action, tuple) else action
                if act_type == target:
                    action_probs[action] *= bias_factor

        return biased

    def _get_info_set(self, state: GameState, player: int) -> InfoSet:
        """Get information set using actual hole cards"""
        raw = state.hole_cards.get(player, [0, 0])
        hole_cards = tuple(sorted(raw))
        return InfoSet(
            hole_cards=hole_cards,
            board_cards=tuple(state.board_cards),
            betting_history=self._encode_betting_history(state),
        )

    def _encode_betting_history(self, state: GameState) -> str:
        bets = tuple(sorted(state.player_bets.items()))
        return f"R{state.betting_round}|L{state.last_bet}|B{bets}"

    def _sample_action(self, strategy: Dict) -> Tuple[Action, int]:
        if not strategy:
            return (Action.CALL, 0)

        actions = list(strategy.keys())
        probs = list(strategy.values())

        if sum(probs) == 0:
            return (Action.CALL, 0)

        probs = np.array(probs, dtype=np.float64)
        probs = probs / probs.sum()

        idx = np.random.choice(len(actions), p=probs)
        return actions[idx]


class PluribusBot:
    """Main poker bot combining blueprint and real-time search"""

    def __init__(self):
        self.abstraction = AbstractionEngine()
        self.blueprint = LinearMCCFR(self.abstraction)
        self.search_engine = DepthLimitedSearch(self.blueprint, self.abstraction)
        self.hand_history: List = []

    def train_blueprint(self, iterations: int = 1000000):
        """Train offline blueprint strategy"""
        print(f"Training blueprint strategy with {iterations} iterations...")
        self.blueprint.train(iterations)
        print("Blueprint training complete")

    def get_action(self, state: GameState, player: int,
                   use_search: bool = True) -> Tuple[Action, int]:
        """Get action for current game state"""
        if state.betting_round == 0 or state.pot < 100:
            info_set = self._get_info_set(state, player)
            strategy = self.blueprint.get_average_strategy(info_set)
            if strategy:
                return self._sample_from_strategy(strategy)
            return (Action.CALL, state.last_bet)

        if use_search:
            return self.search_engine.search(state, player)

        info_set = self._get_info_set(state, player)
        strategy = self.blueprint.get_average_strategy(info_set)
        return self._sample_from_strategy(strategy)

    def _get_info_set(self, state: GameState, player: int) -> InfoSet:
        """Get information set using actual hole cards"""
        raw = state.hole_cards.get(player, [0, 0])
        hole_cards = tuple(sorted(raw))
        return InfoSet(
            hole_cards=hole_cards,
            board_cards=tuple(state.board_cards),
            betting_history=self._encode_betting_history(state),
        )

    def _encode_betting_history(self, state: GameState) -> str:
        bets = tuple(sorted(state.player_bets.items()))
        return f"R{state.betting_round}|L{state.last_bet}|B{bets}"

    def _sample_from_strategy(self, strategy: Dict) -> Tuple[Action, int]:
        if not strategy:
            return (Action.CALL, 0)

        actions = list(strategy.keys())
        probs = list(strategy.values())

        if sum(probs) == 0:
            return (Action.CALL, 0)

        probs = np.array(probs, dtype=np.float64)
        probs = probs / probs.sum()

        idx = np.random.choice(len(actions), p=probs)
        return actions[idx]

    def save_blueprint(self, filename: str):
        """Save trained blueprint to file (gzip-compressed)"""
        data = {
            'regret_sum': {k: dict(v) for k, v in self.blueprint.regret_sum.items()},
            'strategy_sum': {k: dict(v) for k, v in self.blueprint.strategy_sum.items()},
            'iteration': self.blueprint.iteration,
        }
        with gzip.open(filename, 'wb') as f:
            pickle.dump(data, f)

    def load_blueprint(self, filename: str):
        """Load trained blueprint from file"""
        with gzip.open(filename, 'rb') as f:
            data = pickle.load(f)
        self.blueprint.regret_sum = defaultdict(
            lambda: defaultdict(float),
            {k: defaultdict(float, v) for k, v in data['regret_sum'].items()}
        )
        self.blueprint.strategy_sum = defaultdict(
            lambda: defaultdict(float),
            {k: defaultdict(float, v) for k, v in data['strategy_sum'].items()}
        )
        self.blueprint.iteration = data['iteration']


if __name__ == "__main__":
    bot = PluribusBot()
    bot.train_blueprint(iterations=10000)

    example_state = GameState(
        pot=150,
        players_in_hand={0, 1, 2, 3},
        current_player=0,
        board_cards=[10, 22, 35],
        betting_round=1,
        last_bet=50,
        player_bets={1: 50, 2: 50},
        player_chips={i: 10000 for i in range(6)},
        hole_cards={0: [3, 16], 1: [10, 23], 2: [35, 48], 3: [5, 18]},
    )

    action, amount = bot.get_action(example_state, player=0, use_search=False)
    print(f"Bot action: {action.name} ${amount}")
