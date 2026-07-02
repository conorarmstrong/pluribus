import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
import random
from enum import Enum

from poker_bot_pluribus import PluribusBot, GameState, Action
from poker_bot_utils import HandEvaluator, card_to_string, string_to_card


class GamePhase(Enum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    SHOWDOWN = 4


@dataclass
class Player:
    """Represents a player in the game"""
    id: int
    chips: int
    hole_cards: List[int] = field(default_factory=list)
    folded: bool = False
    bet_this_round: int = 0
    total_bet: int = 0
    is_bot: bool = False


class PokerGameEngine:
    """Complete Texas Hold'em game engine"""

    def __init__(self, num_players: int = 6, starting_chips: int = 10000,
                 big_blind: int = 100, small_blind: int = 50):
        self.num_players = num_players
        self.starting_chips = starting_chips
        self.big_blind = big_blind
        self.small_blind = small_blind

        self.evaluator = HandEvaluator()
        self.bot = PluribusBot()

        self.players: List[Player] = []
        self.deck: List[int] = []
        self.board: List[int] = []
        self.pot = 0
        self.current_bet = 0
        self.phase = GamePhase.PREFLOP
        self.dealer_position = 0
        self.action_position = 0

        # Betting round tracking
        self.players_who_acted: Set[int] = set()
        self.last_raise_size: int = big_blind

        self._initialize_players()

    def _initialize_players(self):
        self.players = [
            Player(id=i, chips=self.starting_chips, is_bot=(i > 0))
            for i in range(self.num_players)
        ]

    def start_new_hand(self):
        """Start a new poker hand"""
        self.deck = list(range(52))
        random.shuffle(self.deck)
        self.board = []
        self.pot = 0
        self.current_bet = 0
        self.phase = GamePhase.PREFLOP
        self.players_who_acted = set()
        self.last_raise_size = self.big_blind

        for player in self.players:
            player.hole_cards = []
            player.folded = False
            player.bet_this_round = 0
            player.total_bet = 0

        self.dealer_position = (self.dealer_position + 1) % self.num_players
        self._post_blinds()
        self._deal_hole_cards()

        # Preflop: first voluntary actor is UTG (3 seats after dealer)
        self.action_position = self._next_betting_player(
            (self.dealer_position + 3) % self.num_players
        )

    def _post_blinds(self):
        """Post small and big blinds"""
        sb_pos = (self.dealer_position + 1) % self.num_players
        bb_pos = (self.dealer_position + 2) % self.num_players
        self._player_bet(sb_pos, self.small_blind)
        self._player_bet(bb_pos, self.big_blind)
        self.current_bet = self.big_blind
        self.last_raise_size = self.big_blind

    def _deal_hole_cards(self):
        """Deal two cards to each active player"""
        for player in self.players:
            if player.chips > 0 or player.total_bet > 0:
                player.hole_cards = [self.deck.pop(), self.deck.pop()]

    def _player_bet(self, player_id: int, amount: int):
        """Move chips from player stack to pot"""
        player = self.players[player_id]
        bet_amount = min(amount, player.chips)
        player.chips -= bet_amount
        player.bet_this_round += bet_amount
        player.total_bet += bet_amount
        self.pot += bet_amount
        if player.bet_this_round > self.current_bet:
            self.current_bet = player.bet_this_round

    def process_action(self, player_id: int, action: Action, amount: int = 0):
        """Process a player action"""
        player = self.players[player_id]

        if action == Action.FOLD:
            player.folded = True
            self.players_who_acted.add(player_id)

        elif action == Action.CALL:
            call_amount = self.current_bet - player.bet_this_round
            call_amount = min(call_amount, player.chips)
            self._player_bet(player_id, call_amount)
            self.players_who_acted.add(player_id)

        elif action == Action.RAISE:
            # Enforce minimum raise: must be at least current_bet + last_raise_size
            min_raise_to = self.current_bet + self.last_raise_size
            total_bet = max(amount, min_raise_to)
            total_bet = min(total_bet, player.chips + player.bet_this_round)

            raise_size = total_bet - self.current_bet
            self.last_raise_size = max(self.last_raise_size, raise_size)

            additional = total_bet - player.bet_this_round
            self._player_bet(player_id, additional)

            # Everyone else must act again after a raise
            self.players_who_acted = {player_id}

        if self._is_betting_complete():
            self._advance_phase()
        else:
            self.action_position = self._next_betting_player(
                (self.action_position + 1) % self.num_players
            )

    def _is_betting_complete(self) -> bool:
        """Check if all players have acted and bets are square"""
        active = [p for p in self.players if not p.folded]
        betting = [p for p in active if p.chips > 0]

        if len(active) <= 1:
            return True

        for player in betting:
            if player.id not in self.players_who_acted:
                return False
            if player.bet_this_round < self.current_bet:
                return False

        return True

    def _advance_phase(self):
        """Move to next game phase and deal community cards"""
        for player in self.players:
            player.bet_this_round = 0
        self.current_bet = 0
        self.last_raise_size = self.big_blind
        self.players_who_acted = set()

        if self.phase == GamePhase.PREFLOP:
            self.board.extend([self.deck.pop() for _ in range(3)])
            self.phase = GamePhase.FLOP
        elif self.phase == GamePhase.FLOP:
            self.board.append(self.deck.pop())
            self.phase = GamePhase.TURN
        elif self.phase == GamePhase.TURN:
            self.board.append(self.deck.pop())
            self.phase = GamePhase.RIVER
        elif self.phase == GamePhase.RIVER:
            self.phase = GamePhase.SHOWDOWN
            self._resolve_hand()
            return

        # Post-flop action starts from first active player left of dealer
        self.action_position = self._next_betting_player(
            (self.dealer_position + 1) % self.num_players
        )

    def _next_betting_player(self, start_pos: int) -> int:
        """Find next player who can still bet (not folded, has chips)"""
        pos = start_pos % self.num_players
        for _ in range(self.num_players):
            player = self.players[pos]
            if not player.folded and player.chips > 0:
                return pos
            pos = (pos + 1) % self.num_players
        return -1

    def _next_active_player(self, start_pos: int) -> int:
        """Find next player still in the hand (not folded; includes all-in)"""
        pos = start_pos % self.num_players
        for _ in range(self.num_players):
            if not self.players[pos].folded:
                return pos
            pos = (pos + 1) % self.num_players
        return -1

    def _resolve_hand(self):
        """Determine winner(s) and distribute pot"""
        active_players = [p for p in self.players if not p.folded]

        if len(active_players) == 1:
            winner = active_players[0]
            winner.chips += self.pot
            print(f"Player {winner.id} wins {self.pot} chips (all folded)")
            return

        player_strengths = []
        for player in active_players:
            strength = self.evaluator.evaluate_hand(player.hole_cards, self.board)
            player_strengths.append((player.id, strength))

        player_strengths.sort(key=lambda x: x[1], reverse=True)
        winning_strength = player_strengths[0][1]
        winner_ids = [pid for pid, s in player_strengths if s == winning_strength]

        winnings_per_player = self.pot // len(winner_ids)
        remainder = self.pot % len(winner_ids)

        for winner_id in winner_ids:
            self.players[winner_id].chips += winnings_per_player

        # Award remainder chips to first winner clockwise from dealer
        if remainder > 0:
            for i in range(self.num_players):
                pos = (self.dealer_position + 1 + i) % self.num_players
                if pos in winner_ids:
                    self.players[pos].chips += remainder
                    break

        print(f"\nShowdown:")
        print(f"Board: {[card_to_string(c) for c in self.board]}")
        for player in active_players:
            cards = [card_to_string(c) for c in player.hole_cards]
            print(f"Player {player.id}: {cards}")
        print(f"Winner(s): {winner_ids} split {self.pot} chips")

    def get_game_state(self) -> GameState:
        """Convert engine state to GameState for the bot, including hole cards"""
        players_in_hand = {
            i for i, p in enumerate(self.players)
            if not p.folded
        }
        player_bets = {i: p.bet_this_round for i, p in enumerate(self.players)}
        player_chips = {i: p.chips for i, p in enumerate(self.players)}
        hole_cards = {i: list(p.hole_cards) for i, p in enumerate(self.players)}

        return GameState(
            pot=self.pot,
            players_in_hand=players_in_hand,
            current_player=self.action_position,
            board_cards=self.board.copy(),
            betting_round=self.phase.value,
            last_bet=self.current_bet,
            player_bets=player_bets,
            player_chips=player_chips,
            hole_cards=hole_cards,
            deck=self.deck.copy(),
            players_acted=self.players_who_acted.copy(),
        )

    def get_bot_action(self, player_id: int) -> Tuple[Action, int]:
        """Get action from bot for given player"""
        state = self.get_game_state()
        use_search = self.phase.value > 0 or self.pot > 200
        return self.bot.get_action(state, player_id, use_search=use_search)

    def play_hand(self, verbose: bool = True):
        """Play a complete hand"""
        self.start_new_hand()

        if verbose:
            print(f"\n=== New Hand - Dealer: Player {self.dealer_position} ===")

        while self.phase != GamePhase.SHOWDOWN:
            active_players = [p for p in self.players if not p.folded]
            if len(active_players) <= 1:
                self._resolve_hand()
                break

            if self.action_position == -1:
                self._advance_phase()
                continue

            current_player = self.players[self.action_position]

            if current_player.is_bot:
                action, amount = self.get_bot_action(self.action_position)
                if verbose:
                    print(f"Player {self.action_position} (Bot): {action.name} {amount}")
                self.process_action(self.action_position, action, amount)
            else:
                action = self._get_human_action(self.action_position)
                self.process_action(self.action_position, action[0], action[1])

    def _get_human_action(self, player_id: int) -> Tuple[Action, int]:
        """Get action from human player"""
        player = self.players[player_id]

        print(f"\nYour turn (Player {player_id})")
        print(f"Hole cards: {[card_to_string(c) for c in player.hole_cards]}")
        print(f"Board: {[card_to_string(c) for c in self.board]}")
        print(f"Pot: {self.pot}, Current bet: {self.current_bet}")
        print(f"Your chips: {player.chips}, Bet this round: {player.bet_this_round}")

        if self.current_bet > player.bet_this_round:
            print("Actions: (f)old, (c)all, (r)aise")
            choice = input("Choice: ").lower()
            if choice == 'f':
                return (Action.FOLD, 0)
            elif choice == 'r':
                amount = int(input("Raise to: "))
                return (Action.RAISE, amount)
            return (Action.CALL, self.current_bet)
        else:
            print("Actions: (c)heck, (b)et")
            choice = input("Choice: ").lower()
            if choice == 'b':
                amount = int(input("Bet amount: "))
                return (Action.RAISE, amount)
            return (Action.CALL, 0)


class TrainingHarness:
    """Harness for training the bot through self-play"""

    def __init__(self, bot: PluribusBot):
        self.bot = bot
        self.game = PokerGameEngine(num_players=6)
        for player in self.game.players:
            player.is_bot = True

    def train(self, num_hands: int = 10000):
        """Train bot through self-play"""
        print(f"Training bot with {num_hands} hands of self-play...")

        if self.bot.blueprint.iteration == 0:
            print("Training blueprint strategy...")
            self.bot.train_blueprint(iterations=10000)
        else:
            print(f"Using existing blueprint ({self.bot.blueprint.iteration} iterations)")

        for hand_num in range(num_hands):
            if hand_num % 1000 == 0:
                print(f"Playing hand {hand_num}/{num_hands}")
            self.game.play_hand(verbose=False)

            if hand_num % 5000 == 0 and hand_num > 0:
                self.bot.save_blueprint(f"pluribus_checkpoint_{hand_num}.pkl.gz")

        print("Training complete!")


if __name__ == "__main__":
    game = PokerGameEngine(num_players=6)

    print("Starting poker game with Pluribus bot...")
    print("You are Player 0, all others are Pluribus bots")

    while True:
        game.play_hand(verbose=True)

        if game.players[0].chips <= 0:
            print("\nYou're out of chips! Game over.")
            break

        if input("\nPlay another hand? (y/n): ").lower() != 'y':
            break

    print("\nFinal chip counts:")
    for player in game.players:
        print(f"Player {player.id}: {player.chips} chips")
