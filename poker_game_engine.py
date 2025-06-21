import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import random
from enum import Enum

# Import from previous modules (in practice, these would be separate files)
# from poker_bot_pluribus import PluribusBot, GameState, Action
# from poker_bot_utils import HandEvaluator, card_to_string, string_to_card

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
        
        # Initialize components
        self.evaluator = HandEvaluator()
        self.bot = PluribusBot()
        
        # Game state
        self.players = []
        self.deck = []
        self.board = []
        self.pot = 0
        self.current_bet = 0
        self.phase = GamePhase.PREFLOP
        self.dealer_position = 0
        self.action_position = 0
        
        self._initialize_players()
    
    def _initialize_players(self):
        """Initialize players"""
        self.players = [
            Player(id=i, chips=self.starting_chips, is_bot=(i > 0))
            for i in range(self.num_players)
        ]
    
    def start_new_hand(self):
        """Start a new poker hand"""
        # Reset for new hand
        self.deck = list(range(52))
        random.shuffle(self.deck)
        self.board = []
        self.pot = 0
        self.current_bet = 0
        self.phase = GamePhase.PREFLOP
        
        # Reset players
        for player in self.players:
            player.hole_cards = []
            player.folded = False
            player.bet_this_round = 0
            player.total_bet = 0
        
        # Move dealer button
        self.dealer_position = (self.dealer_position + 1) % self.num_players
        
        # Post blinds
        self._post_blinds()
        
        # Deal hole cards
        self._deal_hole_cards()
        
        # Start betting
        self.action_position = self._next_active_player(self.dealer_position + 3)
    
    def _post_blinds(self):
        """Post small and big blinds"""
        sb_pos = (self.dealer_position + 1) % self.num_players
        bb_pos = (self.dealer_position + 2) % self.num_players
        
        # Small blind
        self._player_bet(sb_pos, self.small_blind)
        
        # Big blind
        self._player_bet(bb_pos, self.big_blind)
        
        self.current_bet = self.big_blind
    
    def _deal_hole_cards(self):
        """Deal two cards to each player"""
        for player in self.players:
            if player.chips > 0:
                player.hole_cards = [self.deck.pop(), self.deck.pop()]
    
    def _player_bet(self, player_id: int, amount: int):
        """Handle player betting"""
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
            
        elif action == Action.CALL:
            call_amount = self.current_bet - player.bet_this_round
            self._player_bet(player_id, call_amount)
            
        elif action == Action.RAISE:
            # Ensure valid raise
            min_raise = self.current_bet * 2 - player.bet_this_round
            raise_amount = max(amount - player.bet_this_round, min_raise)
            self._player_bet(player_id, raise_amount)
        
        # Check if betting round is complete
        if self._is_betting_complete():
            self._advance_phase()
        else:
            # Move to next player
            self.action_position = self._next_active_player(self.action_position + 1)
    
    def _is_betting_complete(self) -> bool:
        """Check if current betting round is complete"""
        active_players = [p for p in self.players if not p.folded and p.chips > 0]
        
        if len(active_players) <= 1:
            return True
        
        # All active players have acted and matched the current bet
        for player in active_players:
            if player.bet_this_round < self.current_bet and player.chips > 0:
                return False
        
        return True
    
    def _advance_phase(self):
        """Move to next game phase"""
        # Reset betting for new round
        for player in self.players:
            player.bet_this_round = 0
        self.current_bet = 0
        
        if self.phase == GamePhase.PREFLOP:
            # Deal flop
            self.board.extend([self.deck.pop() for _ in range(3)])
            self.phase = GamePhase.FLOP
            
        elif self.phase == GamePhase.FLOP:
            # Deal turn
            self.board.append(self.deck.pop())
            self.phase = GamePhase.TURN
            
        elif self.phase == GamePhase.TURN:
            # Deal river
            self.board.append(self.deck.pop())
            self.phase = GamePhase.RIVER
            
        elif self.phase == GamePhase.RIVER:
            # Showdown
            self.phase = GamePhase.SHOWDOWN
            self._resolve_hand()
            return
        
        # Set action to first active player after dealer
        self.action_position = self._next_active_player(self.dealer_position + 1)
    
    def _next_active_player(self, start_pos: int) -> int:
        """Find next active player position"""
        pos = start_pos % self.num_players
        
        for _ in range(self.num_players):
            player = self.players[pos]
            if not player.folded and player.chips > 0:
                return pos
            pos = (pos + 1) % self.num_players
        
        return -1  # No active players
    
    def _resolve_hand(self):
        """Determine winner(s) and distribute pot"""
        active_players = [p for p in self.players if not p.folded]
        
        if len(active_players) == 1:
            # Only one player left, they win
            winner = active_players[0]
            winner.chips += self.pot
            print(f"Player {winner.id} wins {self.pot} chips (all folded)")
            return
        
        # Evaluate hands
        player_strengths = []
        for player in active_players:
            strength = self.evaluator.evaluate_hand(player.hole_cards, self.board)
            player_strengths.append((player.id, strength))
        
        # Sort by strength (descending)
        player_strengths.sort(key=lambda x: x[1], reverse=True)
        
        # Find winner(s) - handle ties
        winning_strength = player_strengths[0][1]
        winners = [p_id for p_id, strength in player_strengths 
                  if strength == winning_strength]
        
        # Split pot among winners
        winnings_per_player = self.pot // len(winners)
        for winner_id in winners:
            self.players[winner_id].chips += winnings_per_player
        
        # Show results
        print(f"\nShowdown:")
        print(f"Board: {[card_to_string(c) for c in self.board]}")
        for player in active_players:
            cards = [card_to_string(c) for c in player.hole_cards]
            print(f"Player {player.id}: {cards}")
        print(f"Winner(s): {winners} split {self.pot} chips")
    
    def get_game_state(self) -> GameState:
        """Convert to GameState for bot"""
        players_in_hand = {i for i, p in enumerate(self.players) 
                          if not p.folded and p.chips > 0}
        
        player_bets = {i: p.bet_this_round for i, p in enumerate(self.players)}
        player_chips = {i: p.chips for i, p in enumerate(self.players)}
        
        return GameState(
            pot=self.pot,
            players_in_hand=players_in_hand,
            current_player=self.action_position,
            board_cards=self.board.copy(),
            betting_round=self.phase.value,
            last_bet=self.current_bet,
            player_bets=player_bets,
            player_chips=player_chips
        )
    
    def get_bot_action(self, player_id: int) -> Tuple[Action, int]:
        """Get action from bot for given player"""
        state = self.get_game_state()
        
        # Inject hole cards into bot's knowledge (in practice, bot tracks this)
        # This is simplified - real implementation maintains hand ranges
        
        use_search = self.phase.value > 0 or self.pot > 200
        return self.bot.get_action(state, player_id, use_search=use_search)
    
    def play_hand(self, verbose: bool = True):
        """Play a complete hand"""
        self.start_new_hand()
        
        if verbose:
            print(f"\n=== New Hand - Dealer: Player {self.dealer_position} ===")
        
        while self.phase != GamePhase.SHOWDOWN:
            # Check if hand should end early
            active_players = [p for p in self.players if not p.folded]
            if len(active_players) <= 1:
                self._resolve_hand()
                break
            
            current_player = self.players[self.action_position]
            
            if current_player.is_bot:
                # Bot action
                action, amount = self.get_bot_action(self.action_position)
                if verbose:
                    print(f"Player {self.action_position} (Bot): {action.name} {amount}")
                self.process_action(self.action_position, action, amount)
            else:
                # Human action (simplified for demo)
                action = self._get_human_action(self.action_position)
                self.process_action(self.action_position, action[0], action[1])
    
    def _get_human_action(self, player_id: int) -> Tuple[Action, int]:
        """Get action from human player (simplified UI)"""
        player = self.players[player_id]
        
        print(f"\nYour turn (Player {player_id})")
        print(f"Hole cards: {[card_to_string(c) for c in player.hole_cards]}")
        print(f"Board: {[card_to_string(c) for c in self.board]}")
        print(f"Pot: {self.pot}, Current bet: {self.current_bet}")
        print(f"Your chips: {player.chips}, Bet this round: {player.bet_this_round}")
        
        # Simple text interface
        if self.current_bet > player.bet_this_round:
            print("Actions: (f)old, (c)all, (r)aise")
            choice = input("Choice: ").lower()
            
            if choice == 'f':
                return (Action.FOLD, 0)
            elif choice == 'c':
                return (Action.CALL, self.current_bet)
            elif choice == 'r':
                amount = int(input("Raise to: "))
                return (Action.RAISE, amount)
        else:
            print("Actions: (c)heck, (b)et")
            choice = input("Choice: ").lower()
            
            if choice == 'c':
                return (Action.CALL, 0)
            elif choice == 'b':
                amount = int(input("Bet amount: "))
                return (Action.RAISE, amount)
        
        return (Action.CALL, 0)  # Default

class TrainingHarness:
    """Harness for training the bot through self-play"""
    
    def __init__(self, bot: PluribusBot):
        self.bot = bot
        self.game = PokerGameEngine(num_players=6)
        
        # Make all players bots for training
        for player in self.game.players:
            player.is_bot = True
    
    def train(self, num_hands: int = 10000):
        """Train bot through self-play"""
        print(f"Training bot with {num_hands} hands of self-play...")
        
        # First, train blueprint if not already trained
        if self.bot.blueprint.iteration == 0:
            print("Training blueprint strategy...")
            self.bot.train_blueprint(iterations=100000)
        
        # Play hands for experience
        for hand_num in range(num_hands):
            if hand_num % 1000 == 0:
                print(f"Playing hand {hand_num}/{num_hands}")
            
            self.game.play_hand(verbose=False)
            
            # Periodically save progress
            if hand_num % 5000 == 0:
                self.bot.save_blueprint(f"pluribus_checkpoint_{hand_num}.pkl")
        
        print("Training complete!")

# Example usage
if __name__ == "__main__":
    # Initialize game with 1 human and 5 bots
    game = PokerGameEngine(num_players=6)
    
    # Train the bot first (optional, can load pre-trained)
    # harness = TrainingHarness(game.bot)
    # harness.train(num_hands=1000)
    
    # Play interactive games
    print("Starting poker game with Pluribus bot...")
    print("You are Player 0, all others are Pluribus bots")
    
    while True:
        game.play_hand(verbose=True)
        
        # Check if human player is broke
        if game.players[0].chips <= 0:
            print("\nYou're out of chips! Game over.")
            break
        
        # Ask if want to continue
        if input("\nPlay another hand? (y/n): ").lower() != 'y':
            break
    
    # Show final chip counts
    print("\nFinal chip counts:")
    for player in game.players:
        print(f"Player {player.id}: {player.chips} chips")
