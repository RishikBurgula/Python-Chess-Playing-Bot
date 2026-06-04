"""
============================================================
  AUTOMATIC CHESS BOT THAT PLAYS AGAINST A HUMAN PLAYER
  Python Project | Beginner Friendly | Clean Code
============================================================

ABSTRACT:
    This project implements a fully playable chess game in Python
    where a human (white) plays against an AI bot (black).
    The bot has three levels:
      Level 1 - Random legal moves
      Level 2 - Greedy best-capture moves
      Level 3 - Minimax algorithm (depth 2)

HOW TO RUN:
    python chess_bot.py
    Then follow the on-screen prompts.
    Enter moves like: e2 e4  (source square space destination square)
============================================================
"""

import random   # Used by Level 1 bot for random move selection
import copy     # Used to deep-copy the board for minimax simulations


# ==============================================================
#  SECTION 1: BOARD SETUP
# ==============================================================

def create_board():
    """
    Creates and returns the initial 8x8 chess board as a list of lists.

    Board layout (row 0 = rank 8 = black's back row):
      Row 0: Black's major pieces  (r n b q k b n r)
      Row 1: Black's pawns         (p p p p p p p p)
      Rows 2-5: Empty squares      (. . . . . . . .)
      Row 6: White's pawns         (P P P P P P P P)
      Row 7: White's major pieces  (R N B Q K B N R)

    Piece symbols:
      UPPER CASE = White pieces  (P R N B Q K)
      lower case = Black pieces  (p r n b q k)
      '.'        = Empty square
    """
    board = [
        ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],   # Row 0 - Black back rank
        ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],   # Row 1 - Black pawns
        ['.', '.', '.', '.', '.', '.', '.', '.'],    # Row 2 - Empty
        ['.', '.', '.', '.', '.', '.', '.', '.'],    # Row 3 - Empty
        ['.', '.', '.', '.', '.', '.', '.', '.'],    # Row 4 - Empty
        ['.', '.', '.', '.', '.', '.', '.', '.'],    # Row 5 - Empty
        ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],   # Row 6 - White pawns
        ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R'],   # Row 7 - White back rank
    ]
    return board


def print_board(board):
    """
    Prints the chess board in a readable format with:
    - Rank numbers (8 down to 1) on the left
    - File letters (a through h) on the bottom
    - Piece symbols in their positions

    Example output:
       a b c d e f g h
    8  r n b q k b n r
    7  p p p p p p p p
    6  . . . . . . . .
    ...
    """
    print()
    print("    a b c d e f g h")      # Column headers (files)
    print("   -----------------")
    for row_index in range(8):
        rank_number = 8 - row_index   # Rank 8 is at row index 0
        # Join all pieces in this row with a space between them
        row_display = ' '.join(board[row_index])
        print(f" {rank_number} | {row_display}")
    print("   -----------------")
    print()


# ==============================================================
#  SECTION 2: COORDINATE HELPERS
# ==============================================================

def parse_move(move_str):
    """
    Converts chess notation like 'e2' into board matrix coordinates (row, col).

    Chess notation:
      File (column): a=0, b=1, c=2, d=3, e=4, f=5, g=6, h=7
      Rank (row):    8=row0, 7=row1, 6=row2, ... 1=row7

    Example:
      'e2' -> file='e' (col=4), rank='2' (row=6) -> returns (6, 4)
      'g1' -> file='g' (col=6), rank='1' (row=7) -> returns (7, 6)

    Returns (row, col) tuple or None if input is invalid.
    """
    move_str = move_str.strip().lower()
    if len(move_str) != 2:
        return None
    file_char = move_str[0]                      # Letter part: a-h
    rank_char  = move_str[1]                     # Number part: 1-8

    if file_char not in 'abcdefgh':
        return None
    if rank_char not in '12345678':
        return None

    col = ord(file_char) - ord('a')             # 'a'->0, 'b'->1, ..., 'h'->7
    row = 8 - int(rank_char)                    # '8'->0, '7'->1, ..., '1'->7
    return (row, col)


def coord_to_notation(row, col):
    """
    Converts board matrix coordinates back to chess notation.
    Example: (6, 4) -> 'e2'
    """
    file_char = chr(ord('a') + col)
    rank_char = str(8 - row)
    return file_char + rank_char


def is_inside_board(row, col):
    """
    Returns True if the given (row, col) is within the 8x8 board.
    Valid range: row 0-7, col 0-7.
    """
    return 0 <= row <= 7 and 0 <= col <= 7


# ==============================================================
#  SECTION 3: PIECE IDENTIFICATION HELPERS
# ==============================================================

def is_white_piece(piece):
    """Returns True if piece is a White piece (uppercase letter)."""
    return piece in 'PRNBQK'


def is_black_piece(piece):
    """Returns True if piece is a Black piece (lowercase letter)."""
    return piece in 'prnbqk'


def is_own_piece(piece, player):
    """
    Returns True if 'piece' belongs to the current 'player'.
    player is either 'white' or 'black'.
    """
    if player == 'white':
        return is_white_piece(piece)
    else:
        return is_black_piece(piece)


def is_opponent_piece(piece, player):
    """
    Returns True if 'piece' belongs to the opponent of 'player'.
    """
    if player == 'white':
        return is_black_piece(piece)
    else:
        return is_white_piece(piece)


# ==============================================================
#  SECTION 4: PATH CLEAR CHECK (for Rook, Bishop, Queen)
# ==============================================================

def is_path_clear(board, start, end):
    """
    Checks whether all squares BETWEEN start and end are empty ('.').
    Used by Rook, Bishop, and Queen move validators.

    The function steps from start toward end one square at a time,
    checking each intermediate square. It does NOT check start or end.

    start, end: (row, col) tuples
    Returns True if the path is clear, False if any piece is in the way.
    """
    r1, c1 = start
    r2, c2 = end

    # Calculate direction: -1, 0, or +1 for each axis
    row_step = 0
    col_step = 0

    if r2 > r1:
        row_step = 1    # Moving down (toward row 7)
    elif r2 < r1:
        row_step = -1   # Moving up (toward row 0)

    if c2 > c1:
        col_step = 1    # Moving right
    elif c2 < c1:
        col_step = -1   # Moving left

    # Start one step in from start, stop one step before end
    current_row = r1 + row_step
    current_col = c1 + col_step

    while (current_row, current_col) != (r2, c2):
        if board[current_row][current_col] != '.':
            return False    # There is a piece blocking the path
        current_row += row_step
        current_col += col_step

    return True   # Path is clear


# ==============================================================
#  SECTION 5: INDIVIDUAL PIECE MOVE VALIDATORS
# ==============================================================

def is_valid_pawn_move(board, start, end, player):
    """
    Validates pawn movement rules:

    WHITE PAWN (moves UP the board, decreasing row index):
      - Normal move: 1 step forward (row - 1), same column, destination must be empty.
      - First move:  2 steps forward if pawn is on row 6 and path is clear.
      - Capture:     1 step diagonally forward to capture an opponent piece.

    BLACK PAWN (moves DOWN the board, increasing row index):
      - Normal move: 1 step forward (row + 1), same column, destination must be empty.
      - First move:  2 steps forward if pawn is on row 1 and path is clear.
      - Capture:     1 step diagonally forward to capture an opponent piece.

    Returns True if the pawn move is valid.
    """
    r1, c1 = start
    r2, c2 = end
    piece = board[r1][c1]
    destination = board[r2][c2]

    if player == 'white':
        direction = -1          # White pawns move toward row 0 (up)
        start_row = 6           # White pawns start at row 6

        # Normal 1-step forward move: destination must be empty
        if c2 == c1 and r2 == r1 + direction and destination == '.':
            return True

        # First move: 2 steps forward from starting row
        if c2 == c1 and r1 == start_row and r2 == r1 + 2 * direction:
            if destination == '.' and board[r1 + direction][c1] == '.':
                return True

        # Diagonal capture: 1 step diagonally, must have opponent piece
        if abs(c2 - c1) == 1 and r2 == r1 + direction:
            if is_opponent_piece(destination, player):
                return True

    else:  # player == 'black'
        direction = 1           # Black pawns move toward row 7 (down)
        start_row = 1           # Black pawns start at row 1

        # Normal 1-step forward move
        if c2 == c1 and r2 == r1 + direction and destination == '.':
            return True

        # First move: 2 steps forward from starting row
        if c2 == c1 and r1 == start_row and r2 == r1 + 2 * direction:
            if destination == '.' and board[r1 + direction][c1] == '.':
                return True

        # Diagonal capture
        if abs(c2 - c1) == 1 and r2 == r1 + direction:
            if is_opponent_piece(destination, player):
                return True

    return False   # None of the above conditions matched


def is_valid_rook_move(board, start, end, player):
    """
    Validates rook movement:
    - Rook moves any number of squares horizontally OR vertically.
    - Cannot jump over pieces (path must be clear).
    - Cannot capture own piece.

    Returns True if the rook move is valid.
    """
    r1, c1 = start
    r2, c2 = end

    # Rook must move in a straight line (same row OR same column)
    if r1 != r2 and c1 != c2:
        return False    # Diagonal move - not allowed for rook

    # Cannot capture own piece
    if is_own_piece(board[r2][c2], player):
        return False

    # Path between start and end must be clear
    return is_path_clear(board, start, end)


def is_valid_knight_move(board, start, end, player):
    """
    Validates knight movement:
    - Knight moves in an L-shape: 2 squares in one direction, 1 in the other.
    - Knights CAN jump over other pieces (no path check needed).
    - Cannot capture own piece.

    Valid knight offsets (row_diff, col_diff):
      (±1, ±2) or (±2, ±1)

    Returns True if the knight move is valid.
    """
    r1, c1 = start
    r2, c2 = end
    row_diff = abs(r2 - r1)
    col_diff = abs(c2 - c1)

    # Must be an L-shape move
    if not ((row_diff == 2 and col_diff == 1) or (row_diff == 1 and col_diff == 2)):
        return False

    # Cannot capture own piece
    if is_own_piece(board[r2][c2], player):
        return False

    return True


def is_valid_bishop_move(board, start, end, player):
    """
    Validates bishop movement:
    - Bishop moves diagonally any number of squares.
    - Cannot jump over pieces (path must be clear).
    - Cannot capture own piece.

    Returns True if the bishop move is valid.
    """
    r1, c1 = start
    r2, c2 = end
    row_diff = abs(r2 - r1)
    col_diff = abs(c2 - c1)

    # Bishop must move diagonally (equal row and col distance)
    if row_diff != col_diff or row_diff == 0:
        return False

    # Cannot capture own piece
    if is_own_piece(board[r2][c2], player):
        return False

    # Path must be clear
    return is_path_clear(board, start, end)


def is_valid_queen_move(board, start, end, player):
    """
    Validates queen movement:
    - Queen combines rook + bishop movement.
    - Can move any number of squares in any direction (horizontal, vertical, diagonal).
    - Cannot jump over pieces.
    - Cannot capture own piece.

    Returns True if the queen move is valid.
    """
    # Queen is valid if EITHER rook OR bishop rules apply
    return (is_valid_rook_move(board, start, end, player) or
            is_valid_bishop_move(board, start, end, player))


def is_valid_king_move(board, start, end, player):
    """
    Validates king movement:
    - King moves exactly 1 square in any direction (8 possible squares).
    - Cannot move to a square occupied by own piece.

    Returns True if the king move is valid.
    """
    r1, c1 = start
    r2, c2 = end
    row_diff = abs(r2 - r1)
    col_diff = abs(c2 - c1)

    # King moves only 1 square in any direction
    if row_diff > 1 or col_diff > 1:
        return False

    # King must actually move (not stay in place)
    if row_diff == 0 and col_diff == 0:
        return False

    # Cannot capture own piece
    if is_own_piece(board[r2][c2], player):
        return False

    return True


# ==============================================================
#  SECTION 6: MASTER MOVE VALIDATOR
# ==============================================================

def is_valid_move(board, start, end, player):
    """
    Master move validation function.
    Checks ALL conditions before allowing a move:

    Step 1: Is source square inside the board?
    Step 2: Is destination square inside the board?
    Step 3: Does source square have a piece?
    Step 4: Does the piece belong to the current player?
    Step 5: Is the destination square different from source?
    Step 6: Does the piece-specific rule allow this move?

    Returns True only if ALL checks pass.
    """
    r1, c1 = start
    r2, c2 = end

    # Step 1 & 2: Both squares must be inside the board
    if not is_inside_board(r1, c1) or not is_inside_board(r2, c2):
        return False

    piece = board[r1][c1]

    # Step 3: Source square must have a piece
    if piece == '.':
        return False

    # Step 4: Piece must belong to current player
    if not is_own_piece(piece, player):
        return False

    # Step 5: Destination must differ from source
    if start == end:
        return False

    # Step 6: Check piece-specific movement rules
    piece_type = piece.lower()   # Lowercase so we handle both colors together

    if piece_type == 'p':
        return is_valid_pawn_move(board, start, end, player)
    elif piece_type == 'r':
        return is_valid_rook_move(board, start, end, player)
    elif piece_type == 'n':
        return is_valid_knight_move(board, start, end, player)
    elif piece_type == 'b':
        return is_valid_bishop_move(board, start, end, player)
    elif piece_type == 'q':
        return is_valid_queen_move(board, start, end, player)
    elif piece_type == 'k':
        return is_valid_king_move(board, start, end, player)

    return False   # Unknown piece type


# ==============================================================
#  SECTION 7: MAKE A MOVE
# ==============================================================

def make_move(board, start, end):
    """
    Executes a move on the board.
    Moves the piece from 'start' to 'end', replacing whatever was there.
    Leaves the source square empty ('.').

    Does NOT validate the move - validation should be done before calling this.

    Also handles basic pawn promotion:
      If a white pawn reaches row 0, it becomes a Queen ('Q').
      If a black pawn reaches row 7, it becomes a Queen ('q').

    Returns the captured piece (or '.' if nothing was captured).
    """
    r1, c1 = start
    r2, c2 = end

    captured_piece = board[r2][c2]   # Save whatever is at destination

    board[r2][c2] = board[r1][c1]    # Move piece to destination
    board[r1][c1] = '.'              # Clear source square

    # Pawn promotion: pawn reaching the opposite end becomes a queen
    if board[r2][c2] == 'P' and r2 == 0:    # White pawn reaches top
        board[r2][c2] = 'Q'
        print("  *** White pawn promoted to Queen! ***")

    if board[r2][c2] == 'p' and r2 == 7:    # Black pawn reaches bottom
        board[r2][c2] = 'q'
        print("  *** Black pawn promoted to Queen! ***")

    return captured_piece


# ==============================================================
#  SECTION 8: GENERATE ALL LEGAL MOVES
# ==============================================================

def generate_legal_moves(board, player):
    """
    Generates a list of ALL legal moves for the given player.

    Algorithm:
      1. Loop through every square on the board.
      2. For each square that has a piece belonging to 'player':
         3. Loop through every possible destination square.
         4. If is_valid_move() returns True, add (start, end) to the list.

    Returns a list of (start, end) tuples where start and end are (row, col).

    This is a brute-force approach: checks all 64x64 = 4096 combinations.
    For a beginner project, this is perfectly acceptable.
    """
    legal_moves = []

    for r1 in range(8):
        for c1 in range(8):
            # Only consider squares with current player's pieces
            if is_own_piece(board[r1][c1], player):
                for r2 in range(8):
                    for c2 in range(8):
                        start = (r1, c1)
                        end   = (r2, c2)
                        if is_valid_move(board, start, end, player):
                            legal_moves.append((start, end))

    return legal_moves


# ==============================================================
#  SECTION 9: BOARD EVALUATION FUNCTION
# ==============================================================

# Piece values used by the evaluation function
PIECE_VALUES = {
    'p': 1,    # Pawn
    'n': 3,    # Knight
    'b': 3,    # Bishop
    'r': 5,    # Rook
    'q': 9,    # Queen
    'k': 100,  # King (very high to make king capture decisive)
}

def evaluate_board(board):
    """
    Evaluates the board position and returns a numeric score.

    Scoring logic:
      - White pieces add their value to the score (positive).
      - Black pieces subtract their value from the score (negative).
      - A positive score means White is winning.
      - A negative score means Black is winning.

    This is called a 'material evaluation' - it only counts piece values,
    not positional advantages. Good enough for a beginner bot.

    Returns an integer score.
    """
    score = 0

    for row in board:
        for piece in row:
            if piece == '.':
                continue
            value = PIECE_VALUES.get(piece.lower(), 0)   # Get piece's value
            if is_white_piece(piece):
                score += value    # White pieces add to score
            else:
                score -= value    # Black pieces subtract from score

    return score


# ==============================================================
#  SECTION 10: BOT LEVEL 1 - RANDOM MOVE
# ==============================================================

def bot_random_move(board):
    """
    BOT LEVEL 1: Random Move Bot.

    Strategy:
      1. Generate all legal moves for black.
      2. Pick one at random.

    This is the simplest possible bot. It plays legally but has no strategy.
    Good for demonstrating that move generation works.

    Returns a (start, end) tuple or None if no moves are available.
    """
    legal_moves = generate_legal_moves(board, 'black')

    if not legal_moves:
        return None    # No moves available - game over

    chosen_move = random.choice(legal_moves)   # Pick a random move
    return chosen_move


# ==============================================================
#  SECTION 11: BOT LEVEL 2 - BEST CAPTURE MOVE
# ==============================================================

def bot_best_capture_move(board):
    """
    BOT LEVEL 2: Greedy Best-Capture Bot.

    Strategy:
      1. Generate all legal moves for black.
      2. For each move, check what piece would be captured.
      3. Score each move based on the captured piece's value.
      4. If captures are available, pick the one with highest value.
      5. If no captures available, fall back to a random move.

    This is a 'greedy' strategy - it always takes the best immediate gain
    without thinking about future consequences.

    Returns a (start, end) tuple or None if no moves available.
    """
    legal_moves = generate_legal_moves(board, 'black')

    if not legal_moves:
        return None

    best_move  = None
    best_score = -1    # We only prefer captures (score > 0)

    for (start, end) in legal_moves:
        r2, c2 = end
        target_piece = board[r2][c2]   # Piece at destination

        # Score this move: value of captured piece (0 if empty square)
        if is_opponent_piece(target_piece, 'black'):
            capture_value = PIECE_VALUES.get(target_piece.lower(), 0)
        else:
            capture_value = 0

        # Update best if this capture is more valuable
        if capture_value > best_score:
            best_score = capture_value
            best_move  = (start, end)

    # If we found a capture move, use it; otherwise pick random
    if best_score > 0:
        return best_move
    else:
        return random.choice(legal_moves)


# ==============================================================
#  SECTION 12: BOT LEVEL 3 - MINIMAX ALGORITHM
# ==============================================================

def minimax(board, depth, maximizing_player):
    """
    MINIMAX ALGORITHM - Simple English Explanation:
    ================================================
    Imagine you are playing a game and you think 2 moves ahead:

    YOUR TURN (Maximizer - here it's actually used for Black which minimizes):
      You look at all your possible moves.
      For each move, you imagine 'what would my opponent do?'

    OPPONENT'S TURN (Minimizer):
      They also look at all THEIR possible moves.
      They pick the move that is WORST for you (best for them).

    You then pick the move that gives you the best result AFTER
    the opponent plays their best response.

    In chess terms:
      - Black wants to MINIMIZE the board score (more negative = better for black)
      - White wants to MAXIMIZE the board score (more positive = better for white)

    Parameters:
      board             : Current board state (2D list)
      depth             : How many moves to look ahead (we use depth 2)
      maximizing_player : True if it's White's turn (maximizer), False for Black

    Returns:
      The best evaluation score from this position.
    """

    # BASE CASE: If we've reached the desired depth, evaluate the board
    if depth == 0:
        return evaluate_board(board)

    if maximizing_player:
        # White's turn: try to MAXIMIZE the score
        max_eval = float('-inf')    # Start with worst possible score for maximizer
        moves = generate_legal_moves(board, 'white')

        if not moves:   # No moves available
            return evaluate_board(board)

        for (start, end) in moves:
            # Create a copy of the board so we don't modify the real one
            board_copy = copy.deepcopy(board)
            make_move(board_copy, start, end)

            # Recursively call minimax for Black's response (depth-1)
            eval_score = minimax(board_copy, depth - 1, False)

            # Keep the maximum score
            max_eval = max(max_eval, eval_score)

        return max_eval

    else:
        # Black's turn: try to MINIMIZE the score
        min_eval = float('inf')     # Start with worst possible score for minimizer
        moves = generate_legal_moves(board, 'black')

        if not moves:   # No moves available
            return evaluate_board(board)

        for (start, end) in moves:
            board_copy = copy.deepcopy(board)
            make_move(board_copy, start, end)

            # Recursively call minimax for White's response (depth-1)
            eval_score = minimax(board_copy, depth - 1, True)

            # Keep the minimum score
            min_eval = min(min_eval, eval_score)

        return min_eval


def bot_minimax_move(board, depth=2):
    """
    BOT LEVEL 3: Minimax Bot.

    Uses the minimax algorithm to pick the move that leads to the
    best board position after 'depth' moves.

    Algorithm:
      1. Generate all legal moves for black.
      2. For each move, simulate it on a copy of the board.
      3. Call minimax() to evaluate resulting position.
      4. Pick the move with the LOWEST score (best for black).

    depth=2 means Black looks 2 half-moves ahead:
      - Black's move (depth 2 -> depth 1)
      - White's best response (depth 1 -> depth 0)

    Returns a (start, end) tuple or None if no moves available.
    """
    legal_moves = generate_legal_moves(board, 'black')

    if not legal_moves:
        return None

    best_move  = None
    best_score = float('inf')    # Black wants to MINIMIZE score

    for (start, end) in legal_moves:
        board_copy = copy.deepcopy(board)
        make_move(board_copy, start, end)

        # Evaluate this move using minimax (now it's White's turn: maximizing=True)
        score = minimax(board_copy, depth - 1, True)

        # Black picks the move with the lowest score
        if score < best_score:
            best_score = score
            best_move  = (start, end)

    return best_move


# ==============================================================
#  SECTION 13: KING ALIVE CHECK
# ==============================================================

def king_is_alive(board, player):
    """
    Checks if the king of the given player is still on the board.

    In our simplified version, the game ends when a king is captured.
    (Full chess would detect check/checkmate before this happens.)

    Returns True if the king exists on the board, False if captured.
    """
    king_piece = 'K' if player == 'white' else 'k'

    for row in board:
        if king_piece in row:
            return True

    return False   # King not found on the board


# ==============================================================
#  SECTION 14: MAIN GAME LOOP
# ==============================================================

def play_game():
    """
    Main function that runs the chess game.

    Flow:
      1. Create and display the initial board.
      2. Ask player to choose bot level.
      3. Loop: White (human) moves, then Black (bot) moves.
      4. After each move, check if a king was captured.
      5. Game ends when a king is captured or no legal moves remain.

    Human Input Format:
      Enter two squares separated by space: e2 e4
      This means: move the piece at e2 to e4.
    """
    print("=" * 55)
    print("   CHESS BOT - Human (White) vs AI Bot (Black)")
    print("=" * 55)
    print()
    print("How to play:")
    print("  Enter moves as: [source] [destination]")
    print("  Example: e2 e4  means move piece at e2 to e4")
    print("  Type 'quit' to exit the game.")
    print()

    # Ask user which bot level to use
    print("Choose Bot Level:")
    print("  1 - Random Bot (easy)")
    print("  2 - Greedy Capture Bot (medium)")
    print("  3 - Minimax Bot (harder, but slower)")
    print()

    while True:
        level_input = input("Enter bot level (1/2/3): ").strip()
        if level_input in ['1', '2', '3']:
            bot_level = int(level_input)
            break
        print("Please enter 1, 2, or 3.")

    print(f"\nBot Level {bot_level} selected. Let's play!\n")

    # Create the board
    board = create_board()
    print_board(board)

    current_player = 'white'   # White always goes first
    move_number = 1

    # ---- MAIN GAME LOOP ----
    while True:

        # Check if current player's king is still alive
        if not king_is_alive(board, current_player):
            winner = 'Black' if current_player == 'white' else 'White'
            print(f"\n{'='*40}")
            print(f"  GAME OVER! {winner} wins by capturing the King!")
            print(f"{'='*40}\n")
            break

        # Check if current player has any legal moves
        legal_moves = generate_legal_moves(board, current_player)
        if not legal_moves:
            print(f"\nNo legal moves for {current_player}. Game over!")
            break

        # ---- HUMAN'S TURN (White) ----
        if current_player == 'white':
            print(f"--- Move {move_number} | WHITE's turn ---")

            while True:    # Keep asking until valid input is given
                user_input = input("Your move (e.g. e2 e4): ").strip()

                if user_input.lower() == 'quit':
                    print("Thanks for playing! Goodbye.")
                    return   # Exit the game

                parts = user_input.split()
                if len(parts) != 2:
                    print("  Invalid format. Enter two squares like: e2 e4")
                    continue

                start = parse_move(parts[0])
                end   = parse_move(parts[1])

                if start is None or end is None:
                    print("  Invalid square names. Use letters a-h and numbers 1-8.")
                    continue

                if not is_valid_move(board, start, end, current_player):
                    print("  That is not a legal move. Try again.")
                    continue

                # Move is valid - execute it
                captured = make_move(board, start, end)
                start_notation = coord_to_notation(*start)
                end_notation   = coord_to_notation(*end)

                if captured != '.':
                    print(f"  White moves {start_notation} -> {end_notation} | Captured: {captured}")
                else:
                    print(f"  White moves {start_notation} -> {end_notation}")

                print_board(board)
                break   # Valid move made, exit inner loop

        # ---- BOT'S TURN (Black) ----
        else:
            print(f"--- Move {move_number} | BLACK Bot is thinking... ---")

            # Pick a move based on selected bot level
            if bot_level == 1:
                bot_move = bot_random_move(board)
            elif bot_level == 2:
                bot_move = bot_best_capture_move(board)
            else:
                bot_move = bot_minimax_move(board, depth=2)

            if bot_move is None:
                print("Bot has no legal moves! Game over.")
                break

            start, end = bot_move
            captured = make_move(board, start, end)
            start_notation = coord_to_notation(*start)
            end_notation   = coord_to_notation(*end)

            if captured != '.':
                print(f"  Black moves {start_notation} -> {end_notation} | Captured: {captured}")
            else:
                print(f"  Black moves {start_notation} -> {end_notation}")

            print_board(board)

            move_number += 1   # Increment after both players have moved

        # Switch player
        current_player = 'black' if current_player == 'white' else 'white'


# ==============================================================
#  SECTION 15: ENTRY POINT
# ==============================================================

if __name__ == "__main__":
    """
    This block runs only when you execute this file directly.
    It starts the chess game.
    """
    play_game()


# ==============================================================
#  DOCUMENTATION SECTION
# ==============================================================
"""
====================================================================
  PROJECT DOCUMENTATION
====================================================================

1. PROJECT ABSTRACT
--------------------
This project is a text-based chess game implemented in Python where
a human player (White) plays against an AI bot (Black). The project
demonstrates core programming concepts including 2D arrays, recursion,
and game tree search. The bot has three difficulty levels ranging from
random moves to minimax-based intelligent play.

====================================================================

2. ALGORITHM USED
------------------

A. Board Representation:
   - 8x8 Python list of lists.
   - Uppercase = White, Lowercase = Black, '.' = Empty.

B. Move Validation:
   - Each piece has its own validation function.
   - Master validator chains all checks together.

C. Move Generation:
   - Brute-force: check all 4096 (64x64) start-end pairs.
   - Filter by is_valid_move() to get legal moves only.

D. Board Evaluation:
   - Material count: sum of all piece values.
   - White positive, Black negative.

E. Minimax:
   - Game tree search to depth 2.
   - Black minimizes, White maximizes.
   - No pruning in this basic version (Alpha-Beta can be added later).

====================================================================

3. FUNCTION EXPLANATIONS
--------------------------

create_board()       : Returns the standard starting chess position as
                       an 8x8 list.

print_board()        : Displays the board with rank/file labels.

parse_move()         : Converts 'e2' to (6, 4) coordinates.

coord_to_notation()  : Converts (6, 4) back to 'e2'.

is_inside_board()    : Bounds check for coordinates.

is_white/black_piece(): Identifies piece color by case.

is_own/opponent_piece(): Determines piece ownership for current player.

is_path_clear()      : Checks no pieces block the path (for R, B, Q).

is_valid_*_move()    : Piece-specific movement rules.

is_valid_move()      : Master validator calling all sub-validators.

make_move()          : Applies a move to the board.

generate_legal_moves(): Returns all legal (start,end) pairs for a player.

evaluate_board()     : Returns a material score (+ = White winning).

bot_random_move()    : Picks a random legal move (Level 1).

bot_best_capture_move(): Picks the highest-value capture (Level 2).

minimax()            : Recursive game tree evaluator (Level 3).

bot_minimax_move()   : Uses minimax to select the best move (Level 3).

king_is_alive()      : Checks if a player's king is still on the board.

play_game()          : Main game loop.

====================================================================

4. ADVANTAGES
--------------
1. No external libraries - entirely built from scratch.
2. Clean, readable code suitable for academic presentation.
3. Three bot levels show progressive improvement.
4. Modular design: each function has a single responsibility.
5. Easy to extend with castling, en passant, etc.
6. Good foundation for learning AI/game programming.

====================================================================

5. LIMITATIONS
---------------
1. No check/checkmate detection (game ends on king capture).
2. No castling, en passant, or draw rules.
3. Minimax at depth 2 is not very strong.
4. Brute-force move generation is slow for deeper search.
5. Text-based interface (no GUI).
6. No time limits for bot moves.

====================================================================

6. FUTURE IMPROVEMENTS
------------------------
1. Add Alpha-Beta pruning to make minimax faster.
2. Implement check, checkmate, and stalemate detection.
3. Add castling and en passant rules.
4. Increase minimax depth with Alpha-Beta pruning.
5. Add positional evaluation (piece-square tables).
6. Build a GUI using pygame or tkinter.
7. Add move history and undo functionality.
8. Implement iterative deepening for better time management.

====================================================================

7. VIVA QUESTIONS AND ANSWERS
-------------------------------

Q1: What data structure is used to represent the chess board?
A1: A 2D Python list (list of lists), 8 rows × 8 columns.
    Each cell contains a character: uppercase for White,
    lowercase for Black, '.' for empty.

Q2: How do you validate if a move is legal?
A2: We use is_valid_move() which checks:
    (1) Source and destination are inside the board.
    (2) Source square has a piece.
    (3) Piece belongs to the current player.
    (4) The piece-specific movement rule is satisfied.
    (5) Path is clear (for sliding pieces).
    (6) Not capturing own piece.

Q3: How does the minimax algorithm work?
A3: Minimax simulates the game tree to a certain depth.
    Black tries to minimize the board score (bad for White).
    White tries to maximize the board score (good for White).
    At each level, the algorithm alternates between
    maximizing and minimizing, choosing the best possible
    outcome assuming both players play optimally.

Q4: What is the evaluation function?
A4: It sums the values of all pieces on the board.
    White pieces add to the score, Black pieces subtract.
    Pawn=1, Knight=3, Bishop=3, Rook=5, Queen=9, King=100.
    Positive score means White is ahead, negative means Black.

Q5: What is the difference between the three bot levels?
A5: Level 1: Picks a random legal move.
    Level 2: Prefers the highest-value capture available.
    Level 3: Uses minimax to think ahead 2 moves.

Q6: How is move generation done?
A6: Brute-force: For every square containing a player's piece,
    we try every possible destination (64×64 = 4096 checks).
    is_valid_move() filters out illegal ones.

Q7: What is Alpha-Beta pruning and why isn't it used here?
A7: Alpha-Beta pruning skips branches in the minimax tree
    that cannot affect the final result, making it much faster.
    It's not used here to keep the code beginner-friendly,
    but it's the natural next improvement.

Q8: How does the pawn's two-square first move work?
A8: In is_valid_pawn_move(), we check if the pawn is on its
    starting row (row 6 for White, row 1 for Black).
    If so, we allow a 2-square forward move, but only if
    BOTH the intermediate and destination squares are empty.

Q9: Why is the King's value set to 100?
A9: The King is given a very high value so the evaluation
    function strongly favors positions where the King is safe.
    Capturing the opponent's King instantly makes the score
    jump dramatically, making king safety a top priority.

Q10: How would you add check detection to this project?
A10: After each move, generate all opponent's legal moves.
     If any of those moves can capture the King, the current
     player is in check. To detect checkmate, verify that
     no legal move removes the King from check.

====================================================================

8. CONCLUSION
--------------
This project demonstrates the core concepts of chess programming
in Python: board representation, move validation, move generation,
board evaluation, and game tree search. The three bot levels
illustrate a natural progression from random play to strategic
thinking using the minimax algorithm. The code is written in a
clean, modular, and beginner-friendly style that is easy to
understand, present, and extend in a college project setting.

====================================================================
"""
