import random
from deap import base, creator, tools, algorithms


# --------------------------------------------------
# Board configuration
# 0 = empty cell
# 1 = food
# --------------------------------------------------

board = [
    [0, 1, 0, 0, 1],
    [0, 0, 1, 0, 0],
    [1, 0, 0, 1, 0],
    [0, 1, 0, 0, 1],
    [0, 0, 1, 0, 0]
]

ROWS = len(board)
COLS = len(board[0])

START_ROW = 0
START_COL = 0

NUMBER_OF_MOVES = 20
BITS_PER_MOVE = 2
INDIVIDUAL_SIZE = NUMBER_OF_MOVES * BITS_PER_MOVE


# --------------------------------------------------
# Decode two bits into one move
#
# 00 = up
# 01 = down
# 10 = left
# 11 = right
# --------------------------------------------------

def decode_move(bit1, bit2):

    moves = {
        (0, 0): (-1, 0, "UP"),
        (1, 1): (1, 0, "DOWN"),
        (1, 0): (0, -1, "LEFT"),
        (0, 1): (0, 1, "RIGHT")
    }

    return moves[(bit1, bit2)]


# --------------------------------------------------
# Fitness function
# --------------------------------------------------

def evaluate(individual):

    row = START_ROW
    col = START_COL

    # Copy the board because food can only be collected once.
    game_board = [line.copy() for line in board]

    score = 0

    for i in range(0, INDIVIDUAL_SIZE, 2):

        row_change, col_change, _ = decode_move(
            individual[i],
            individual[i + 1]
        )

        new_row = row + row_change
        new_col = col + col_change

        # Move only if the ant remains inside the board.
        if 0 <= new_row < ROWS and 0 <= new_col < COLS:
            row = new_row
            col = new_col

            # Collect food.
            if game_board[row][col] == 1:
                score += 1
                game_board[row][col] = 0

    return (score,)


# --------------------------------------------------
# DEAP setup
# --------------------------------------------------

if not hasattr(creator, "FitnessMax"):
    creator.create(
        "FitnessMax",
        base.Fitness,
        weights=(1.0,)
    )

if not hasattr(creator, "Individual"):
    creator.create(
        "Individual",
        list,
        fitness=creator.FitnessMax
    )


toolbox = base.Toolbox()

toolbox.register("bit", random.randint, 0, 1)

toolbox.register(
    "individual",
    tools.initRepeat,
    creator.Individual,
    toolbox.bit,
    INDIVIDUAL_SIZE
)

toolbox.register(
    "population",
    tools.initRepeat,
    list,
    toolbox.individual
)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)

toolbox.register(
    "mutate",
    tools.mutFlipBit,
    indpb=0.05
)

toolbox.register(
    "select",
    tools.selTournament,
    tournsize=3
)


# --------------------------------------------------
# Genetic Algorithm
# --------------------------------------------------

def main():

    random.seed(42)

    population = toolbox.population(n=200)
    hall_of_fame = tools.HallOfFame(1)

    algorithms.eaSimple(
        population,
        toolbox,
        cxpb=0.7,
        mutpb=0.2,
        ngen=200,
        halloffame=hall_of_fame,
        verbose=True
    )

    best_individual = hall_of_fame[0]

    print("\nBest fitness:", best_individual.fitness.values[0])
    print("Best individual:", best_individual)

    print("\nBest sequence of moves:")

    moves = []

    for i in range(0, INDIVIDUAL_SIZE, 2):
        _, _, move_name = decode_move(
            best_individual[i],
            best_individual[i + 1]
        )

        moves.append(move_name)

    print(moves)

    print("\nBoard simulation:")
    show_solution(best_individual)


# --------------------------------------------------
# Show the best solution
# --------------------------------------------------

def show_solution(individual):

    row = START_ROW
    col = START_COL
    score = 0

    game_board = [line.copy() for line in board]

    print(f"Start: ({row}, {col})")

    for move_number in range(NUMBER_OF_MOVES):

        index = move_number * 2

        row_change, col_change, move_name = decode_move(
            individual[index],
            individual[index + 1]
        )

        new_row = row + row_change
        new_col = col + col_change

        if 0 <= new_row < ROWS and 0 <= new_col < COLS:
            row = new_row
            col = new_col

            if game_board[row][col] == 1:
                score += 1
                game_board[row][col] = 0
                result = "FOOD"
            else:
                result = "empty"
        else:
            result = "wall"

        print(
            f"Move {move_number + 1:2}: "
            f"{move_name:5} -> ({row}, {col}) -> {result}"
        )

    print("\nFinal score:", score)


if __name__ == "__main__":
    main()