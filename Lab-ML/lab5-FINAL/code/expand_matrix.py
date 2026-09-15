import numpy as np


def enlarge_smiley(input_file, output_file, scale):
    with open(input_file, "r") as f:
        rows = int(f.readline())
        cols = int(f.readline())

        matrix = []
        for _ in range(rows):
            matrix.append(list(map(int, f.readline().split())))

    matrix = np.array(matrix)

    if matrix.shape != (rows, cols):
        raise ValueError("Dimensioni della matrice non corrette")

    enlarged = np.repeat(
        np.repeat(matrix, scale, axis=0),
        scale,
        axis=1
    )

    new_rows, new_cols = enlarged.shape

    with open(output_file, "w") as f:
        f.write(f"{new_rows}\n")
        f.write(f"{new_cols}\n")

        for row in enlarged:
            f.write(" ".join(map(str, row)) + "\n")

enlarge_smiley(
    "smiley.txt",
    "smiley_large_20.txt",
    scale=20
)