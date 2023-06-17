import matplotlib.pyplot as plt

def read_input(input_path=None):
    input_cost_lines = [
        "1 1 1 1 50 1 1 1 1 1",
        "1 1 1 1 1 1 1 1 1 1",
        "1 5 5 5 1 5 5 5 1 1",
        "1 5 5 5 1 5 5 5 1 1",
        "1 5 5 5 1 5 5 5 1 1",
        "10 0 0 0 0 0 0 0 0 1",
    ]

    input_cost = []
    if input_path is None:
        lines = input_cost_lines
    else:
        with open(input_path) as f:
            lines = f.readlines()
    for line in lines:
        input_cost.append([int(s) for s in line.split(" ")])
    return input_cost

matrix = read_input()
plt.imshow(matrix)
plt.show()
