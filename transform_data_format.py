from problem.pdp import PDP


def transform_dataset(dataset, file_loc):
    for i, data in enumerate(dataset, 1):
        transform_data(data, f"{file_loc}_{i}.dat")
        print(i)


def transform_data(data, file_loc):
    # helpers
    tab = '\t'
    newline = '\n'
    comma = ", "

    # data information
    len_data = len(data['loc'])

    with open(file_loc, "w") as file:
        file.write(f"data;{newline}")
        file.write(f"{newline}")
        file.write(f"set Nv:= {str(list(range(1, len_data+3)))[1:-1].replace(',', tab)};{newline}")
        file.write(f"set Np:= {str(list(range(1, len_data+1)[::2]))[1:-1].replace(',', '')};{newline}")
        file.write(f"set Nd:= {str(list(range(1, len_data+1)[1::2]))[1:-1].replace(',', '')};{newline}")
        file.write(f"{newline}")
        file.write(f"param o:= {len_data+1};{newline}")
        file.write(f"param d:= {len_data+2};{newline}")
        file.write(f"{newline}")

        q_list = data['demand'][::2].tolist()
        param_Q = str(list(enumerate(q_list, 1)))[2:-2].replace("), (", newline).replace(", ", tab)
        file.write(f"param Q:= {newline}{param_Q};{newline}")
        file.write(f"{newline}")

        file.write(f"param K:= {10000};{newline}")
        file.write(f"param BigM:= {25};{newline}")
        file.write(f"param cs:= {999999};{newline}")
        file.write(f"{newline}")

        adjacency_matrix = [[None] * len_data for _ in range(len_data)]
        points = data['loc'].tolist()
        for i, p1 in enumerate(points):
            for j, p2 in enumerate(points):
                adjacency_matrix[i][j] = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

        file.write(f"param D: {newline}{tab}")
        file.write(f"{str(list(range(1, len_data+1)))[1:-1].replace(',', tab)}:={newline}")
        for i in range(1, len_data+1):
            file.write(f"{i}{tab}{str(adjacency_matrix[i - 1])[1:-1].replace(comma, tab)}{newline}")
        file.write(";")


dataset = PDP.make_dataset(size=20, num_samples=10)
data = dataset[0]
print(data)
file_loc = "transformed_data/InstancePDP"
transform_dataset(dataset, file_loc)



