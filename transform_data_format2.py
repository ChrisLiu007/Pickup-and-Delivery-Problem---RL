from problem.pdp import PDP


def transform_dataset(dataset, file_loc):
    for i, data in enumerate(dataset, 1):
        transform_data(data, f"{file_loc}_{i}.txt")
        print(i)


def transform_data(data, file_loc):
    # helpers
    newline = '\n'
    space = " "
    empty = ""

    # data information
    len_data = len(data['loc'])
    points = data['loc'].tolist() + [data['depot'].tolist()]
    demands = data['demand'].tolist()

    with open(file_loc, "w") as file:
        file.write(f"%  number of nodes{newline}")
        file.write(f"{len_data+1}{newline}")
        file.write(f"%  number of vehicles{newline}")
        file.write(f"1{newline}")
        file.write(f"%  for each vehicle: vehicle index, home node, starting time, capacity{newline}")
        file.write(f"1,{len_data+1},0,1{newline}")
        file.write(f"% number of calls{newline}")
        file.write(f"{len_data//2}{newline}")
        file.write(f"%  for each vehicle, vehicle index, and then a list of calls that can be transported using that vehicle{newline}")
        file.write(f"1,{str(list(range(1, (len_data//2)+1)))[1:-1].replace(space, empty)}{newline}")
        file.write(f"% for each call: call index, origin node, destination node, size, cost of not transporting, lowerbound timewindow for pickup, upper_timewindow for pickup, lowerbound timewindow for delivery, upper_timewindow for delivery{newline}")
        for i in range(0, len_data, 2):
            file.write(f"{i//2+1},{i+1},{i+2},{demands[i]},{9999999999},{0},{999999999},{0},{9999999999}{newline}")

        file.write(f"%  travel times and costs: vehicle, origin node, destination node, travel time (in hours), travel cost (in ?){newline}")
        for i, p1 in enumerate(points):
            for j, p2 in enumerate(points):
                dist = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
                file.write(f"1,{i+1},{j+1},{dist},{dist}{newline}")

        file.write(f"%  node times and costs: vehicle, call, origin node time (in hours), origin node costs (in ?), destination node time (in hours), destination node costs (in ?){newline}")
        for i in range(len_data//2):
            file.write(f"1,{i+1},{0},{0},{0},{0}{newline}")
        file.write(f"% EOF{newline}")


dataset = PDP.make_dataset(size=100, num_samples=10)
file_loc = "transformed_data2/InstanInstancePDP_1.datcePDP"
transform_dataset(dataset, file_loc)



