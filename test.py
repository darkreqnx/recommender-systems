import csv

filename = "ratings50.csv"

sparsematrix = []

overall_movie_rating_mean = 0

mx = 0

with open(filename) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            print(f'{row[0]} \t {row[1]} \t {row[2]}')

            if int(row[1]) > mx:
                mx = int(row[1])
                i = row[0]

            entry = []

            entry.append(row[0])
            entry.append(row[1])
            entry.append(row[2])

            sparsematrix.append(entry)

            line_count += 1
            # if line_count == 3:
            #     break

    line_count -= 1
    print(f'Ratings Processsed: {line_count} \n max count : {mx} {i}')
