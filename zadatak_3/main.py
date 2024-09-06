


l = [[1,2,1], [1,2,2], [0,1,1]]

most_frequent = list(map(lambda x: max(x, key=x.count), l))


print(most_frequent)


