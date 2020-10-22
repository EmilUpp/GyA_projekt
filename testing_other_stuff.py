
a = 3

for n in range(1,9):
    print("Distance: ", round(n**a, 3), " distance to prior: ", round(n**a - (n-1)**a, 3))
