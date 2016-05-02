from SMO import SMO

data_matrix = []
labels = []
with open('data.txt') as f:
    for line in f:
        numbers_str = line.split()
        cols = [float(x) for x in numbers_str]
        data_matrix.append(cols)

with open('labels.txt') as f:
    for line in f:
        labels.append(float(line.strip()))

smo = SMO(data_matrix, labels, 1)
smo.apply()

w, b = smo.get_params()

print(w)
print(b)
