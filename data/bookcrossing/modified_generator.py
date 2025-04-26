import os

dir_name = 'modified_data_25'
with open(os.path.join(os.path.dirname(__file__), 'removable_items_books_25.txt'), 'r') as f:
    removable_items = f.read()

removable_items = set(list(map(int, removable_items.strip().split(' '))))
print(len(removable_items))

# old_train_set = []
# old_test_set = []
# old_val_set = []

new_train_set = []
new_test_set = []
new_val_set = []

train_empty_sets =set()
test_empty_sets =set()
val_empty_sets =set()
val_on = []
test_on = []
train_on = []

with open(os.path.join(os.path.dirname(__file__),'train.txt'), 'r') as f:
    for line in f.readlines():
        old_items = list(map(int, line.split(' ')))
        new_items = list(set(old_items[1:]).difference(set(removable_items)))
        if len(new_items) ==0:
            train_empty_sets.add(old_items[0])
        else:
            new_train_set.append(old_items[:1] + new_items)
            train_on.append(old_items[0])

with open(os.path.join(os.path.dirname(__file__),'test.txt'), 'r') as f:
    for line in f.readlines():
        old_items = list(map(int, line.split(' ')))
        new_items = list(set(old_items[1:]).difference(set(removable_items)))
        if old_items[0] in train_empty_sets:
            test_empty_sets.add(old_items[0])
        elif len(new_items) ==0:
            test_empty_sets.add(old_items[0])
            new_test_set.append(old_items[:1] + new_train_set[0][1:])
        else:
            new_test_set.append(old_items[:1] + new_items)
            test_on.append(train_on.index(old_items[0]))

with open(os.path.join(os.path.dirname(__file__),'val.txt'), 'r') as f:
    for line in f.readlines():
        old_items = list(map(int, line.split(' ')))
        new_items = list(set(old_items[1:]).difference(set(removable_items)))
        if old_items[0] in train_empty_sets:
            val_empty_sets.add(old_items[0])
        elif len(new_items) ==0:
            val_empty_sets.add(old_items[0])
            new_val_set.append(old_items[:1] + new_train_set[0][1:])
        else:
            new_val_set.append(old_items[:1] + new_items)
            val_on.append(train_on.index(old_items[0]))

# all_empty = train_empty_sets.union(val_empty_sets)
# all_empty = val_empty_sets
# remove_set = set()
# remove_set = remove_set.union(test_empty_sets)
# for item in all_empty:
#     # if len(new_train_set[item]) + len(new_test_set[item]) + len(new_val_set[item]) >=6:
#     if len(new_train_set[item]) + len(new_val_set[item]) >=6:
#         if len(new_val_set[item]) <2:

#             if len(new_train_set) >3:
#                 new_val_set[item].append(new_train_set[item].pop())

#             # elif len(new_test_set[item]) >3:
#             #     new_val_set[item].append(new_test_set[item].pop())

#         # if len(new_test_set[item]) <2:
#         #     if len(new_train_set) >3:
#         #         new_test_set[item].append(new_train_set[item].pop())
#         #     elif len(new_val_set[item]) >3:
#         #         new_test_set[item].append(new_val_set[item].pop())
#         # if len(new_train_set[item]) < 2:
#         #     pass
#     else:
#         remove_set.add(item) 

with open(f'/home/dsls/Desktop/projects/nematgh/Disentangled/data/bookcrossing/{dir_name}/val_on.txt', 'w') as f1, open(f'/home/dsls/Desktop/projects/nematgh/Disentangled/data/bookcrossing/{dir_name}/test_on.txt', 'w') as f2:
    f1.write(' '.join(list(map(str, val_on))))
    f2.write(' '.join(list(map(str, test_on))))

def write_data(path, data):
    with open(path, "w") as f:
        for x in data:
            f.write(" ".join(map(str, x)))
            f.write('\n')

write_data(os.path.join(os.path.dirname(__file__), dir_name,'train.txt'), new_train_set)
write_data(os.path.join(os.path.dirname(__file__), dir_name,'test.txt'), new_test_set)
write_data(os.path.join(os.path.dirname(__file__), dir_name,'val.txt'), new_val_set)