import random
import time



rand_dict = {k: random.random() for k in range(100000000)}



start = time.time()

for key, vals in rand_dict.items():
    val = rand_dict[key]

end = time.time()


print(end - start)


start = time.time()

for key in rand_dict.keys():
    val = rand_dict[key]

end = time.time()


print(end - start)






