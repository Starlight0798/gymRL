import numpy as np

# 指数衰减
def get_decay(start, end, cnt, val):
    # val = end + (start - end) * np.exp(-cnt / decay)
    return cnt / np.log((start - end) / (val - end))


start = 0.95
end = 0.01
pair = (50000, 0.1)

decay = get_decay(start, end, pair[0], pair[1])

print('start =', start)
print('end =', end)
print('pair =', pair)
print('decay =', decay)

def get_val(start, end, cnt, decay):
    return end + (start - end) * np.exp(-cnt / decay)

def get_cnt(start, end, val, decay):
    return -decay * np.log((val - end) / (start - end))

cnt = 1000000
val = get_val(start, end, cnt, decay)
print('-' * 20)
print('cnt =', cnt)
print(f'val = {val:e}')

val = 0.15
cnt = get_cnt(start, end, val, decay)
print('-' * 20)
print(f'val = {val:e}')
print('cnt =', cnt)
