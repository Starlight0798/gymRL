import numpy as np

# 指数衰减
def get_decay(start, end, cnt, val):
    # val = end + (start - end) * np.exp(-cnt / decay)
    return cnt / np.log((start - end) / (val - end))

start = 1e-3
end = 1e-5
pair = (400, 1e-3)

decay = get_decay(start, end, pair[0], pair[1])

print('start =', start)
print('end =', end)
print('pair =', pair)
print('decay =', decay)

def get_val(start, end, cnt, decay):
    return end + (start - end) * np.exp(-cnt / decay)

def get_cnt(start, end, val, decay):
    return -decay * np.log((val - end) / (start - end))

cnt = 450
val = get_val(start, end, cnt, decay)
print('-' * 20)
print('cnt =', cnt)
print(f'val = {val:e}')

val = 2e-4
cnt = get_cnt(start, end, val, decay)
print('-' * 20)
print(f'val = {val:e}')
print('cnt =', cnt)