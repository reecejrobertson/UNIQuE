import time
import emulator

def demo_shors(a, N, m, n, iters=1000, ws='', display=False):
    times = []
    accuracy = 0
    for i in range(iters):
        start_time = time.time()
        guesses = emulator.shors(N, a, m, n)
        times.append(time.time() - start_time)
        if (len(guesses) > 1):
            if (guesses[0] != 1 and N % guesses[0] == 0):
                accuracy += 1
            elif (guesses[1] != 1 and N % guesses[1] == 0):
                accuracy += 1
        if display == True:
            print('Guesses: ' + str(guesses) + ';' + ws + '\t ' + str(N) + '/' + str(guesses[0]) + ' = ' + str(N/guesses[0]))
    
    with open('output.txt', 'a') as out_file:
        out_file.write('----\n')
        out_file.write('a = ' + str(a) + '\n')
        out_file.write('N = ' + str(N) + '\n')
        out_file.write('m = ' + str(m) + '\n')
        out_file.write('n = ' + str(n) + '\n')
        out_file.write('Accuracy:     ' + str(accuracy/iters) + '\n')
        out_file.write('Average time: ' + str(sum(times)/iters) + '\n')
        out_file.write('\n')
        
        
N =  67 * 127
for i in range(1, 30):
    print('i =', i)
    demo_shors(200, N, i, 14)
    
N =  42781
for i in range(1, 30):
    print('i =', i)
    demo_shors(200, N, i, 16)