for N in range(100, 2600, 100)+range(3000, 11000, 1000):	
	f = open(str(N)+".txt")

	t = ''

	for i in f.xreadlines():
		s = map(lambda x: int(x), i.replace('e+', '').replace('.', '').strip().split(' '))
		if(len(s) == 6):
			t += ('%d %d %d %d %d %d\n' % (s[0], s[1], s[2], s[3], s[3], s[3]))
		elif(len(s) == 9):
			t += ('%d %d %d %d %d %d\n' % (s[0], s[1], (N+1)%s[1], s[2], s[4], s[6]))
		elif(len(s) == 8):
			t += ('%d %d %d %d %d %d\n' % (s[0], s[1], s[2], s[3], s[5], s[5]))
		else:
			t += ('%d %d %d %d %d %d\n' % (s[0], s[1], s[2], s[3], s[5], s[7]))

	f.close()

	f = open(str(N)+".txt", 'w')
	f.write(t)
	f.close()
