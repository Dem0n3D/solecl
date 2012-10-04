for N in range(100, 2600, 100)+range(3000, 11000, 1000):	
	f = open(str(N)+".txt")

	t = ''

	for i in f.xreadlines():
		s = map(lambda x: int(x), i.replace('e+', '').replace('.', '').strip().split(' '))
		t += ("%d %d %d %d %d %d %0.3f\n") % (s[0], s[1], s[2], s[3], s[4], s[5], (s[3]+s[4]+s[5])/3.0)

	
	f.close()

	f = open(str(N)+".txt", 'w')
	f.write(t)
	f.close()
