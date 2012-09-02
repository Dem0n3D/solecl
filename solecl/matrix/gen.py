# -*- coding: utf-8 -*-

import random, sys

random.seed()

nlist = map(lambda i: int(i), sys.argv[1:])

for n in nlist:
	M = 10
	f = open('matrix%d.txt' % n, 'w')
	for i in xrange(n):
		s = ''
		for j in xrange(n):
			s += '%.2f ' % (random.random() * M * (-1 if random.randint(1, 2)%2 == 0 else 1))
		s += '%.2f\n' % (random.random() * M * 10 * (-1 if random.randint(1, 2)%2 == 0 else 1))
		f.write(s)
	f.close()
	print(n)