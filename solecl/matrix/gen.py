# -*- coding: utf-8 -*-

import random

random.seed()

nlist = [10, 100, 1000, 10000]

for n in nlist:
	M = n*10
	f = open('matrix%d.txt' % n, 'w')
	for i in xrange(n):
		s = ''
		for j in xrange(n):
			s += '%f ' % (random.random() * M * (-1 if random.randint(1, 2)%2 == 0 else 1))
		s += '%f\n' % (random.random() * M * (-1 if random.randint(1, 2)%2 == 0 else 1))
		f.write(s)
	f.close()
	print(n)