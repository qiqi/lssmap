s = 0.02

n = 1000000
x = zeros(n)
x[0] = rand()
for i in range(1,n):
    x[i] = (x[i-1] * 2) % 1 + s * sin(2 * pi * x[i-1])

hist(x, 1024, normed=True)

x0 = x.copy()
for i in range(n-1,0,-1):
    x0[i-1] += (x0[i] - (x0[i-1] * 2) % 1) / 2
r = x0[1:] - (x0[:-1] * 2) % 1
print((r**2).sum())

figure()
hist(x0, 1024, normed=True)
xlabel('x')
ylabel(r'$\rho$')
savefig('plus')
