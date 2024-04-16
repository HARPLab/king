inp = []
fileName = '../PlanT.txt' # 'aim-bev.txt'/'transfuser.txt'
numRoutes = 0
totSum = 0
totRoutes = 0
with open(fileName, 'r') as file:
    for line in file:
        val = line.split(",")
        totRoutes += 1
        if(val[-1] != '0\n'):
            numRoutes += 1
            totSum += len(val)
print(f"Total routes tested: {totRoutes}")
print(f"Routes with collisions: {numRoutes}")
if(numRoutes > 0):
    print(f"Avg num timesteps to collision: {totSum / numRoutes}")
