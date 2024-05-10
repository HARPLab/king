import json
import matplotlib.pyplot as plt

def visualizePath():
    fileN = "../collisionPlanT.json"
    with open(fileN, 'r') as file:
        parsed_data = json.load(file)

    path_x, path_y = [], []
    agent_2_end, agent_3_end, agent_4_end = False, False, False
    path_x_1, path_y_1 = [], []
    path_x_2, path_y_2 = [], []
    path_x_3, path_y_3 = [], []

    for i in range(len(parsed_data)):
        dp = parsed_data[i]
        path_x.append(dp[0][1])
        path_y.append(dp[0][2])
        print(f"{dp[0][1]},{dp[0][2]}")
        if(not agent_2_end):
            if(len(dp) == 2):
                path_x_1.append(dp[1][1])
                path_y_1.append(dp[1][2])
            elif(len(path_x_1) > 0):
                agent_2_end = True
        if(not agent_3_end):
            if(len(dp) == 3):
                path_x_2.append(dp[2][1])
                path_y_2.append(dp[2][2])
            elif(len(path_x_2) > 0):
                agent_3_end = True
        if(not agent_4_end):
            if(len(dp) == 4):
                path_x_3.append(dp[3][1])
                path_y_3.append(dp[3][2])
            elif(len(path_x_3) > 0):
                agent_4_end = True

    plt.figure(figsize = (8,6))
    plt.plot(path_x, path_y, label='Ego', marker='o')
    plt.plot(path_x_1, path_y_1, label='Non-ego-1', marker='o')
    plt.plot(path_x_2, path_y_2, label='Non-ego-2', marker='o')
    plt.plot(path_x_3, path_y_3, label='Non-ego-3', marker='o')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Visualization of two paths')
    plt.grid(True)
    plt.legend()
    plt.savefig('PlanTPath.png', format='png')
    plt.close()

if __name__ == "__main__":
    visualizePath()
