import json as js
from turtle import color
import matplotlib.pyplot as plt

## want four image high low run walk
# each has five line 256 32 15 10 5


with open('errors_2.json', newline='') as jsonfile:
    data = js.load(jsonfile)
    data_key = data['hr_errors']
    high={'256.0Hz':[], '32.0Hz':[], '15.0Hz':[], '10.0Hz':[], '5.0Hz':[]}
    low={'256.0Hz':[], '32.0Hz':[], '15.0Hz':[], '10.0Hz':[], '5.0Hz':[]}
    run={'256.0Hz':[], '32.0Hz':[], '15.0Hz':[], '10.0Hz':[], '5.0Hz':[]}
    walk={'256.0Hz':[], '32.0Hz':[], '15.0Hz':[], '10.0Hz':[], '5.0Hz':[]}
    epochs=[i for i in range(1,700)]

    for result in data_key:
        if result["exercise"]=="high":
            if(result["frequency"]=="256.0Hz"):
                high['256.0Hz'].append(round(float(result["error"]),2))
            elif(result["frequency"]=="32.0Hz"):
                high['32.0Hz'].append(round(float(result["error"]),2))
            elif(result["frequency"]=="15.0Hz"):
                high['15.0Hz'].append(round(float(result["error"]),2))
            elif(result["frequency"]=="10.0Hz"):
                high['10.0Hz'].append(round(float(result["error"]),2))
            else:
                high['5.0Hz'].append(round(float(result["error"]),2))
        elif result["exercise"]=="low":
            if(result["frequency"]=="256.0Hz"):
                low['256.0Hz'].append(round(float(result["error"]),2))
            elif(result["frequency"]=="32.0Hz"):
                low['32.0Hz'].append(round(float(result["error"]),2))
            elif(result["frequency"]=="15.0Hz"):
                low['15.0Hz'].append(round(float(result["error"]),2))
            elif(result["frequency"]=="10.0Hz"):
                low['10.0Hz'].append(round(float(result["error"]),2))
            else:
                low['5.0Hz'].append(round(float(result["error"]),2))      
        elif result["exercise"]=="run":
            if(result["frequency"]=="256.0Hz"):
                run['256.0Hz'].append(round(float(result["error"]),2))
            elif(result["frequency"]=="32.0Hz"):
                run['32.0Hz'].append(round(float(result["error"]),2))
            elif(result["frequency"]=="15.0Hz"):
                run['15.0Hz'].append(round(float(result["error"]),2))
            elif(result["frequency"]=="10.0Hz"):
                run['10.0Hz'].append(round(float(result["error"]),2))
            else:
                run['5.0Hz'].append(round(float(result["error"]),2))
        else:
            if(result["frequency"]=="256.0Hz"):
                walk['256.0Hz'].append(round(float(result["error"]),2))
            elif(result["frequency"]=="32.0Hz"):
                walk['32.0Hz'].append(round(float(result["error"]),2))
            elif(result["frequency"]=="15.0Hz"):
                walk['15.0Hz'].append(round(float(result["error"]),2))
            elif(result["frequency"]=="10.0Hz"):
                walk['10.0Hz'].append(round(float(result["error"]),2))
            else:
                walk['5.0Hz'].append(round(float(result["error"]),2))

    # plt.title("High resist bike")
    # plt.ylabel("Error rate")
    # plt.xlabel("Epochs")
    # plt.plot(epochs,high["256.0Hz"][1:],color=(255/255,0/255,0/255), label='256.0Hz')
    # plt.plot(epochs,high["32.0Hz"][1:],color=(255/255,0/255,255/255), label='32.0Hz')
    # plt.plot(epochs,high["15.0Hz"][1:],color=(255/255,255/255,0/255), label='15.0Hz')
    # plt.plot(epochs,high["10.0Hz"][1:],color=(0/255,0/255,255/255), label='10.0Hz')
    # plt.legend(loc = 'upper right')
    # plt.savefig("high.png")



    # plt.title("low resist bike")
    # plt.ylabel("Error rate")
    # plt.xlabel("Epochs")
    # plt.plot(epochs,low["256.0Hz"][1:],color=(255/255,0/255,0/255), label='256.0Hz')
    # plt.plot(epochs,low["32.0Hz"][1:],color=(255/255,0/255,255/255), label='32.0Hz')
    # plt.plot(epochs,low["15.0Hz"][1:],color=(255/255,255/255,0/255), label='15.0Hz')
    # plt.plot(epochs,low["10.0Hz"][1:],color=(0/255,0/255,255/255), label='10.0Hz')
    # plt.legend(loc = 'upper right')
    # plt.savefig("low.png")


    # plt.title("run")
    # plt.ylabel("Error rate")
    # plt.xlabel("Epochs")
    # plt.plot(epochs,run["256.0Hz"][1:],color=(255/255,0/255,0/255), label='256.0Hz')
    # plt.plot(epochs,run["32.0Hz"][1:],color=(255/255,0/255,255/255), label='32.0Hz')
    # plt.plot(epochs,run["15.0Hz"][1:],color=(255/255,255/255,0/255), label='15.0Hz')
    # plt.plot(epochs,run["10.0Hz"][1:],color=(0/255,0/255,255/255), label='10.0Hz')
    # plt.legend(loc = 'upper right')
    # plt.savefig("run.png")


    plt.title("walk")
    plt.ylabel("Error rate")
    plt.xlabel("Epochs")
    plt.plot(epochs,walk["256.0Hz"][1:],color=(255/255,0/255,0/255),label='256.0Hz')
    plt.plot(epochs,walk["32.0Hz"][1:],color=(255/255,0/255,255/255),label='32.0Hz')
    plt.plot(epochs,walk["15.0Hz"][1:],color=(255/255,255/255,0/255),label='15.0Hz')
    plt.plot(epochs,walk["10.0Hz"][1:],color=(0/255,0/255,255/255), label='10.0Hz')
    plt.legend(loc = 'upper right')
    plt.savefig("walk.png")


