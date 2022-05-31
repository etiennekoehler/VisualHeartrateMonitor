from pipeline import Pipeline
import matplotlib.pyplot as plt

def run_pipeline():
    # Use a breakpoint in the code line below to debug your script.

    pipe = Pipeline()
    time, BPM, uncertainty = pipe.run_on_video('/Users/etienne/Movies/HeartrateVideo2.mov', roi_approach="patches", roi_method="faceparsing")

    plt.figure()
    plt.plot(time, BPM)
    plt.fill_between(time, BPM - uncertainty, BPM + uncertainty, alpha=0.2)
    plt.show()

def run_realtime_pipeline():


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_pipeline()


