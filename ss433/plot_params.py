import matplotlib.pyplot as plt


plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['xtick.minor.size'] = 6
plt.rcParams['xtick.minor.width'] = 2

plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.major.width'] = 3
plt.rcParams['ytick.minor.size'] = 6
plt.rcParams['ytick.minor.width'] = 2

plt.rcParams['axes.linewidth'] = 3
plt.rcParams['xtick.labelsize'] = 'Large'
plt.rcParams['ytick.labelsize'] = 'Large'
plt.rcParams['xtick.major.pad']='7'
plt.rcParams['ytick.major.pad']='7'

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['lines.linewidth'] = 2.5

plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.top'] = True
plt.rcParams['axes.unicode_minus'] = False

SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 24
HUGE_SIZE = 30

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title