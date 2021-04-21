#%%
import os

def fix_csv(name):
    """ 
    HFSS will sometime output commas in its columns names rendering de csv
    hard to use, this function will remove the undesired commas
    """
    with open(name) as f:
        ls = f.readlines()

    with open(name, "w") as f:
        ls[0] = ls[0]   .replace('","', '_placeholder_')\
                        .replace(",", " ")\
                        .replace("_placeholder_", '","')
        f.writelines(ls)



def AgilentParse(folder):
    nfolder = folder+'_Agilent_raw'
    i = nfolder.rfind('/')
    surfolder = folder[:i]
    subfolder = nfolder[i+1:]
    if subfolder in os.listdir(surfolder):
        print(f'{folder} already parsed')
        return
    os.rename(folder, nfolder)
    os.mkdir(folder)
    for i in os.listdir(nfolder):
        lines = []
        with open(nfolder + '/' + i, 'r') as f:
            lines = [i for i in f.readlines() if i[0] != '!' and 'BEGIN' not in i and 'END' not in i]

        with open(folder + '/' + i, 'w+') as f:
            f.writelines(lines)


# %%
