# ----------------------------------------------
# Script Recorded by ANSYS Electronics Desktop Version 2020.2.0
# 11:57:52  janv. 28, 2021
# ----------------------------------------------
import subprocess, sys

import ScriptEnv
ScriptEnv.Initialize("Ansoft.ElectronicsDesktop")
oDesktop.RestoreWindow()
oProject = oDesktop.SetActiveProject("MSR_yshift")
oDesign = oProject.SetActiveDesign("NewDesign_resSinJunc_resSQUID_2nd_struct")
oModule = oDesign.GetModule("ReportSetup")
oModule.ExportToFile(" Plot 1", "C:/Users/themr/OneDrive/Documents/Stage 1/HSFF/Plot 1.csv", False)

#import calculs

commands = [
    ['python', 'calculs.py'],
    ['cd','"Rapport de simulation"'],
    ['pdflatex', 'main.tex'],
    ['start', '"main.pdf"']
]

subprocess.call(['python', 'calculs.py'])
subprocess.call(['pdflatex', 'main.tex'], cwd = 'Rapport de simulation')
subprocess.call(['start','main.pdf'],shell=True, cwd = 'Rapport de simulation')
