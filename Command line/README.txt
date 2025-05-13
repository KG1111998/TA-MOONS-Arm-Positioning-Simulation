#Main Entry Point: TAMOONS_main.py

Command: python main.py --target "03 44 34 +32 09 48" --source IC_348

Execution Order:

Module 1: Gaia YSO Fetch → selected_ysos.csv

Module 2: Grouping, Collision Check → grouped_ysos.csv

Module 3: FITS Image Fetch & Annotate → group_summary.csv

Module 4: Robotic Arm Assignment → Polar Plot Simulation

The software handles all intermediate data handling internally. No manual input needed after command-line start.
