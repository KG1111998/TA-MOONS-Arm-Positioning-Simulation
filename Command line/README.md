Main Entry Point: TAMOONS_main.py

Command: python TAMOONS_main.py --target "03 44 34 +32 09 48" --source IC_348

Execution Order:
Module 1: Gaia YSO Fetch → selected_ysos.csv

Module 2: Grouping, Collision Check → grouped_ysos.csv

Module 3: FITS Image Fetch & Annotate → group_summary.csv

Module 4: Robotic Arm Assignment → Polar Plot Simulation

The software handles all intermediate data handling internally. No manual input needed after command-line start.

![p1](https://github.com/user-attachments/assets/efbae0f3-26b0-4dbf-99d0-c24308ea6834)

![p2](https://github.com/user-attachments/assets/da0b85d8-3cc7-4b04-ad66-ba5c8c9c72bb)

![p3](https://github.com/user-attachments/assets/6ecc846e-7388-4c21-a94c-3359a14b9833)

![Figure_3](https://github.com/user-attachments/assets/c3a94836-6487-4d67-942b-9c209832fe8c)





















