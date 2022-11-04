
#!/bin/bash
digits=(01 27 89)
for val in ${digits[@]}; 
do
    python main.py $val 1
    python main.py $val 2
    python main.py $val 3
done