 while true
 do
     free -g | grep Mem 2>&1 | tee -a memory_usage.txt
     sleep 5
 done
