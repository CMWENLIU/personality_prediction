nohup ~/anaconda/bin/python3.6 train.py > result.out 2> err.err &
wait
echo "Process done" | mail -s "Process done" xxliu10@ualr.edu
echo "result saved on result.out"

# python eval.py  --checkpoint_dir="./runs/1521568263/checkpoints/"
