for f in /home/pdr34/learning-tsp/scripts/big_size/*.sh
do
  echo "processing $f"
  sbatch --time=10:00:00 /home/pdr34/learning-tsp/scripts/submit.sh $f
done