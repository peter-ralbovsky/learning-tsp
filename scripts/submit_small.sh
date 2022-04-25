for f in /home/pdr34/learning-tsp/scripts/very_small/*.sh
do
  echo "processing $f"
  sbatch --array=0-4 /home/pdr34/learning-tsp/scripts/submit.sh $f
done
