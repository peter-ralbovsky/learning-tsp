for f in /home/pdr34/learning-tsp/scripts/medium_size/gegnn/*.sh
do
  echo "processing $f"
  sbatch --time=1:00:00 --array=0-4 /home/pdr34/learning-tsp/scripts/submit.sh $f
done
