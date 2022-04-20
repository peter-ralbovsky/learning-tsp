for f in /home/pdr34/learning-tsp/scripts/standard_size/*.sh
do
  echo "processing $f"
  sbatch /home/pdr34/learning-tsp/scripts/submit.sh $f
done