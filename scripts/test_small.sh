for f in /home/pdr34/learning-tsp/scripts/very_small/*.sh
do
  echo "processing $f"
  $f &
done
