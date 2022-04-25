for MODEL in outputs/**/*-1234_*
 do
    echo $MODEL
    sbatch --time=10:00:00 /home/pdr34/learning-tsp/scripts/submit.sh /home/pdr34/learning-tsp/scripts/eval_model.sh $MODEL
done
