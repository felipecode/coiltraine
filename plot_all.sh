#!/usr/bin/env bash
DEBUG=$1
PARAM_FOLDER=visualization/plotting_params
TEMPLATE_FILE=plotting_template.py


sed -e 's/@CAMERA@//g' -e 's/@NOISE@/False/g' -e 's/@BEST_N_PERCENT@/None/g' ${PARAM_FOLDER}/${TEMPLATE_FILE} > ${PARAM_FOLDER}/3cam_nonoise_allmodels.py
sed -e 's/@CAMERA@//g' -e 's/@NOISE@/False/g' -e 's/@BEST_N_PERCENT@/50./g' ${PARAM_FOLDER}/${TEMPLATE_FILE} > ${PARAM_FOLDER}/3cam_nonoise_best50percent.py
sed -e 's/@CAMERA@/'\''camera'\'': '\''central'\''/g' -e 's/@NOISE@/False/g' -e 's/@BEST_N_PERCENT@/None/g' ${PARAM_FOLDER}/${TEMPLATE_FILE} > ${PARAM_FOLDER}/central_nonoise_allmodels.py
sed -e 's/@CAMERA@/'\''camera'\'': '\''central'\''/g' -e 's/@NOISE@/False/g' -e 's/@BEST_N_PERCENT@/50./g' ${PARAM_FOLDER}/${TEMPLATE_FILE} > ${PARAM_FOLDER}/central_nonoise_best50percent.py
sed -e 's/@CAMERA@//g' -e 's/@NOISE@/True/g' -e 's/@BEST_N_PERCENT@/None/g' ${PARAM_FOLDER}/${TEMPLATE_FILE} > ${PARAM_FOLDER}/3cam_noise_allmodels.py
sed -e 's/@CAMERA@//g' -e 's/@NOISE@/True/g' -e 's/@BEST_N_PERCENT@/50./g' ${PARAM_FOLDER}/${TEMPLATE_FILE} > ${PARAM_FOLDER}/3cam_noise_best50percent.py
sed -e 's/@CAMERA@/'\''camera'\'': '\''central'\''/g' -e 's/@NOISE@/True/g' -e 's/@BEST_N_PERCENT@/None/g' ${PARAM_FOLDER}/${TEMPLATE_FILE} > ${PARAM_FOLDER}/central_noise_allmodels.py
sed -e 's/@CAMERA@/'\''camera'\'': '\''central'\''/g' -e 's/@NOISE@/True/g' -e 's/@BEST_N_PERCENT@/50./g' ${PARAM_FOLDER}/${TEMPLATE_FILE} > ${PARAM_FOLDER}/central_noise_best50percent.py

#declare -a MODES=("central_nonoise_allmodels" "central_nonoise_best50percent")
if [ -z "$DEBUG" ]
then 
  declare -a MODES=("central_nonoise_allmodels" "central_nonoise_best50percent" "3cam_nonoise_allmodels" "3cam_nonoise_best50percent" "central_noise_allmodels" "central_noise_best50percent" "3cam_noise_allmodels" "3cam_noise_best50percent")
else
  declare -a MODES=("central_nonoise_allmodels")
fi

## now loop through the above array
for MODE in "${MODES[@]}"
do
  if [ -z "$DEBUG" ]
  then 
    COMMAND="source activate tf; export COIL_DATASET_PATH=./data; python3 -u run_plotting.py --folder eccv_debug -p $MODE -f $MODE -ebv > logs/$MODE.log 2>&1 ; sleep 1"
  else
    COMMAND="source activate tf; export COIL_DATASET_PATH=./data; python3 -u run_plotting.py --folder eccv_debug_debug -p $MODE -f $MODE -ebv > logs/$MODE.log 2>&1; sleep 100000"
    #COMMAND="source activate tf; export COIL_DATASET_PATH=./data; python3 -u run_plotting.py --folder eccv_debug_debug -p $MODE -f $MODE -ebv ; sleep 100000"
  fi
  echo $COMMAND
  screen -d -m -S plotting_$MODE bash -c "$COMMAND"
done
