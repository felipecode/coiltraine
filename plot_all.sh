PARAM_FOLDER=visualization/plotting_params
TEMPLATE_FILE=plotting_template.py

sed -e 's/@CAMERA@//g' -e 's/@NOISE@/False/g' ${PARAM_FOLDER}/${TEMPLATE_FILE} > ${PARAM_FOLDER}/3cam_nonoise.py
sed -e 's/@CAMERA@/'\''camera'\'': '\''central'\''/g' -e 's/@NOISE@/False/g' ${PARAM_FOLDER}/${TEMPLATE_FILE} > ${PARAM_FOLDER}/central_nonoise.py
sed -e 's/@CAMERA@//g' -e 's/@NOISE@/True/g' ${PARAM_FOLDER}/${TEMPLATE_FILE} > ${PARAM_FOLDER}/3cam_noise.py
sed -e 's/@CAMERA@/'\''camera'\'': '\''central'\''/g' -e 's/@NOISE@/True/g' ${PARAM_FOLDER}/${TEMPLATE_FILE} > ${PARAM_FOLDER}/central_noise.py

declare -a MODES=("central_nonoise" "3cam_nonoise" "central_noise" "3cam_noise")
#declare -a MODES=("central_nonoise")

## now loop through the above array
for MODE in "${MODES[@]}"
do
  COMMAND="source activate tf; export COIL_DATASET_PATH=./data; python3 run_plotting.py --folder eccv_debug -p $MODE -f $MODE -ebv; sleep 100000"
  echo $COMMAND
  screen -d -m -S plotting_$MODE bash -c "$COMMAND"
done
