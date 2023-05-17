conda activate imgCap
#python train.py --config configs/config_model1.yaml
#python train.py --config configs/config_model3.yaml
#python train.py --config configs/config_model4.yaml
echo "###############################"
echo "####   Runing Model5 Now   ####"
echo "###############################"
python train.py --config configs/config_model5.yaml

echo "###############################"
echo "####   Runing Model6 Now   ####"
echo "###############################"

python train.py --config configs/config_model6.yaml

echo "###############################"
echo "####   Runing Model7 Now   ####"
echo "###############################"

python train.py --config configs/config_model7.yaml

echo "###############################"
echo "####   Runing Model2 Now   ####"
echo "###############################"
python train.py --config configs/config_model2.yaml
