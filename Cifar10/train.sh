#python3 train.py --train True --network resnet50 --loss BCE --resume 1
#python3 train.py --train True --network resnet50 --loss ECE --resume 1
#python3 train.py --train True --network resnet50 --loss focal  --resume 1
#python3 train.py --train True --network resnet50 --loss ASL --resume 1
#python3 train.py --train True --network resnet50 --loss F-ECE --resume 1
python3 train.py --train True --network resnet50 --loss Recall  --resume 0
