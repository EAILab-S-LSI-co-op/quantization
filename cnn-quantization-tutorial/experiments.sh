# vgg train
python vgg-train.py --model_name="vgg11" --data_path="/workspace/shared/data" --model_path="./models" > vgg11
python vgg-train.py --model_name="vgg13" --data_path="/workspace/shared/data" --model_path="./models" > vgg13
python vgg-train.py --model_name="vgg16" --data_path="/workspace/shared/data" --model_path="./models" > vgg16
python vgg-train.py --model_name="vgg19" --data_path="/workspace/shared/data" --model_path="./models" > vgg19

# qat
python qat.py --model_name="vgg11" --data_path="/workspace/shared/data" --model_path="./models" >> quantization
python qat.py --model_name="vgg13" --data_path="/workspace/shared/data" --model_path="./models" >> quantization
python qat.py --model_name="vgg16" --data_path="/workspace/shared/data" --model_path="./models" >> quantization
python qat.py --model_name="vgg19" --data_path="/workspace/shared/data" --model_path="./models" >> quantization

# ptq
python ptq.py --model_name="vgg11" --data_path="/workspace/shared/data" --model_path="./models" >> quantization
python ptq.py --model_name="vgg13" --data_path="/workspace/shared/data" --model_path="./models" >> quantization
python ptq.py --model_name="vgg16" --data_path="/workspace/shared/data" --model_path="./models" >> quantization
python ptq.py --model_name="vgg19" --data_path="/workspace/shared/data" --model_path="./models" >> quantization
