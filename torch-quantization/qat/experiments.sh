# vgg train
# python vgg-train.py --model_name="VGG11" --data_path="/workspace/shared/data" --model_path="./models" > vgg11
# python vgg-train.py --model_name="VGG13" --data_path="/workspace/shared/data" --model_path="./models" > vgg13
# python vgg-train.py --model_name="VGG16" --data_path="/workspace/shared/data" --model_path="./models" > vgg16
# python vgg-train.py --model_name="VGG19" --data_path="/workspace/shared/data" --model_path="./models" > vgg19

# vgg quantization
python qat.py --model_name="VGG11" --data_path="/workspace/shared/data" --model_path="./models" > vgg11_quant
python qat.py --model_name="VGG13" --data_path="/workspace/shared/data" --model_path="./models" > vgg13_quant
python qat.py --model_name="VGG16" --data_path="/workspace/shared/data" --model_path="./models" > vgg16_quant
python qat.py --model_name="VGG19" --data_path="/workspace/shared/data" --model_path="./models" > vgg19_quant
