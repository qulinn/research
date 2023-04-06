data="/data/Users/izumi/cospa/dataset-all/test/2009-testdata/images"
root="/data/Users/izumi/cospa/result/2008/2009-testdata/resnet18"

dataset="/patchsize127"
epochs="/epoch800"
# #/p1e1 /p1e3 /p1e5 /default /p2e3 /p2e5 /p3e1 /p3e3 /p3e5 /p4e1 /p4e3 /p4e5

for param in /default /p2e3 /p2e5 /p3e3 /p3e5 /p4e3 /p4e5
do
    $model_path = ${root}${dataset}${epochs}"/model"${param}"/model.h5"
    $save_dir = ${root}${dataset}${epochs}"/seg"${param}
    $log_file = ${root}${dataset}${epochs}"/log"${param}".txt"

    echo $model_path
    echo $save_dir
    echo $log_file
    # #model_dir=${root}${dataset}${epochs}"/model"${param}

    # model_path = "/data/Users/izumi/cospa/result/2008/2008-testdata/undergraduate-thesis/ResNet18/patchsize-127/epoch800/model/p4e5/model.h5"
    # #save_dir=${root}${dataset}${epochs}"/seg"${param}
    # save_path = "/data/Users/izumi/cospa/result/2008/2009-testdata/resnet18/patchsize127/epoch800/seg"
    # #log_dir="/data/Users/izumi/cospa/result/2008/2009-testdata/resnet18/patchsize127/epoch800/log"
    # log = "/data/Users/izumi/cospa/result/2008/2009-testdata/resnet18/patchsize127/epoch800/log"

    # nohup python /data/Users/izumi/cospa/cospa/segmentation.py\
    #     --dataset /data/Users/izumi/cospa/dataset-all/test/2009-testdata/images \
    #     --model /data/Users/izumi/cospa/result/2008/2008-testdata/undergraduate-thesis/resnet18/patchsize127/epoch800/model/p4e5/model.h5 \
    #     --patch_size 127 --stride 23 \

    #     > /data/Users/izumi/cospa/result/2008/2009-testdata/resnet18

done
