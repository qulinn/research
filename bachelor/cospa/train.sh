#最適epochでモデル訓練
#train dataset
dataset="/patch_127"
root="/home/izumi/kumo/2008/dataset"$dataset

P_dataset=${root}"/Positive"
N_dataset=${root}"/Negative"
U_dataset=${root}"/Unlabeled"

#save trained model
save_root="/home/izumi/kumo/2008/result/cnn"

epochs=200

save_folder=${save_root}${dataset}"/epoch"${epochs}"/model"
log_folder=${save_root}${dataset}"/epoch"${epochs}"/log/train"

echo $save_folder
echo $log_folder

for Prior in 0.1 0.2 0.3 0.4
do
    echo prior $Prior
    for Eta in 0.1 0.3 0.5
    do
        echo eta $Eta
        case $Prior in
            "0.1")
                case $Eta in
                    "0.1")
                        filename="p1e1"
                    ;;
                    "0.3")
                        filename="p1e3"
                    ;;
                    "0.5")
                        filename="p1e5"
                    ;;
                    *)
                        echo "nothing"
                    ;;
                esac
            ;;
            "0.2")
                case $Eta in
                    "0.1")
                        filename="default"
                    ;;
                    "0.3")
                        filename="p2e3"
                    ;;
                    "0.5")
                        filename="p2e5"
                    ;;
                    *)
                        echo "nothing"
                    ;;
                esac
            ;;
            "0.3")
                case $Eta in
                    "0.1")
                        filename="p3e1"
                    ;;
                    "0.3")
                        filename="p3e3"
                    ;;
                    "0.5")
                        filename="p3e5"
                    ;;
                    *)
                        echo "nothing"
                    ;;
                esac
            ;;
            "0.4")
                case $Eta in
                    "0.1")
                        filename="p4e1"
                    ;;
                    "0.3")
                        filename="p4e3"
                    ;;
                    "0.5")
                        filename="p4e5"
                    ;;
                    *)
                        echo "nothing"
                    ;;
                esac
            ;;
            *)
                echo "NOTHING"
            ;;
        esac
        echo $filename
        nohup python /home/izumi/kumo/2008/cospa/train.py \
            --PNU --P_dataset ${P_dataset} --N_dataset ${N_dataset} --U_dataset ${U_dataset} --save_dir ${save_folder}/$filename \
            --P_n 1000 --N_n 1000 --U_n 2000 \
            --model CNN --prior ${Prior} --eta ${Eta} --epochs ${epochs} --batchsize 128 --lr 0.000001 \
            > ${log_folder}/$filename.txt 
    done
done    
