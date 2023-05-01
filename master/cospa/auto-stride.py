
# last update: 2023-05-01

"""
Change these values before run this code:
- FILENAME
- TEST_DATA_PATH
- MODEL
- ARGS_PATCHSIZE
- PATCH_SIZE
- ARGS_STRIDE
- SAVE_DIR_PATH
"""

FILENAME = './segmentation.py'
TEST_DATA_PATH = '--dataset /TEST_DATASET_PATH'
MODEL = '--model /MODEL_PATH/model.h5'
ARGS_PATCHSIZE = '--patch_size'
PATCH_SIZE = 255
ARGS_STRIDE = '--stride'
SAVE_DIR_PATH = 'SAVE_DIR_PATH'

# command example:
# nohup python ./segmentation.py --dataset ./module-test/testdata/images --model ./module-test/model.h5 --patch_size 255 --stride 3 --save_dir ./module-test/result/



import subprocess



command = 'nohup python ' + FILENAME + ' ' + TEST_DATA_PATH + ' ' + MODEL + ' ' + ARGS_PATCHSIZE + ' ' + PATCH_SIZE + ' ' + ARGS_STRIDE + ' '
save_dir = ' --save_dir ' + SAVE_DIR_PATH

stride_list = [False for i in range(0, PATCH_SIZE+1)]
checked = set()
not_checked = set([i for i in range(1, PATCH_SIZE+1)])


# print('PATCH_SIZE = ', PATCH_SIZE)

def findStride(stride, left, right):
    try:
        # range = [1:255]
        # stride = (1 + PATCH_SIZE) // 2
        print('TRY: stride = ', stride)
        print(command + str(stride) + save_dir)

        subprocess.run(command + str(stride) + save_dir, shell=True, check=True)

        for i in range(stride, PATCH_SIZE+1):
            stride_list[i] = True
            if i not in checked:
                checked.add(i)
                not_checked.remove(i)

        print('COMPLETED: subprocess.CompletedProcess, when stride = ',stride)

        right = stride - 1
        stride = (left + right) // 2
        if stride not in checked:
            findStride(stride, left, right)
        else:
            if len(checked) == len(stride_list):
                for min_stride in range(len(stride_list)):
                    if stride_list[min_stride] == True:
                        print('Minimum stride value: ', min_stride)
            else:
                if len(not_checked) > 0:
                    findStride(not_checked.pop(), left, right)
    
    except subprocess.CalledProcessError:
        print('FAILED: subprocess.CalledProcessError, when stride = ',stride)

        for i in range(1, stride+1):
            if i not in checked:
                checked.add(i)
                not_checked.remove(i)

        left = stride + 1
        stride = (left + right) // 2
        if stride not in checked:
            findStride(stride, left, right)
        else:
            if len(checked) == len(stride_list):
                for min_stride in range(len(stride_list)):
                    if stride_list[min_stride] == True:
                        print('Minimum stride value: ', min_stride)
            else:
                if len(not_checked) > 0:
                    findStride(not_checked.pop(), left, right)



if __name__ == '__main__':
    left = 1
    right = PATCH_SIZE
    stride = (left + right) // 2
    findStride(stride, left, right)

    for value in range(len(stride_list)):
        if stride_list[value] == True:
            min_stride = value
            break
    try:
        print('TRY: Main Function: min_stride = ', min_stride)
        subprocess.run(command + str(min_stride) + save_dir, shell=True, check=True)
        print('RESULT: COMPLETED: subprocess.CompletedProcess: Minimum stride value = ', min_stride)

    except subprocess.CalledProcessError:
        print("subprocess.CalledProcessError")