mkdir -p logs/nursescheduling

# preprocess
nohup python preprocess.py dataset=nursescheduling num_workers=10 > logs/nursescheduling/preprocess.log 2>&1 &

# train
nohup python train.py dataset=nursescheduling cuda=0 num_workers=10 job_name=nurse:default > logs/nursescheduling/train.log 2>&1 &

# generate
# ${dir} should be changed to your own path
nohup python generate.py dataset=nursescheduling generator.mask_ratio=0.01 cuda=0 num_workers=10 \
    dir=outputs/models/nursescheduling \
    > logs/nursescheduling/generate:0.01.log 2>&1 &
nohup python generate.py dataset=nursescheduling generator.mask_ratio=0.05 cuda=0 num_workers=10 \
    dir=outputs/models/nursescheduling \
    > logs/nursescheduling/generate:0.05.log 2>&1 &
nohup python generate.py dataset=nursescheduling generator.mask_ratio=0.1 cuda=0 num_workers=10 \
    dir=outputs/models/nursescheduling \
    > logs/nursescheduling/generate:0.1.log 2>&1 &