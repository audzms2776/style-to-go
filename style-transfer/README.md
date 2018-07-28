# style trasfer 

## AWS sagemaker 

### reference
https://github.com/awslabs/amazon-sagemaker-examples/tree/master/sagemaker-python-sdk/pytorch_mnist <br/>


## 실행시키기!
```python
import sagemaker

sagemaker_session = sagemaker.Session()

bucket = 'sagemakkeryong'
prefix = 'iimage'

role = sagemaker.get_execution_role()
```
sagemaker가 사용할 s3 경로 지정 


```python
from  sagemaker.pytorch  import PyTorch

estimator = PyTorch(entry_point="sage.py",
                    role=role,
                    framework_version='0.4.0',
                    train_instance_count=1,
                    train_instance_type='ml.c4.xlarge',
                    hyperparameters={})
```
실행할 py 파일 입력


```python
estimator.fit({'training': 's3://sagemakkeryong/iimage'})
```
training 값으로 s3경로 입력 

## 소스 설정 

* 저장시에 경로 path = os.path.join(model_dir, 'model.pth')
* 데이터 가져오는 경로 os.path.join(args.data_dir, 'content.jpg')
