# ann-comparision

## Local MacBook Prod
### Recommended: Install the Mac-optimized version
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

pip install tensorflow-macos
pip install tensorflow-metal  # This allows usage of the Mac GPU
pip install pandas numpy scikit-learn

python ann.py
```

## Windows
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Standard installation
pip install tensorflow
pip install pandas numpy scikit-learn
```

Epochs  | Optimizer  | Time(s)  | Best Acc   | Epoch  | Best Loss  | Epoch  | Final Acc  | Final Loss
------- | ---------- | -------- | ---------- | ------ | ---------- | ------ | ---------- | ----------
10      | adam       | 1.45     | 0.8043     | 10     | 0.425      | 10     | 0.8043     | 0.425     
10      | adagrad    | 1.27     | 0.7766     | 10     | 0.543      | 10     | 0.7766     | 0.543     
10      | adadelta   | 1.37     | 0.4625     | 10     | 0.7072     | 10     | 0.4625     | 0.7072    
10      | adamax     | 1.4      | 0.8026     | 10     | 0.436      | 10     | 0.8026     | 0.436     
10      | rmsprop    | 1.3      | 0.8518     | 10     | 0.3575     | 10     | 0.8518     | 0.3575    
10      | sgd        | 1.18     | 0.8087     | 10     | 0.4333     | 10     | 0.8087     | 0.4333    
50      | adam       | 5.8      | 0.864      | 44     | 0.3363     | 50     | 0.8636     | 0.3363    
50      | adagrad    | 5.62     | 0.7971     | 50     | 0.5672     | 50     | 0.7971     | 0.5672    
50      | adadelta   | 5.64     | 0.6674     | 50     | 0.6365     | 50     | 0.6674     | 0.6365    
50      | adamax     | 5.83     | 0.8577     | 50     | 0.3458     | 50     | 0.8577     | 0.3458    
50      | rmsprop    | 5.63     | 0.8401     | 50     | 0.3789     | 50     | 0.8401     | 0.3789    
50      | sgd        | 5.26     | 0.8596     | 48     | 0.3469     | 50     | 0.8583     | 0.3469    
100     | adam       | 11.29    | 0.8662     | 95     | 0.3335     | 100    | 0.8652     | 0.3335    
100     | adagrad    | 10.71    | 0.796      | 48     | 0.476      | 100    | 0.796      | 0.476     
100     | adadelta   | 10.95    | 0.58       | 100    | 0.7323     | 100    | 0.58       | 0.7323    
100     | adamax     | 11.26    | 0.8627     | 96     | 0.3379     | 100    | 0.8616     | 0.3379    
100     | rmsprop    | 11.11    | 0.8655     | 92     | 0.3328     | 100    | 0.8655     | 0.3328    
100     | sgd        | 10.27    | 0.8611     | 99     | 0.3437     | 100    | 0.8608     | 0.3437    

## Google Colab
To upload `Churn_Modelling.csv`
```
from google.colab import files
uploaded = files.upload()
```
Epochs  | Optimizer  | Time(s)  | Best Acc   | Epoch  | Best Loss  | Epoch  | Final Acc  | Final Loss
------- | ---------- | -------- | ---------- | ------ | ---------- | ------ | ---------- | ----------
10      | adam       | 6.82     | 0.8209     | 10     | 0.42       | 10     | 0.8209     | 0.42      
10      | adagrad    | 4.83     | 0.7575     | 10     | 0.6055     | 10     | 0.7575     | 0.6055    
10      | adadelta   | 5.93     | 0.2481     | 10     | 0.9737     | 10     | 0.2481     | 0.9737    
10      | adamax     | 5.79     | 0.796      | 4      | 0.4405     | 10     | 0.796      | 0.4405    
10      | rmsprop    | 4.94     | 0.833      | 10     | 0.4034     | 10     | 0.833      | 0.4034    
10      | sgd        | 5.35     | 0.7956     | 7      | 0.4558     | 10     | 0.7951     | 0.4558    
50      | adam       | 24.36    | 0.8648     | 46     | 0.3362     | 49     | 0.8619     | 0.3365    
50      | adagrad    | 23.0     | 0.794      | 50     | 0.5573     | 50     | 0.794      | 0.5573    
50      | adadelta   | 25.51    | 0.796      | 1      | 0.552      | 50     | 0.796      | 0.552     
50      | adamax     | 24.24    | 0.8597     | 50     | 0.347      | 50     | 0.8597     | 0.347     
50      | rmsprop    | 23.77    | 0.862      | 49     | 0.3425     | 49     | 0.8593     | 0.3425    
50      | sgd        | 22.11    | 0.8356     | 49     | 0.3965     | 50     | 0.8355     | 0.3965    
100     | adam       | 48.37    | 0.864      | 62     | 0.3367     | 100    | 0.8624     | 0.3367    
100     | adagrad    | 43.52    | 0.796      | 1      | 0.4655     | 100    | 0.796      | 0.4655    
100     | adadelta   | 49.68    | 0.7674     | 100    | 0.6075     | 100    | 0.7674     | 0.6075    
100     | adamax     | 47.03    | 0.8643     | 98     | 0.3388     | 100    | 0.8637     | 0.3388    
100     | rmsprop    | 44.39    | 0.8664     | 83     | 0.3339     | 100    | 0.8644     | 0.3339    
100     | sgd        | 41.6     | 0.8639     | 83     | 0.3385     | 99     | 0.862      | 0.3387  


Based on the metrics provided, the RMSprop optimizer with 100 Epochs is the absolute best model in terms of raw performance, while Adam with 50 Epochs is the most efficient.
