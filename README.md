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

python ann-optimizers.py
```

## Windows
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Standard installation
pip install tensorflow
pip install pandas numpy scikit-learn

python ann-optimizers.py
```

Epochs  | Optimizer  | Time(s)  | Best Acc   | Epoch  | Best Loss  | Epoch  | Final Acc  | Final Loss
------- | ---------- | -------- | ---------- | ------ | ---------- | ------ | ---------- | ----------
10      | adam       | 1.66     | 0.8279     | 10     | 0.4099     | 10     | 0.8279     | 0.4099    
10      | adagrad    | 1.5      | 0.6186     | 10     | 0.6541     | 10     | 0.6186     | 0.6541    
10      | adadelta   | 1.63     | 0.6256     | 10     | 0.702      | 10     | 0.6256     | 0.702     
10      | adamax     | 1.7      | 0.8073     | 10     | 0.4333     | 10     | 0.8073     | 0.4333    
10      | rmsprop    | 1.53     | 0.8282     | 10     | 0.388      | 10     | 0.8282     | 0.388     
10      | sgd        | 1.37     | 0.8105     | 10     | 0.4323     | 10     | 0.8105     | 0.4323    
50      | adam       | 6.81     | 0.8618     | 46     | 0.3458     | 50     | 0.8608     | 0.3458    
50      | adagrad    | 6.25     | 0.774      | 50     | 0.5202     | 50     | 0.774      | 0.5202    
50      | adadelta   | 6.74     | 0.4579     | 50     | 0.8217     | 50     | 0.4579     | 0.8217    
50      | adamax     | 6.74     | 0.8525     | 50     | 0.358      | 50     | 0.8525     | 0.358     
50      | rmsprop    | 7.09     | 0.8643     | 45     | 0.34       | 50     | 0.8631     | 0.34      
50      | sgd        | 6.21     | 0.8634     | 49     | 0.3429     | 50     | 0.8634     | 0.3429    
100     | adam       | 13.85    | 0.8651     | 81     | 0.3356     | 97     | 0.8622     | 0.3362    
100     | adagrad    | 13.18    | 0.7955     | 78     | 0.4555     | 100    | 0.7951     | 0.4555    
100     | adadelta   | 13.62    | 0.5571     | 100    | 0.6747     | 100    | 0.5571     | 0.6747    
100     | adamax     | 13.63    | 0.8645     | 74     | 0.3393     | 99     | 0.8626     | 0.3393    
100     | rmsprop    | 12.89    | 0.8654     | 73     | 0.338      | 99     | 0.8634     | 0.3382    
100     | sgd        | 11.8     | 0.863      | 90     | 0.3404     | 99     | 0.861      | 0.3406    

The best model is a close tie between RMSprop and Adam at 100 epochs, but RMSprop takes the lead in overall accuracy

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
